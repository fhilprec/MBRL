import jax
import jax.numpy as jnp
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pygame
import os
import argparse
import flax
from flax.training.train_state import TrainState
from typing import Any

from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer, SCALING_FACTOR, WIDTH, HEIGHT
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper
from actor_critic import ActorCritic, CustomTrainState


def load_model(path, train_state=None):
    """Load a previously saved model."""
    with open(path, 'rb') as f:
        bytes_data = f.read()
    
    if train_state is not None:
        return flax.serialization.from_bytes(train_state, bytes_data)
    else:
        return flax.serialization.msgpack_restore(bytes_data)


def create_network(num_actions, obs_shape, activation="relu"):
    """Create the actor-critic network."""
    network = ActorCritic(num_actions, activation=activation)
    
    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(obs_shape)
    params = network.init(rng, init_x)
    return network, params


def reset_environment():
    """Set up and reset the Seaquest environment."""
    env = JaxSeaquest()
    env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(int(time.time()))
    obs, state = env.reset(rng)
    return env, obs, state


def render_agent(model_path, num_episodes=5, fps=60, record=False, output_path=None):
    """
    Render the agent playing Seaquest.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to play
        fps: Frames per second for rendering
        record: Whether to record a video
        output_path: Output path for the recorded video
    """
    # Initialize environment and renderer
    env, obs, state = reset_environment()
    
    # Create network with correct shapes
    network, _ = create_network(
        env.action_space().n, 
        env.observation_space().shape, 
        activation="relu"
    )
    
    # Load trained parameters
    loaded_params = load_model(model_path)
    
    print(f"Loaded model from: {model_path}")
    
    # Fix the parameters structure - extract the inner params if needed
    if "params" in loaded_params and isinstance(loaded_params["params"], dict):
        model_params = loaded_params["params"]
    else:
        model_params = loaded_params
    
    import optax
    # Create a dummy optimizer that doesn't do anything
    dummy_tx = optax.identity()

    # Create train state with properly structured params
    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=model_params,  # Use the fixed params structure
        tx=dummy_tx,
    )
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    pygame.display.set_caption(f"JAX Seaquest - {os.path.basename(model_path)}")
    clock = pygame.time.Clock()
    
    # Set up renderer
    renderer = SeaquestRenderer()
    
    # Set up recording if needed
    frames = []
    
    # Jit the policy function
    @jax.jit
    def get_action(params, obs):
        pi, _ = network.apply(params, obs)
        return pi.mode()
    
    # Main rendering loop
    episode = 0
    steps = 0
    total_reward = 0
    running = True
    
    print(f"Rendering agent from: {model_path}")
    print("Controls: Q to quit, P to pause/unpause")
    
    # Game loop with rendering
    paused = False
    
    while running and episode < num_episodes:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    print("Paused" if paused else "Unpaused")
        
        if not paused:
            # Get action from policy
            action = get_action(train_state.params, obs)
            
            # Take step in environment - corrected order of arguments
            rng, next_rng = jax.random.split(jax.random.PRNGKey(int(time.time()) + steps))
            obs, state, reward, done, info = env.step(rng, state, action)
            steps += 1
            total_reward += reward
            
            # Render current state
            # raster = renderer.render(state)
            raster = renderer.render(state.env_state.env_state)
            
            # Add frame to recording if enabled
            if record:
                frames.append(np.array(raster * 255, dtype=np.uint8))
            
            # Update pygame display
            update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
            
            # Check for episode termination
            if done:
                print(f"Episode {episode+1}/{num_episodes} finished. Score: {total_reward:.1f}, Steps: {steps}")
                episode += 1
                steps = 0
                total_reward = 0
                
                if episode < num_episodes:
                    # Reset environment for next episode
                    rng = jax.random.PRNGKey(int(time.time()) + episode)
                    obs, state = env.reset(rng)
        
        # Control rendering speed
        clock.tick(fps)
    
    pygame.quit()
    
    # Save video if recording was enabled
    if record and frames and output_path:
        save_video(frames, output_path, fps)
        print(f"Video saved to {output_path}")


def update_pygame(screen, raster, scale, width, height):
    """Update the pygame display with the current frame."""
    # Convert raster to a format pygame can use
    raster_np = np.array(raster * 255, dtype=np.uint8)
    surface = pygame.surfarray.make_surface(raster_np.transpose(1, 0, 2))
    
    # Scale the surface to the desired size
    scaled_surface = pygame.transform.scale(surface, (width * scale, height * scale))
    
    # Display the scaled surface
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()


def save_video(frames, filename, fps=60):
    """Save a sequence of frames as an MP4 video."""
    print(f"Saving video with {len(frames)} frames at {fps} fps...")
    
    # Create figure for plotting frames
    fig, ax = plt.subplots(figsize=(frames[0].shape[1]/100, frames[0].shape[0]/100))
    ax.set_axis_off()
    
    # Plot the first frame
    img = ax.imshow(frames[0])
    
    def init():
        img.set_data(frames[0])
        return (img,)
    
    def animate(i):
        img.set_data(frames[i])
        return (img,)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(frames), interval=1000/fps, blit=True
    )
    
    # Save as MP4
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='JAX Seaquest Agent'), bitrate=1800)
    anim.save(filename, writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained Seaquest agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for rendering")
    parser.add_argument("--record", action="store_true", help="Record a video of the agent playing")
    parser.add_argument("--output", type=str, default="seaquest_agent.mp4", help="Output path for the recorded video")
    
    args = parser.parse_args()
    
    render_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        fps=args.fps,
        record=args.record,
        output_path=args.output if args.record else None
    )