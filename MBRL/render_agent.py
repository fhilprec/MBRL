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

from jaxatari.games.jax_seaquest import (
    JaxSeaquest,
    SeaquestRenderer,
    SCALING_FACTOR,
    WIDTH,
    HEIGHT,
)
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper
from actor_critic import ActorCritic, CustomTrainState

from typing import Tuple, List, Dict, Any


def flatten_obs(
    state, single_state: bool = False, is_list=False
) -> Tuple[jnp.ndarray, Any]:
    """
    Flatten the state PyTree into a single array.
    This is useful for debugging and visualization.
    """
    # check whether it is a single state or a batch of states

    if type(state) == list:
        flat_states = []

        for s in state:
            flat_state, _ = jax.flatten_util.ravel_pytree(s)
            flat_states.append(flat_state)
        flat_states = jnp.stack(flat_states, axis=0)  # Shape: (1626, 160)
        print(flat_states.shape)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener
    batch_shape = state.player_x.shape[0]

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)
    return flat_state, unflattener


def load_model(path, train_state=None):
    """Load a previously saved model."""
    import pickle

    # Load the checkpoint using pickle (as shown in your saving code)
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract policy_params from the checkpoint dictionary
    if isinstance(checkpoint, dict) and "policy_params" in checkpoint:
        return checkpoint["policy_params"]
    else:
        # Fallback to original methods if the structure is different
        if train_state is not None:
            with open(path, "rb") as f:
                bytes_data = f.read()
            return flax.serialization.from_bytes(train_state, bytes_data)
        else:
            return checkpoint


def create_network(num_actions, obs_shape, activation="relu"):
    """Create the actor-critic network using the same architecture as training."""
    import flax.linen as nn
    import distrax
    from flax.linen.initializers import constant, orthogonal
    import numpy as np

    def create_actor_critic_network(obs_shape: tuple, action_dim: int):
        """Create an ActorCritic network compatible with your existing implementation."""

        class ActorCritic(nn.Module):
            action_dim: int
            activation: str = "relu"

            @nn.compact
            def __call__(self, x):
                if self.activation == "relu":
                    activation = nn.relu
                else:
                    activation = nn.tanh

                actor_mean = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(x)
                actor_mean = activation(actor_mean)
                actor_mean = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(actor_mean)
                actor_mean = activation(actor_mean)
                actor_mean = nn.Dense(
                    self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                )(actor_mean)
                pi = distrax.Categorical(logits=actor_mean)

                critic = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(x)
                critic = activation(critic)
                critic = nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(critic)
                critic = activation(critic)
                critic = nn.Dense(
                    1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
                )(critic)

                return pi, jnp.squeeze(critic, axis=-1)

        return ActorCritic(action_dim=action_dim, activation=activation)

    # Use the exact same network creation function
    network = create_actor_critic_network(obs_shape, num_actions)

    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(obs_shape)
    params = network.init(rng, init_x)
    return network, params


def reset_environment():
    """Set up and reset the Seaquest environment."""
    env = JaxSeaquest()
    env = AtariWrapper(
        env, sticky_actions=False, episodic_life=False, frame_stack_size=1
    )
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(int(time.time()))
    obs, state = env.reset(rng)
    return env, obs, state


def render_agent(model_path, num_episodes=5, fps=60, record=False, output_path=None):
    """
    Render the agent playing Seaquest with debugging.
    """
    # Initialize environment and renderer
    env, obs, state = reset_environment()

    print(f"Observation type: {type(obs)}")
    flattened_obs = flatten_obs(obs, single_state=True)[0]
    print(f"Flattened observation shape: {flattened_obs.shape}")

    # Create network with correct shapes
    network, dummy_params = create_network(
        env.action_space().n, flattened_obs.shape, activation="relu"
    )

    # Load trained parameters
    print(f"Loading model from: {model_path}")
    loaded_data = load_model(model_path)
    print(f"Loaded data type: {type(loaded_data)}")

    # Extract policy parameters
    if isinstance(loaded_data, dict) and "policy_params" in loaded_data:
        model_params = loaded_data["policy_params"]
        print("Extracted policy_params from checkpoint")
    else:
        model_params = loaded_data
        print("Using loaded data directly as parameters")

    # Debug parameter structures
    def print_param_structure(params, name, max_depth=3, current_depth=0):
        indent = "  " * current_depth
        if isinstance(params, dict):
            print(f"{indent}{name}: dict with keys {list(params.keys())}")
            if current_depth < max_depth:
                for key, value in params.items():
                    print_param_structure(value, key, max_depth, current_depth + 1)
        elif hasattr(params, "shape"):
            print(f"{indent}{name}: array with shape {params.shape}")
        else:
            print(f"{indent}{name}: {type(params)}")

    print("\n=== LOADED PARAMETERS STRUCTURE ===")
    print_param_structure(model_params, "model_params")

    print("\n=== EXPECTED PARAMETERS STRUCTURE ===")
    print_param_structure(dummy_params, "dummy_params")

    # Try to match the parameter structures
    if isinstance(model_params, dict) and isinstance(dummy_params, dict):
        # Check if we need to extract from 'params' key
        if "params" in model_params and "params" in dummy_params:
            final_params = model_params
            print("Using model_params directly (both have 'params' key)")
        elif "params" not in model_params and "params" in dummy_params:
            # Wrap model_params in expected structure
            final_params = {"params": model_params}
            print("Wrapped model_params in 'params' key")
        elif "params" in model_params and "params" not in dummy_params:
            # Extract from model_params
            final_params = model_params["params"]
            print("Extracted from model_params['params']")
        else:
            final_params = model_params
            print("Using model_params as-is")
    else:
        final_params = model_params
        print("Using model_params directly (not dicts)")

    print(f"\n=== FINAL PARAMETERS STRUCTURE ===")
    print_param_structure(final_params, "final_params")

    # Create train state with dummy optimizer
    import optax

    dummy_tx = optax.identity()

    try:
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=final_params,
            tx=dummy_tx,
        )
        print("Successfully created train state")
    except Exception as e:
        print(f"Error creating train state: {e}")
        return

    # Test the network before starting the game loop
    print("\n=== TESTING NETWORK ===")
    try:
        # Test with dummy input first
        test_obs = jnp.zeros_like(flattened_obs)
        pi, value = network.apply(train_state.params, test_obs)
        print(f"Network test with dummy input successful")
        print(f"Policy logits shape: {pi.logits.shape}")
        print(f"Value shape: {value.shape}")

        # Test with actual observation
        pi, value = network.apply(train_state.params, flattened_obs)
        print(f"Network test with real observation successful")
        action = pi.mode()
        print(f"Action: {action}")

    except Exception as e:
        print(f"Network test failed: {e}")
        print("Parameter structure mismatch - cannot proceed")
        return

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
        flattened_obs, _ = flatten_obs(obs, single_state=True)
        pi, _ = network.apply(params, flattened_obs)
        return pi.mode()

    # Main rendering loop
    episode = 0
    steps = 0
    total_reward = 0
    running = True

    print(f"\nStarting to render agent from: {model_path}")
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

            # Take step in environment
            rng, next_rng = jax.random.split(
                jax.random.PRNGKey(int(time.time()) + steps)
            )
            obs, state, reward, done, info = env.step(state, action)
            steps += 1
            total_reward += reward

            # Render current state
            print(f"Atari state type: {type(state.atari_state)}")
            print(
                f"Atari state attributes: {[attr for attr in dir(state.atari_state.env_state) if not attr.startswith('_')]}"
            )
            raster = renderer.render(state.atari_state.env_state)
            # Add frame to recording if enabled
            if record:
                frames.append(np.array(raster * 255, dtype=np.uint8))

            # Update pygame display
            update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)

            # Check for episode termination
            if done:
                print(
                    f"Episode {episode+1}/{num_episodes} finished. Score: {total_reward:.1f}, Steps: {steps}"
                )
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


def update_pygame(screen, raster, scale, width, height):
    """Update the pygame display with the current frame."""
    # Convert raster to a format pygame can use
    raster_np = np.array(raster * 255, dtype=np.uint8)

    # Try without transpose first
    surface = pygame.surfarray.make_surface(raster_np)

    # Scale the surface to the desired size
    scaled_surface = pygame.transform.scale(surface, (width * scale, height * scale))

    # Display the scaled surface
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained Seaquest agent")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run"
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Frames per second for rendering"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record a video of the agent playing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seaquest_agent.mp4",
        help="Output path for the recorded video",
    )

    args = parser.parse_args()

    render_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        fps=args.fps,
        record=args.record,
        output_path=args.output if args.record else None,
    )
