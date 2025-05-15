import jax
import jax.numpy as jnp
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pygame
import os

# Import your actor-critic model and environment
from actor_critic import build_actor_critic
# Import with correct path structure
from src.jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer

def load_actor_critic(path):
    """Load the trained actor-critic model."""
    try:
        with open(path, 'rb') as f:
            saved_data = pickle.load(f)
        
        actor, _ = build_actor_critic()
        return actor, saved_data['actor_params']
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find model file at {path}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def render_episode(game, actor, actor_params, renderer, max_steps=1000, render_mode='human', fps=30):
    """Render an episode with the trained agent."""
    frames = []
    total_reward = 0
    
    # Initialize the environment
    obs, state = game.reset()
    flat_obs = game.obs_to_flat_array(obs)
    
    # Create jitted step function
    jitted_step = jax.jit(game.step)
    
    # Initialize pygame if using human mode
    if render_mode == 'human':
        pygame.init()
        screen = pygame.display.set_mode((160*3, 210*3))
        pygame.display.set_caption("Seaquest Agent")
        clock = pygame.time.Clock()
    
    running = True
    for step in range(max_steps):
        if not running:
            break
            
        # Get action from policy
        logits = actor.apply(actor_params, None, flat_obs)
        action = jnp.argmax(logits).item()

        print(f"Step {step}: Action {action}, Logits: {logits}")
        
        # Take step in environment
        next_obs, next_state, reward, done, _ = jitted_step(state, action)
        total_reward += reward
        
        # Render AFTER taking the step so we see the result of the action
        if render_mode in ['rgb_array', 'human']:
            try:
                # Render the current state
                frame = renderer.render(next_state)
                
                if render_mode == 'rgb_array':
                    frames.append(np.array(frame))
                else:  # render_mode == 'human'
                    # Convert JAX array to numpy for pygame
                    frame_np = np.array(frame)
                    
                    # Create a surface and scale it
                    surf = pygame.surfarray.make_surface(frame_np.transpose(1, 0, 2))
                    scaled_surf = pygame.transform.scale(surf, (160*3, 210*3))
                    
                    # Display on screen
                    screen.blit(scaled_surf, (0, 0))
                    pygame.display.flip()
                    clock.tick(fps)
                    
                    # Check for exit events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
            except Exception as e:
                print(f"Error rendering frame: {e}")
                if step == 0:  # Only print once
                    print("Continuing without rendering...")
        
        # Update for next iteration
        obs, state = next_obs, next_state
        flat_obs = game.obs_to_flat_array(obs)
        
        if done:
            print(f"Episode ended after {step} steps")
            break
    
    # Clean up pygame if initialized
    if render_mode == 'human':
        pygame.quit()
    
    print(f"Episode completed with total reward: {total_reward}")
    return frames

def create_animation(frames, filename=None, fps=30):
    """Create an animation from frames."""
    if not frames or len(frames) == 0:
        print("No frames to create animation")
        return None
    
    print(f"Creating animation with {len(frames)} frames")
    
    # Check frame data
    print(f"Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
    
    plt.figure(figsize=(frames[0].shape[1]/72, frames[0].shape[0]/72), dpi=72)
    plt.axis('off')
    
    patch = plt.imshow(frames[0])
    
    def animate(i):
        patch.set_data(frames[i])
        return [patch]
    
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=1000/fps)
    
    if filename:
        print(f"Saving animation to {filename}...")
        try:
            # Try different writers
            writers = ['ffmpeg', 'imagemagick', 'pillow']
            for writer in writers:
                try:
                    print(f"Trying to save with {writer}...")
                    anim.save(filename, fps=fps, writer=writer)
                    print(f"Animation saved to {filename} using {writer}")
                    break
                except Exception as e:
                    print(f"Failed with {writer}: {e}")
            else:
                print("All specific writers failed, trying default...")
                anim.save(filename, fps=fps)
                print(f"Animation saved to {filename} with default writer")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            
            # Last resort: save individual frames as images
            print("Saving individual frames...")
            os.makedirs("frames", exist_ok=True)
            for i, frame in enumerate(frames):
                plt.imsave(f"frames/frame_{i:04d}.png", frame)
            print(f"Saved {len(frames)} individual frames to 'frames/' directory")
    
    plt.close()
    return None

if __name__ == "__main__":
    # Initialize game
    print("Initializing Seaquest environment...")
    game = JaxSeaquest()
    
    # Initialize the renderer
    print("Initializing Seaquest renderer...")
    try:
        renderer = SeaquestRenderer()
        print("Renderer initialized successfully")
    except Exception as e:
        print(f"Error initializing renderer: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    try:
        # Load trained actor-critic model
        print("Loading trained actor-critic model...")
        model_path = 'actor_critic_model.pkl'
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Looking for model files...")
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pkl'):
                        print(f"Found: {os.path.join(root, file)}")
            
            model_path = input("Enter the path to your model file: ")
            if not model_path:
                model_path = 'actor_critic_model.pkl'
                
        actor, actor_params = load_actor_critic(model_path)
        
        # Choose rendering mode
        render_mode = 'human'  # or 'human' for interactive display
        print(f"Rendering episode with mode: {render_mode}")
        
        # Render an episode
        frames = render_episode(game, actor, actor_params, renderer, render_mode=render_mode)
        
        if render_mode == 'rgb_array' and frames and len(frames) > 0:
            # Create and save animation
            print("Creating animation...")
            create_animation(frames, filename="agent_gameplay.mp4", fps=30)
        else:
            print("No frames were captured.")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()