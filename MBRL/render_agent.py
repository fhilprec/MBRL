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
        print(f"Attempting to load model from {path}")
        with open(path, 'rb') as f:
            saved_data = pickle.load(f)
            
        print(f"Model loaded. Available keys in saved data: {list(saved_data.keys())}")
        
        # Import the actual ActorCritic class used in training
        from actor_critic import ActorCritic
        
        # Create the original network to match the parameter structure
        network = ActorCritic(action_dim=18)
        
        # Check different possible parameter key formats
        if 'params' in saved_data:
            params = saved_data['params']
            print("Found 'params' key in saved model")
        elif 'runner_state' in saved_data:
            # This format is from the PPO implementation
            print("Found 'runner_state' key in saved model")
            train_state = saved_data['runner_state'][0]
            if hasattr(train_state, 'params'):
                params = train_state.params
                print("Found params in train_state")
            else:
                print("Available attributes in train_state:", dir(train_state))
                raise KeyError("Could not find params in train_state")
        elif isinstance(saved_data, dict) and len(saved_data) > 0:
            # Look for any dictionary that might contain parameters
            print("Looking for parameters in alternative dictionary structure")
            
            # If we have a dictionary with a single key, try using that
            if len(saved_data) == 1:
                key = list(saved_data.keys())[0]
                params = saved_data[key]
                print(f"Using value from key '{key}' as parameters")
            else:
                # Print the structure to help diagnose
                for key, value in saved_data.items():
                    print(f"Key: {key}, Type: {type(value)}")
                raise KeyError("Multiple keys found but none matched expected parameter structure")
        else:
            raise KeyError("Could not find parameters in saved model")
            
        # Create a dummy observation to initialize the separate actor
        dummy_obs = jnp.zeros((241,))  # Match your observation size
        
        # This will extract just the actor part of the network
        def get_actor_prediction(params, obs):
            pi, _ = network.apply(params, obs)
            return pi.logits
        
        # Create a wrapper function to use with the renderer
        def actor_fn(params, obs):
            return get_actor_prediction(params, obs)
        
        return actor_fn, params
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's get more info about the model file
        try:
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024
                print(f"Model file exists, size: {file_size:.2f} KB")
                
                # Peek at the file structure
                with open(path, 'rb') as f:
                    peek_data = pickle.load(f)
                    if isinstance(peek_data, dict):
                        print("Model contains a dictionary with keys:", list(peek_data.keys()))
                    else:
                        print(f"Model contains a {type(peek_data)} (not a dictionary)")
            else:
                print(f"Model file does not exist at: {path}")
                
                # List available pickle files
                print("\nAvailable pickle files:")
                for file in os.listdir(os.path.dirname(path) or "."):
                    if file.endswith(".pkl"):
                        full_path = os.path.join(os.path.dirname(path) or ".", file)
                        print(f"- {full_path} ({os.path.getsize(full_path)/1024:.2f} KB)")
        except Exception as peek_error:
            print(f"Error examining model file: {peek_error}")
        
        raise Exception(f"Failed to load model: {e}")


def render_episode(game, actor_fn, actor_params, renderer, max_steps=10000, render_mode='human', fps=30):
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
        # Increase window size (swapped width/height for 90 degree rotation)
        # Original was 160*3 x 210*3, making it larger and rotated
        display_width = 210*4  # Larger and swapped
        display_height = 160*4  # Larger and swapped
        screen = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("Seaquest Agent (Rotated)")
        clock = pygame.time.Clock()
    
    running = True
    for step in range(max_steps):
        if not running:
            break
            
        # Get action from policy using our wrapper function
        logits = actor_fn(actor_params, flat_obs)
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
                    
                    # Create surface from the frame
                    surf = pygame.surfarray.make_surface(frame_np.transpose(1, 0, 2))
                    
                    # Rotate 90 degrees clockwise
                    rotated_surf = pygame.transform.rotate(surf, 270)  # 270° = 90° clockwise
                    
                    # Scale to new size (4x larger)
                    scaled_surf = pygame.transform.scale(rotated_surf, (display_width, display_height))
                    
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
        # Look for the PPO model file first
        model_path = 'ppo_agent_real_env.pkl'
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, trying model_world...")
            model_path = 'ppo_agent_world_model.pkl'
        
        if not os.path.exists(model_path):
            print(f"Standard models not found, looking for model files...")
            model_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pkl'):
                        model_files.append(os.path.join(root, file))
                        print(f"Found: {os.path.join(root, file)}")
            
            if model_files:
                model_path = model_files[0]  # Pick the first one by default
                print(f"Using model at: {model_path}")
            else:
                model_path = input("Enter the path to your model file: ")
                if not model_path:
                    raise FileNotFoundError("No model file specified")
                
        actor, actor_params = load_actor_critic(model_path)
        print(model_path)
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