import os
# Add CUDA paths to environment
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
# Remove CPU restriction to allow GPU usage
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Comment this out

import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from jaxatari.games.jax_seaquest import SeaquestRenderer, JaxSeaquest
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper


VERBOSE = True



# Global variable to hold model for evaluate_model function
model = None


def build_world_model():
    def forward(state, action):
        # Flatten the state tree structure to a 1D vector
        flat_state_raw = hk.Flatten()(jax.flatten_util.ravel_pytree(state)[0])
        
        # Get batch size from action shape
        batch_size = action.shape[0] if len(action.shape) > 0 else 1
        
        # Reshape flat_state to ensure correct batch dimension
        if len(flat_state_raw.shape) == 1:
            # No batch dimension, need to reshape
            feature_size = flat_state_raw.shape[0]
            flat_state_full = flat_state_raw.reshape(batch_size, feature_size // batch_size)
        else:
            # Already has batch dimension
            flat_state_full = flat_state_raw
        
        # Exclude the last two values which are random numbers
        flat_state = flat_state_full[:, :-2]
            
        # Convert action to one-hot encoding with correct batch dimension
        action_one_hot = jax.nn.one_hot(action, num_classes=18)
        
        # Concatenate along feature dimension (axis=1)
        inputs = jnp.concatenate([flat_state, action_one_hot], axis=1)
        
        # Feed through MLP
        x = hk.Linear(512)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        
        # Final output layer - output size should match input state size minus the 2 random values
        output_size = flat_state.shape[1]  # Features per example (without the last 2 random values)
        flat_next_state = hk.Linear(output_size)(x)
        
        return flat_next_state
    
    return hk.transform(forward)






#this function does not differentiate between done and not done environments
#maybe this is not a problem since the model should learn to predict the next state regardless of whether the environment is done or not
def collect_experience(game: JaxSeaquest, num_episodes: int = 100, 
                       max_steps_per_episode: int = 1000, num_envs: int = 512) -> Tuple[List, List, List]:

    print(f"Collecting experience data from {num_envs} parallel environments...")
    
    
    
    # Create vectorized reset and step functions stolen from ppo
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
        jax.random.split(rng, n_envs)
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step
    )(jax.random.split(rng, n_envs), env_state, action)
    
    # Initialize storage for collected data
    states = []
    next_states = []
    actions = []
    rewards = []
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)  # Use a fixed seed for reproducibility
    
    # JIT compile the reset and step functions
    jitted_reset = jax.jit(vmap_reset(num_envs))
    jitted_step = jax.jit(vmap_step(num_envs))
    
    # Reset all environments
    rng, reset_rng = jax.random.split(rng) #returns two random keys
    # Reset all environments in parallel
    _, state = jitted_reset(reset_rng)
    

    
    total_steps = 0
    total_episodes = 0
    

    while total_episodes < num_episodes * num_envs:
        # Store the current state
        current_state_repr = jax.tree.map(lambda x: x, state.env_state.env_state)
        
        # Generate random actions for all environments
        rng, action_rng = jax.random.split(rng)
        action_batch = jax.random.randint(action_rng, (num_envs,), 0, 18)
        
        # Step all environments
        rng, step_rng = jax.random.split(rng)
        _, next_state, reward_batch, done_batch, _ = jitted_step(step_rng, state, action_batch)
        

        
        if jnp.any(done_batch):
            # Reset environments that are done
            rng, reset_rng = jax.random.split(rng)
            _, reset_states = jitted_reset(reset_rng)
            
            # Create a function to update only the done states
            def update_where_done(old_state, new_state, done_mask):
                """Update states only where done_mask is True."""
                def where_with_correct_broadcasting(x, y, mask):
                    # Handle broadcasting for different array dimensions
                    if hasattr(x, 'shape') and hasattr(y, 'shape'):
                        if x.ndim > 1:
                            # Create mask with right shape for broadcasting
                            new_shape = (mask.shape[0],) + (1,) * (x.ndim - 1)
                            reshaped_mask = mask.reshape(new_shape)
                            return jnp.where(reshaped_mask, y, x)
                        else:
                            return jnp.where(mask, y, x)
                    else:
                        # For non-array elements
                        return x  # Keep original for simplicity
                
                return jax.tree.map(
                    lambda x, y: where_with_correct_broadcasting(x, y, done_mask),
                    old_state, new_state
                )
            
            # Update only the states that are done
            next_state = update_where_done(next_state, reset_states, done_batch)
        




        # Extract state representation from the new state
        next_state_repr = jax.tree.map(lambda x: x, next_state.env_state.env_state) #not sure here whether there are multiple states or not
        

        # Store experience
        states.append(current_state_repr)
        actions.append(action_batch)
        next_states.append(next_state_repr)
        rewards.append(reward_batch)
        
        # Count completed episodes from done signals
        newly_completed = jnp.sum(done_batch)
        total_episodes += newly_completed
        total_steps += num_envs
        
        
        # Just update state for next iteration, ignoring done environments (inefficient but simple)
        state = next_state

        # Break if we've collected enough episodes
        if total_episodes >= num_episodes * num_envs:
            break
    

    if VERBOSE:
        print(f"Experience collection completed:")
        print(f"- Total steps: {total_steps}")
        print(f"- Total episodes: {total_episodes}")
        print(f"- Total transitions: {len(states)}")
    

    # Convert lists to arrays
    states = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *states)
    actions = jnp.concatenate(actions, axis=0)
    next_states = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *next_states)
    rewards = jnp.concatenate(rewards, axis=0)

    if VERBOSE:
        print(f"Final flattened shape: states: {jax.tree.map(lambda x: x.shape, states)}")
        print(f"Actions shape: {actions.shape}, Rewards shape: {rewards.shape}")

    return states, actions, next_states, rewards



def train_world_model(states, actions, next_states, rewards, 
                     learning_rate=3e-4, batch_size=256, num_epochs=10):
    
    # Initialize model and optimizer
    model = build_world_model()
    optimizer = optax.adam(learning_rate)
    
    # Simple scaling for states to reduce the magnitude
    # Normalize all values to 0-1 range using fixed constants for Atari
    # Most Atari game values are in 0-255 range for pixels
    SCALE_FACTOR = 1/255.0  # Simple scaling factor for pixel values
    
    # Scale states and next_states (simple multiplication is safe for tree structures)
    scaled_states = jax.tree.map(lambda x: x * SCALE_FACTOR, states)
    scaled_next_states = jax.tree.map(lambda x: x * SCALE_FACTOR, next_states)
    
    # Initialize parameters with dummy data
    rng = jax.random.PRNGKey(42)
    dummy_state = jax.tree.map(lambda x: x[:1], scaled_states)  # Take first state
    dummy_action = actions[:1]  # Take first action
    params = model.init(rng, dummy_state, dummy_action)
    opt_state = optimizer.init(params)
    
    # Define loss function with L1 loss (more stable than MSE for high values)
    def loss_fn(params, state_batch, action_batch, next_state_batch):
        # Predict next state
        pred_next_state = model.apply(params, None, state_batch, action_batch)
        
        # Flatten actual next state for comparison
        flat_next_state_raw = jax.flatten_util.ravel_pytree(next_state_batch)[0]
        
        # Get shapes for proper reshaping
        batch_size = pred_next_state.shape[0]
        feature_size = pred_next_state.shape[1]
        
        # Print shape information in the first iteration for debugging
        if hasattr(loss_fn, 'first_call') == False:
            print(f"Prediction shape: {pred_next_state.shape}")
            print(f"Target shape before reshape: {flat_next_state_raw.shape}")
            loss_fn.first_call = True
        
        if len(flat_next_state_raw.shape) == 1:
            # Ensure the total size is consistent before reshaping
            total_size = flat_next_state_raw.shape[0]
            # Adjust feature_size if dimensions don't match
            full_feature_size = total_size // batch_size
            flat_next_state_full = flat_next_state_raw.reshape(batch_size, full_feature_size)
        else:
            flat_next_state_full = flat_next_state_raw
    
        # Exclude the last two values which are random
        flat_next_state = flat_next_state_full[:, :-2]
    
        # Make sure we're comparing the same dimensions
        if flat_next_state.shape[1] != feature_size:
            # Safely reshape while preserving batch dimension
            min_size = min(flat_next_state.shape[1], feature_size)
            flat_next_state = flat_next_state[:, :min_size]
            pred_next_state = pred_next_state[:, :min_size]
    
        # Use L1 loss (mean absolute error) - more stable with high values
        # pred_next_state = jax.tree.map(lambda x: jnp.clip(x, 0, 255).astype(jnp.int32), pred_next_state) -> model not differentiable anymore when doing this here
        mae = jnp.mean(jnp.abs(pred_next_state - flat_next_state))
        return mae
    
    # Set attribute for first call detection
    loss_fn.first_call = False
    
    # Define a single update step (to be JIT-compiled)
    @jax.jit
    def update_step(params, opt_state, state_batch, action_batch, next_state_batch):
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params, state_batch, action_batch, next_state_batch)
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Create batches using the scaled data
    num_batches = len(actions) // batch_size
    batches = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        state_batch = jax.tree.map(lambda x: x[start_idx:end_idx], scaled_states)
        action_batch = actions[start_idx:end_idx]
        next_state_batch = jax.tree.map(lambda x: x[start_idx:end_idx], scaled_next_states)
        batches.append((state_batch, action_batch, next_state_batch))
    
    # Convert to JAX arrays for more efficient processing
    batches = jax.device_put(batches)
    
    # Process multiple batches in parallel for each epoch
    for epoch in range(num_epochs):
        
        # Simple implementation to process batches sequentially but with JIT acceleration
        losses = []
        for batch in batches:
            state_batch, action_batch, next_state_batch = batch
            params, opt_state, loss = update_step(params, opt_state, state_batch, action_batch, next_state_batch)
            losses.append(loss)
        
        # Calculate mean loss for the epoch
        epoch_loss = jnp.mean(jnp.array(losses))
      
        if VERBOSE:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    
    # Return both the trained parameters and the scaling factor for future use
    return params, {"final_loss": epoch_loss, "scale_factor": SCALE_FACTOR}
    
    


import os
import sys
import pygame
import jax
import jax.numpy as jnp
import pickle
import numpy as np
import time
from typing import Tuple, Dict, Any

def compare_real_vs_model(num_steps: int = 1000, render_scale: int = 2):
    """
    Compare the real environment with the world model predictions.
    
    Args:
        num_steps: Number of steps to run the comparison
        render_scale: Scale factor for rendering
    """
    # Initialize game and renderer
    real_env = JaxSeaquest()
    renderer = SeaquestRenderer()
    
    # Check if world model exists
    model_path = "world_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: World model not found at {model_path}")
        return
    
    # Load world model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        dynamics_params = model_data['dynamics_params']
        scale_factor = model_data.get('scale_factor', 1/255.0)
    
    # Initialize model
    world_model = build_world_model()
    
    # Initialize pygame
    pygame.init()
    
    # Set up display - side by side comparison
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20  # Extra 20px for separation
    WINDOW_HEIGHT = HEIGHT * render_scale
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs World Model")
    
    # Initialize environments
    rng = jax.random.PRNGKey(int(time.time()))
    rng, reset_key = jax.random.split(rng)
    real_obs, real_state = real_env.reset(reset_key)
    model_state = real_state  # Start with the same state
    
    # Prepare surfaces for rendering
    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))
    
    # Pre-define a jitted step function for the real environment
    jitted_step = jax.jit(real_env.step)
    
    # Define a world model prediction function
    def predict_next_state(params, state, action, scale_factor=1/255.0):
        """Predict next state using the world model"""
        # Scale the state (normalization)
        scaled_state = jax.tree.map(lambda x: x * scale_factor, state)
        
        # Convert the single action to a batched action
        if not isinstance(action, jnp.ndarray) or action.ndim == 0:
            action = jnp.array([action])  # Add batch dimension
    
        # Get prediction from model
        pred_next_state = world_model.apply(params, None, scaled_state, action)
        
        # Get flat representation of state and pytree structure
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    
        # Make a copy of the flat state to modify
        new_flat_state = jnp.array(flat_state)  # Create a copy
    
        # Calculate how many elements to replace
        # Exclude the last two random values
        elements_to_replace = len(flat_state) - 2
        pred_size = pred_next_state.shape[1]  # Second dimension has the features
    
        # Make sure we don't overrun the array bounds
        copy_size = min(elements_to_replace, pred_size)
    
        # Replace all but the last two values with our prediction
        # Need to reshape prediction to match the expected flat shape
        new_flat_state = new_flat_state.at[:copy_size].set(
            (pred_next_state[0][:copy_size] / scale_factor)
        )
    
        # Use the unflattener function returned by ravel_pytree
        return unflattener(new_flat_state)
    # For efficiency
    jitted_predict = jax.jit(predict_next_state)
    
    # Main loop
    running = True
    step_count = 0
    clock = pygame.time.Clock()
    
    while running and step_count < num_steps:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Random action
        rng, action_key = jax.random.split(rng)
        action = jax.random.randint(action_key, shape=(), minval=0, maxval=18)
        if step_count % 4 == 0: #force agent to swim down to extent episode
            action = 5
        # Step real environment
        real_obs, real_state, real_reward, real_done, real_info = jitted_step(real_state, action)
        
        # Step world model
        model_state = jitted_predict(dynamics_params, model_state, action, scale_factor)
        model_obs = real_env._get_observation(model_state)
        
        # Render real environment
        real_raster = renderer.render(real_state)
        # Convert JAX array to numpy and then to pygame surface
        real_img = np.array(real_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(real_surface, real_img)
        
        # Render model environment
        # Convert JAX array to numpy and then to pygame surface     
        # 
        # clip values to sensible range but preserve original type
        model_state = jax.tree.map(lambda x: jnp.clip(x, 0, 255).astype(jnp.int32), model_state)
        
        model_raster = renderer.render(model_state)
        model_img = np.array(model_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(model_surface, model_img)

        print(real_state)
        print(model_state)

        
        # Scale and blit to screen
        screen.fill((0, 0, 0))
        # Left side: real environment
        scaled_real = pygame.transform.scale(real_surface, (WIDTH * render_scale, HEIGHT * render_scale))
        screen.blit(scaled_real, (0, 0))
        
        # Right side: world model
        scaled_model = pygame.transform.scale(model_surface, (WIDTH * render_scale, HEIGHT * render_scale))
        screen.blit(scaled_model, (WIDTH * render_scale + 20, 0))
        
        # Add labels
        font = pygame.font.SysFont(None, 24)
        real_text = font.render("Real Environment", True, (255, 255, 255))
        model_text = font.render("World Model", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        
        # Update display
        pygame.display.flip()
        
        # Reset if either environment is done
        if real_done:
            rng, reset_key = jax.random.split(rng)
            real_obs, real_state = real_env.reset(reset_key)
            model_state = real_state  # Sync model state with real state
        
        step_count += 1
        clock.tick(30)
    
    pygame.quit()
    print("Comparison completed")


if __name__ == "__main__":
    # Initialize the game environment
    game = JaxSeaquest()
    env = AtariWrapper(game, sticky_actions=False, episodic_life=False)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    
    # Train the world model
    save_path = "world_model.pkl"

    # Set the global model variable
    model = build_world_model()




    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data['dynamics_params']
    else:
        print("No existing model found. Training a new model...")

        # Define a file path for the experience data
        experience_data_path = "experience_data.pkl"

        # Check if experience data file exists
        if os.path.exists(experience_data_path):
            print(f"Loading existing experience data from {experience_data_path}...")
            with open(experience_data_path, 'rb') as f:
                saved_data = pickle.load(f)
                states = saved_data['states']
                actions = saved_data['actions']
                next_states = saved_data['next_states']
                rewards = saved_data['rewards']
        else:
            print("No existing experience data found. Collecting new experience data...")
            # Collect experience data
            states, actions, next_states, rewards = collect_experience(
                game,
                num_episodes=1,
                max_steps_per_episode=10000,
                num_envs=512
            )
            
            # Save the collected experience data
            with open(experience_data_path, 'wb') as f:
                pickle.dump({
                    'states': states,
                    'actions': actions,
                    'next_states': next_states,
                    'rewards': rewards
                }, f)
            print(f"Experience data saved to {experience_data_path}")

        #train world model
        dynamics_params, training_info = train_world_model(
            states,
            actions,
            next_states,
            rewards,
            learning_rate=3e-4,
            batch_size=4096,
            num_epochs=1000,
        )

        # Save the model and scaling factor
        with open(save_path, 'wb') as f:
            pickle.dump({
                'dynamics_params': dynamics_params,
                'scale_factor': training_info['scale_factor']
            }, f)
        print(f"Model saved to {save_path}")

    # # Evaluate the model
    # eval_mse = evaluate_model(dynamics_params, env, model)
    # print(f"Final evaluation MSE: {eval_mse:.6f}")


    #  # Then visualize the predictions
    # print("Visualizing model predictions vs. actual gameplay...")
    # stats = visualize_predictions(dynamics_params, env, model, num_steps=500, delay=0.05)
    # print(f"Visualization stats: {stats}")

    compare_real_vs_model(num_steps=5000, render_scale=3)