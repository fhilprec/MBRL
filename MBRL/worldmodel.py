import os
# Add CUDA paths to environment
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

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

def render_trajectory(states, num_frames: int = 100, render_scale: int = 3, delay: int = 50):
    """
    Render a trajectory of states in a single window.
    
    Args:
        states: PyTree containing the collected states to visualize
        num_frames: Maximum number of frames to show
        render_scale: Scaling factor for rendering
        delay: Milliseconds to delay between frames
    """
    # Initialize pygame and create renderer
    import pygame
    import time
    pygame.init()
    
    # Get renderer for the game
    renderer = SeaquestRenderer()
    
    # Set up display
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale
    WINDOW_HEIGHT = HEIGHT * render_scale
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("State Trajectory Visualization")
    
    # Prepare surface for rendering
    surface = pygame.Surface((WIDTH, HEIGHT))
    
    # Create font for rendering text
    font = pygame.font.SysFont(None, 24)
    
    # Determine how many states to render
    if isinstance(states, dict) or hasattr(states, 'env_state'):
        # Single state case
        total_frames = 1
    else:
        # Get the size from the first field in the PyTree
        first_field = jax.tree_util.tree_leaves(states)[0]
        total_frames = first_field.shape[0] if hasattr(first_field, 'shape') else 1
    
    frames_to_show = min(total_frames, num_frames)
    
    print(f"Rendering trajectory with {frames_to_show} frames...")
    
    # Main loop
    running = True
    frame_idx = 0
    
    while running and frame_idx < frames_to_show:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get the current state
        if total_frames > 1:
            # Extract the frame_idx'th state from the batch
            current_state = jax.tree.map(lambda x: x[frame_idx] if hasattr(x, 'shape') and x.shape[0] > frame_idx else x, states)
        else:
            current_state = states
        
        # Render the state
        try:
            # Try to render the current state
            raster = renderer.render(current_state)
            img = np.array(raster * 255, dtype=np.uint8)
            pygame.surfarray.blit_array(surface, img)
            
            # Scale and display
            screen.fill((0, 0, 0))
            scaled_surface = pygame.transform.scale(surface, (WIDTH * render_scale, HEIGHT * render_scale))
            screen.blit(scaled_surface, (0, 0))
            
            # Add frame number indicator
            frame_text = font.render(f"Frame: {frame_idx + 1}/{frames_to_show}", True, (255, 255, 255))
            screen.blit(frame_text, (10, 10))
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            # If there's an error, try the next frame
            frame_idx += 1
            continue
        
        # Delay between frames
        pygame.time.wait(delay)
        
        frame_idx += 1
    
    # Keep window open for a brief moment at the end
    if running:
        pygame.time.wait(1000)
    
    pygame.quit()
    print(f"Rendered {frame_idx} frames from trajectory")

def build_world_model():
    def forward(state, action):
        # Flatten the state tree structure to a 1D vector
        # The AtariWrapper with frame_stack_size=4 already gives us 4 stacked frames in the state
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
        
        # Feed through MLP - larger network since we now have 4x more input (4 stacked frames)
        # Much larger network
        x = hk.Linear(1024)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(1024)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        
        # Final output layer - output size should match input state size minus the 2 random values
        # But since we have 4 stacked frames as input, we predict only 1 frame (1/4 of input size)
        single_frame_size = flat_state.shape[1] // 4  # Predict single frame from 4 stacked frames
        flat_next_state = hk.Linear(single_frame_size)(x)
        
        return flat_next_state
    
    return hk.transform(forward)

def collect_experience(env, num_episodes: int = 100, 
                       max_steps_per_episode: int = 1000, num_envs: int = 512) -> Tuple[List, List, List]:

    print(f"Collecting experience data from {num_envs} parallel environments...")
    print("Note: AtariWrapper provides 4 stacked frames automatically")
    
    # Create vectorized reset and step functions
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
    rng = jax.random.PRNGKey(42)
    
    # JIT compile the reset and step functions
    jitted_reset = jax.jit(vmap_reset(num_envs))
    jitted_step = jax.jit(vmap_step(num_envs))
    
    # Reset all environments
    rng, reset_rng = jax.random.split(rng)
    _, state = jitted_reset(reset_rng)
    
    total_steps = 0
    total_episodes = 0
    
    while total_episodes < num_episodes * num_envs:
        # Store the current state - AtariWrapper state structure is different
        # For AtariWrapper, the state contains the stacked frames directly
        current_state_repr = jax.tree.map(lambda x: x, state.env_state)
        
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
        next_state_repr = jax.tree.map(lambda x: x, next_state.env_state)
        
        # Store experience
        states.append(current_state_repr)
        actions.append(action_batch)
        next_states.append(next_state_repr)
        rewards.append(reward_batch)
        
        # Count completed episodes from done signals
        newly_completed = jnp.sum(done_batch)
        total_episodes += newly_completed
        total_steps += num_envs
        
        # Update state for next iteration
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
    

    # if VERBOSE:
    #     render_trajectory(states, num_frames=100, render_scale=3, delay=50)

    # Initialize model and optimizer
    model = build_world_model()



    # optimizer = optax.adam(learning_rate)

    #TEST use different optimizer
    # Add learning rate schedule
    lr_schedule = optax.exponential_decay(
        init_value=1e-4,
        transition_steps=500,  # Faster decay
        decay_rate=0.95  # Less aggressive per-step decay
    )

    # Use the schedule in your optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )

    
    # Simple scaling for states to reduce the magnitude
    SCALE_FACTOR = 1/255.0
    
    # Scale states and next_states
    scaled_states = jax.tree.map(lambda x: x * SCALE_FACTOR, states)
    scaled_next_states = jax.tree.map(lambda x: x * SCALE_FACTOR, next_states)
    
    # Initialize parameters with dummy data
    rng = jax.random.PRNGKey(42)
    dummy_state = jax.tree.map(lambda x: x[:1], scaled_states)
    dummy_action = actions[:1]
    params = model.init(rng, dummy_state, dummy_action)
    opt_state = optimizer.init(params)
    
    # Define loss function
    def loss_fn(params, state_batch, action_batch, next_state_batch):
        # Predict next state (single frame from 4 stacked frames)
        pred_next_state = model.apply(params, None, state_batch, action_batch)
        
        # Flatten actual next state for comparison
        flat_next_state_raw = jax.flatten_util.ravel_pytree(next_state_batch)[0]
        
        # Get shapes for proper reshaping
        batch_size = pred_next_state.shape[0]
        pred_feature_size = pred_next_state.shape[1]
        
        # Print shape information in the first iteration for debugging
        if hasattr(loss_fn, 'first_call') == False:
            print(f"Prediction shape: {pred_next_state.shape}")
            print(f"Target shape before reshape: {flat_next_state_raw.shape}")
            loss_fn.first_call = True
        
        if len(flat_next_state_raw.shape) == 1:
            # Ensure the total size is consistent before reshaping
            total_size = flat_next_state_raw.shape[0]
            full_feature_size = total_size // batch_size
            flat_next_state_full = flat_next_state_raw.reshape(batch_size, full_feature_size)
        else:
            flat_next_state_full = flat_next_state_raw
    
        # Exclude the last two values which are random
        flat_next_state = flat_next_state_full[:, :-2]
        
        # Since we're predicting single frame from 4 stacked frames,
        # we need to extract the most recent frame from the target
        # The next_state contains 4 stacked frames, we want the newest one
        single_frame_size = flat_next_state.shape[1] // 4
        target_single_frame = flat_next_state[:, -single_frame_size:]  # Last frame
    
        # Make sure we're comparing the same dimensions
        if target_single_frame.shape[1] != pred_feature_size:
            min_size = min(target_single_frame.shape[1], pred_feature_size)
            target_single_frame = target_single_frame[:, :min_size]
            pred_next_state = pred_next_state[:, :min_size]
    
        # Use L1 loss (mean absolute error)
        mae = jnp.mean(jnp.abs(pred_next_state - target_single_frame))
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

        
      
        if VERBOSE and (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    
    # Return both the trained parameters and the scaling factor for future use
    return params, {"final_loss": epoch_loss, "scale_factor": SCALE_FACTOR}

def compare_real_vs_model(num_steps: int = 1000, render_scale: int = 2):
    """
    Compare the real environment with the world model predictions.
    Now works with 4 stacked frames from AtariWrapper.
    """
    # Initialize base game and wrapped environment
    base_game = JaxSeaquest()
    # Use AtariWrapper with frame stacking (default frame_stack_size=4)
    real_env = AtariWrapper(base_game, sticky_actions=False, episodic_life=False, frame_stack_size=4)
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
    import pygame
    import time
    pygame.init()
    
    # Set up display - side by side comparison
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs World Model (AtariWrapper Frame Stack)")
    
    # Initialize environments
    rng = jax.random.PRNGKey(int(time.time()))
    rng, reset_key = jax.random.split(rng)
    real_obs, real_state = real_env.reset(reset_key)
    model_state = real_state  # Start with the same state
    
    # Prepare surfaces for rendering
    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))
    
    # Pre-define a jitted step function for the real environment
    def step_with_rng(rng, state, action):
        return real_env.step(rng, state, action)
    jitted_step = jax.jit(step_with_rng)
    
    # Define a world model prediction function
    def predict_next_state(params, state, action, scale_factor=1/255.0):
        """Predict next state using the world model with 4 stacked frames"""
        # Scale the state (normalization)
        scaled_state = jax.tree.map(lambda x: x * scale_factor, state.env_state)
        
        # Convert the single action to a batched action
        if not isinstance(action, jnp.ndarray) or action.ndim == 0:
            action = jnp.array([action])
    
        # Get prediction from model (single frame prediction)
        pred_next_frame = world_model.apply(params, None, scaled_state, action)
        
        # Get flat representation of current state and pytree structure
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state.env_state)
    
        # Make a copy of the flat state to modify
        new_flat_state = jnp.array(flat_state)
        
        # The state contains 4 stacked frames. We want to:
        # 1. Shift the 4 frames: [frame1, frame2, frame3, frame4] -> [frame2, frame3, frame4, predicted_frame]
        # 2. Replace the last frame with our prediction
        
        # Calculate frame sizes
        total_state_size = len(flat_state) - 2  # Exclude 2 random values
        single_frame_size = total_state_size // 4
        pred_size = pred_next_frame.shape[1]
        
        # Make sure we don't overrun bounds
        copy_size = min(single_frame_size, pred_size)
        
        # Shift frames: move frames 1,2,3 to positions 0,1,2
        for i in range(3):
            start_old = (i + 1) * single_frame_size
            end_old = start_old + copy_size
            start_new = i * single_frame_size  
            end_new = start_new + copy_size
            new_flat_state = new_flat_state.at[start_new:end_new].set(
                new_flat_state[start_old:end_old]
            )
        
        # Set the last frame to our prediction
        last_frame_start = 3 * single_frame_size
        last_frame_end = last_frame_start + copy_size
        new_flat_state = new_flat_state.at[last_frame_start:last_frame_end].set(
            (pred_next_frame[0][:copy_size] / scale_factor)
        )
    
        # Reconstruct the state
        new_env_state = unflattener(new_flat_state)
        
        # Return the full state structure
        return state.replace(env_state=new_env_state)
    
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
        if step_count % 4 == 0:  # Force agent to swim down to extend episode
            action = 5
        
        # Step real environment
        rng, step_rng = jax.random.split(rng)
        real_obs, real_state, real_reward, real_done, real_info = jitted_step(step_rng, real_state, action)
        
        # Step world model
        model_state = jitted_predict(dynamics_params, model_state, action, scale_factor)
        
        # Clip model state values for rendering
        model_state = jax.tree.map(lambda x: jnp.clip(x, 0, 255).astype(jnp.int32), model_state)
        
        if VERBOSE and step_count % 100 == 0:
            print(f"Step {step_count}: Real vs Model state comparison")
        
        # For rendering, we need to get the base game state
        # AtariWrapper stores the base game state differently - we need to extract it properly
        # Let's get the most recent frame from the stacked frames for rendering
        
        # Extract base game state for rendering from the AtariWrapper
        real_base_state = real_state.env_state  # This should be the stacked frames
        model_base_state = model_state.env_state
        
        # The renderer expects the base game state, but we have stacked frames
        # We need to extract just the most recent frame for rendering
        # This requires understanding the AtariWrapper's frame stacking structure
        
        # For now, let's try rendering the full state and see what happens

        print("---------------------------------------------------------------Real State-------------------------------------------------")
        print(real_base_state)
        print("---------------------------------------------------------------Model State------------------------------------------------")
        print(model_base_state)


        # Render real environment - try to extract the most recent frame
        real_raster = renderer.render(real_base_state)
        real_img = np.array(real_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(real_surface, real_img)


        model_raster = renderer.render(model_base_state)
        model_img = np.array(model_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(model_surface, model_img)

        
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
        model_text = font.render("World Model (4 Frames)", True, (255, 255, 255))
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
    # Initialize the game environment with AtariWrapper (includes frame stacking)
    game = JaxSeaquest()
    env = AtariWrapper(game, sticky_actions=False, episodic_life=False, frame_stack_size=4)
    
    # Train the world model
    save_path = "world_model.pkl"

    # Set the global model variable
    model = build_world_model()

    '''
    # First, let's understand the state structure by doing a single step
    print("Analyzing state structure...")
    rng = jax.random.PRNGKey(42)
    rng, reset_key = jax.random.split(rng)
    initial_obs, initial_state = env.reset(reset_key)
    
    print("AtariWrapper state structure:")
    print(f"Full state: {initial_state}")
    print(f"State type: {type(initial_state)}")
    print(f"State env_state: {initial_state.env_state}")
    print(f"env_state type: {type(initial_state.env_state)}")
    
    # Get all leaf values from the initial state tree structure
    state_leaves = jax.tree_util.tree_leaves(initial_state.env_state)
    
    # Convert any JAX arrays to lists and flatten everything into a single list
    all_values = []
    for leaf in state_leaves:
        if isinstance(leaf, jnp.ndarray):
            all_values.extend(leaf.flatten().tolist())
        else:
            all_values.append(leaf)
    
    print("All state values as a flat list:")
    print(f"Number of values in the wrapped state: {len(all_values)}")
    
    # Compare with base game
    base_obs, base_state = game.reset(reset_key)
    base_leaves = jax.tree_util.tree_leaves(base_state)
    base_values = []
    for leaf in base_leaves:
        if isinstance(leaf, jnp.ndarray):
            base_values.extend(leaf.flatten().tolist())
        else:
            base_values.append(leaf)
    
    print(f"Number of values in base game state: {len(base_values)}")
    print(f"Frame stacking multiplier: {len(all_values) / len(base_values):.1f}")

    '''

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
            # Collect experience data (AtariWrapper handles frame stacking automatically)
            states, actions, next_states, rewards = collect_experience(
                env,  # Use wrapped environment
                num_episodes=1,
                max_steps_per_episode=10000,
                num_envs=256 # usually 512, but for debugging we use 1
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

        # Train world model
        dynamics_params, training_info = train_world_model(
            states,
            actions,
            next_states,
            rewards,
            learning_rate=3e-4,
            batch_size=4096, #usually 4096
            num_epochs=10000,
        )

        # Save the model and scaling factor
        with open(save_path, 'wb') as f:
            pickle.dump({
                'dynamics_params': dynamics_params,
                'scale_factor': training_info['scale_factor']
            }, f)
        print(f"Model saved to {save_path}")

    # # Compare real vs model
    compare_real_vs_model(num_steps=5000, render_scale=6)