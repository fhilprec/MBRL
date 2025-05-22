import os
# Add CUDA paths to environment
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
# Remove CPU restriction to allow GPU usage
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Comment this out

import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from jaxatari.games.jax_seaquest import SeaquestRenderer, JaxSeaquest
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper

VERBOSE = False



# Global variable to hold model for evaluate_model function
model = None

def build_world_model():
    def forward(state, action):
        # Flatten the state tree structure to a 1D vector
        flat_state = hk.Flatten()(jax.flatten_util.ravel_pytree(state)[0])
        
        # Convert action to one-hot
        action_one_hot = jax.nn.one_hot(action, num_classes=18)
        
        # Determine if we have a batch dimension
        is_batched = len(action_one_hot.shape) > 1
        
        # Concatenate along the appropriate axis
        if is_batched:
            inputs = jnp.concatenate([flat_state, action_one_hot], axis=1)
        else:
            inputs = jnp.concatenate([flat_state, action_one_hot], axis=0)
        
        # Feed through MLP
        x = hk.Linear(512)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        
        # Final output layer
        # For simplicity, predict the flattened state vector directly
        # The state size will be determined by the flattened input
        output_size = flat_state.shape[-1]
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

def create_batches(states, actions, next_states, rewards, batch_size):
    """Create mini-batches for training with flattened data."""
    # Determine total number of samples (should match first dimension of actions)
    num_samples = actions.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Extract batches from the flattened arrays
        # For states and next_states which are PyTrees, we need to select indices from each leaf
        states_batch = jax.tree.map(lambda x: x[batch_indices], states)
        actions_batch = actions[batch_indices]
        next_states_batch = jax.tree.map(lambda x: x[batch_indices], next_states)
        rewards_batch = rewards[batch_indices]
        
        yield states_batch, actions_batch, next_states_batch, rewards_batch


def train_world_model(game, num_epochs=50, batch_size=1024, 
                     num_episodes_collect=100, save_path=None):
    """Train the world model on flattened experience data."""
    global model
    
    # Initialize models
    model = build_world_model()
    
    # Initialize random keys
    rng = jax.random.PRNGKey(42)
    rng, init_rng_dynamics = jax.random.split(rng)
    
    # Initialize the environment to get a sample state
    _, state = game.reset(init_rng_dynamics)
    dummy_state = state.env_state.env_state  # Extract the inner state
    dummy_action = jnp.array(0)
    
    # Initialize model parameters
    dynamics_params = model.init(init_rng_dynamics, dummy_state, dummy_action)
    
    # Define dynamics loss function for batched, flattened data
    def dynamics_loss_fn(params, rng, states_batch, actions_batch, next_states_batch):
        # Convert PyTree states to flat arrays for all samples in batch at once
        states_flat = jax.vmap(lambda s: jax.flatten_util.ravel_pytree(s)[0])(states_batch)
        next_states_flat = jax.vmap(lambda s: jax.flatten_util.ravel_pytree(s)[0])(next_states_batch)
        
        # Apply model to all state-action pairs in batch at once
        pred_next_states_flat = jax.vmap(model.apply, in_axes=(None, None, 0, 0))(
            params, rng, states_batch, actions_batch
        )
        
        # Compute MSE loss across the batch
        mse = jnp.mean(jnp.square(pred_next_states_flat - next_states_flat))
        return mse
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(dynamics_params)
    
    # JIT-compile the training step
    @jax.jit
    def train_step(params, opt_state, states_batch, actions_batch, next_states_batch, rng):
        loss_val, grads = jax.value_and_grad(dynamics_loss_fn)(
            params, rng, states_batch, actions_batch, next_states_batch
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
    
    # Collect experience data
    print("Collecting experience data...")
    states, actions, next_states, rewards = collect_experience(
        game, num_episodes=num_episodes_collect
    )
    
    # No need to extract single environment data - we're using all data now
    print(f"Training on {actions.shape[0]} total transitions")
    
    # Training loop
    print("Training world model...")
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Create batches for this epoch
        batch_gen = create_batches(
            states, actions, next_states, rewards, batch_size
        )
        
        # Track progress with tqdm
        num_batches = actions.shape[0] // batch_size
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for states_batch, actions_batch, next_states_batch, _ in batch_gen:
                # Train dynamics model
                rng, step_rng = jax.random.split(rng)
                dynamics_params, opt_state, loss = train_step(
                    dynamics_params, opt_state, states_batch, actions_batch, next_states_batch, step_rng
                )
                epoch_losses.append(loss)
                pbar.update(1)
                pbar.set_postfix({"loss": float(loss)})
        
        # Compute average loss for this epoch
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(avg_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
    
    # Save trained model if path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'dynamics_params': dynamics_params,
            }, f)
        print(f"Model saved to {save_path}")
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('World Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('world_model_training_loss.png')
    plt.show()
    
    return dynamics_params

def evaluate_model(params, game, model, num_steps=100):
    """Evaluate model prediction quality on tree-structured states."""
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    
    _, state = game.reset(reset_rng)
    current_state = state.env_state.env_state
    
    mse_values = []
    
    for step in range(num_steps):
        # Choose a random action
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (), 0, 18)
        
        # Step environment to get actual next state
        rng, step_rng = jax.random.split(rng)
        _, next_state, _, _, _ = game.step(step_rng, state, action)
        actual_next_state = next_state.env_state.env_state
        
        # Get prediction from world model
        # Flatten states for comparison
        actual_next_state_flat = jax.flatten_util.ravel_pytree(actual_next_state)[0]
        pred_next_state_flat = model.apply(params, None, current_state, action)
        
        # Compute MSE between prediction and actual
        mse = jnp.mean(jnp.square(pred_next_state_flat - actual_next_state_flat))
        mse_values.append(mse)
        
        # Update for next step
        state = next_state
        current_state = actual_next_state
    
    return jnp.mean(jnp.array(mse_values))







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
        dynamics_params = train_world_model(
            env, 
            num_epochs=50, 
            batch_size=1024, 
            num_episodes_collect=5,
            save_path=save_path
        )

    # Evaluate the model
    eval_mse = evaluate_model(dynamics_params, game, model)
    print(f"Final evaluation MSE: {eval_mse:.6f}")