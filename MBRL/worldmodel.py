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



# Import the Seaquest environment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestObservation, SeaquestState

def build_world_model(game):
    def forward(flat_obs, action):
        # Check if we have a batch dimension
        batched = len(flat_obs.shape) > 1
        
        # Convert action to one-hot (handling both batched and single inputs)
        if batched:
            action_one_hot = jax.nn.one_hot(action, num_classes=18)  # Shape: (batch, 18)
            # Concatenate along the feature dimension (axis=1)
            inputs = jnp.concatenate([flat_obs, action_one_hot], axis=1)
        else:
            action_one_hot = jax.nn.one_hot(action, num_classes=18)  # Shape: (18,)
            # Concatenate along the feature dimension (axis=0)
            inputs = jnp.concatenate([flat_obs, action_one_hot], axis=0)
        
        # Simple MLP to predict next observation
        x = hk.Linear(256)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(256)(x)
        x = jax.nn.relu(x)
        
        # Predict next observation (output shape matches input observation shape)
        output_size = flat_obs.shape[-1]  # Get last dimension regardless of batching
        pred_next_obs = hk.Linear(output_size)(x)
        
        return pred_next_obs
    
    return hk.transform(forward)

# Global variable to hold model for evaluate_model function
model = None

# Collect experience from environment
def collect_experience(game: JaxSeaquest, num_episodes: int = 100, 
                       max_steps_per_episode: int = 1000) -> Tuple[List, List, List, List]:
    """Collect experience data by playing random actions in the environment."""
    print("Collecting experience data...")
    
    observations = []
    actions = []
    next_observations = []
    rewards = []
    
    # Get jitted step and reset functions for efficiency
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    
    for episode in tqdm(range(num_episodes)):
        obs, state = jitted_reset()
        
        for step in range(num_episodes):
            # Select random action
            action = jax.random.randint(
                jax.random.PRNGKey(episode * 10000 + step), 
                shape=(), 
                minval=0, 
                maxval=18
            )
            
            # Take a step in the environment
            next_obs, next_state, reward, done, _ = jitted_step(state, action)
            
            # Store experience
            observations.append(game.obs_to_flat_array(obs))
            actions.append(action)
            next_observations.append(game.obs_to_flat_array(next_obs))
            rewards.append(reward)
            
            if done:
                break
                
            # Update for next step
            obs, state = next_obs, next_state
    
    print(f"Collected {len(observations)} transitions from {num_episodes} episodes")
    return observations, actions, next_observations, rewards

# Create batches for training
def create_batches(observations, actions, next_observations, batch_size):
    """Create mini-batches from collected data."""
    num_samples = len(observations)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield (
            jnp.array([observations[i] for i in batch_indices]),
            jnp.array([actions[i] for i in batch_indices]),
            jnp.array([next_observations[i] for i in batch_indices])
        )

# Evaluate model predictions
def evaluate_model(params, game, model, num_steps=100):  # Pass model as parameter
    """Evaluate model prediction quality by comparing with actual environment steps."""
    obs, state = game.reset()
    jitted_step = jax.jit(game.step)
    
    mse_values = []
    
    for _ in range(num_steps):
        # Choose a random action
        action = jax.random.randint(jax.random.PRNGKey(42), (), 0, 18)
        
        # Get actual next observation from environment
        next_obs, next_state, _, _, _ = jitted_step(state, action)
        actual_next_obs = game.obs_to_flat_array(next_obs)
        
        # Get prediction from world model
        flat_obs = game.obs_to_flat_array(obs)
        pred_next_obs = model.apply(params, None, flat_obs, action)
        
        # Compute MSE between prediction and actual
        mse = jnp.mean(jnp.square(pred_next_obs - actual_next_obs))
        mse_values.append(mse)
        
        # Update for next step
        obs, state = next_obs, next_state
    
    return jnp.mean(jnp.array(mse_values))

# Main training function
def train_world_model(game, num_epochs=50, batch_size=64, 
                     num_episodes_collect=100, save_path=None):
    """Train the world model on collected experience."""
    global model  # Use global model for evaluate_model
    
    # Initialize model with game
    model = build_world_model(game)
    
    # Define loss function
    def loss_fn(params, rng, obs_batch, action_batch, next_obs_batch):
        pred_next_obs = model.apply(params, rng, obs_batch, action_batch)
        return jnp.mean(jnp.square(pred_next_obs - next_obs_batch))
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Get a sample observation to initialize parameters
    dummy_obs, _ = game.reset()
    dummy_flat_obs = game.obs_to_flat_array(dummy_obs)
    dummy_action = jnp.array(0)
    params = model.init(init_rng, dummy_flat_obs, dummy_action)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    # JIT-compile the training step for efficiency
    @jax.jit
    def train_step(params, opt_state, obs, actions, next_obs, rng):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, rng, obs, actions, next_obs)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
    
    # Collect experience data
    observations, actions, next_observations, _ = collect_experience(
        game, num_episodes=num_episodes_collect
    )
    
    # Training loop
    losses = []
    print("Training world model...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Create batches for this epoch
        for obs_batch, action_batch, next_obs_batch in create_batches(
            observations, actions, next_observations, batch_size
        ):
            # Training step
            rng, step_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(
                params, opt_state, obs_batch, action_batch, next_obs_batch, step_rng
            )
            epoch_losses.append(loss)
        
        # Compute average loss for this epoch
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(avg_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            eval_mse = evaluate_model(params, game, model)  # Pass model
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}, Eval MSE: {eval_mse:.6f}")
    
    # Save trained model if path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({'params': params}, f)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('World Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.savefig('world_model_training_loss.png')
    plt.show()
    
    return params

if __name__ == "__main__":
    # Check JAX configuration
    print("JAX version:", jax.__version__)
    print("Available devices:", jax.devices())
    print("Default backend:", jax.default_backend())
    
    # Check if GPU is available
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    if gpu_devices:
        print(f"GPU devices found: {gpu_devices}")
    else:
        print("No GPU devices found, falling back to CPU")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    # Initialize the game environment
    game = JaxSeaquest()
    
    # Train the world model
    params = train_world_model(game, 
                              num_epochs=50, 
                              batch_size=64, 
                              num_episodes_collect=10,
                              save_path="world_model.pkl")