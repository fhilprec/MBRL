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
from jaxatari.games.jax_seaquest import SeaquestRenderer



# Import the Seaquest environment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestObservation, SeaquestState
import os

def build_world_model():
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


def build_reward_model():
    """Build a reward prediction model."""
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
        
        # Simple MLP to predict reward
        x = hk.Linear(128)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        
        # Predict scalar reward
        reward = hk.Linear(1)(x)
        
        return reward.squeeze()
    
    return hk.transform(forward)


# Global variable to hold model for evaluate_model function
model = None

# Collect experience from environment
def collect_experience(game: JaxSeaquest, num_episodes: int = 1000, 
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
        
        for step in range(max_steps_per_episode):
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
    """Train the world model and reward model on collected experience."""
    global model  # Use global model for evaluate_model
    
    # Initialize models
    model = build_world_model()
    reward_model = build_reward_model()
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng_dynamics, init_rng_reward = jax.random.split(rng, 3)
    
    # Get a sample observation to initialize parameters
    dummy_obs, _ = game.reset()
    dummy_flat_obs = game.obs_to_flat_array(dummy_obs)
    dummy_action = jnp.array(0)
    
    # Initialize parameters
    dynamics_params = model.init(init_rng_dynamics, dummy_flat_obs, dummy_action)
    reward_params = reward_model.init(init_rng_reward, dummy_flat_obs, dummy_action)
    
    # Define loss functions
    def dynamics_loss_fn(params, rng, obs_batch, action_batch, next_obs_batch):
        pred_next_obs = model.apply(params, rng, obs_batch, action_batch)
        return jnp.mean(jnp.square(pred_next_obs - next_obs_batch))
    
    def reward_loss_fn(params, rng, obs_batch, action_batch, reward_batch):
        pred_rewards = reward_model.apply(params, rng, obs_batch, action_batch)
        return jnp.mean(jnp.square(pred_rewards - reward_batch))
    
    # Create optimizers
    dynamics_optimizer = optax.adam(learning_rate=1e-3)
    reward_optimizer = optax.adam(learning_rate=1e-3)
    
    dynamics_opt_state = dynamics_optimizer.init(dynamics_params)
    reward_opt_state = reward_optimizer.init(reward_params)
    
    # JIT-compile the training steps for efficiency
    @jax.jit
    def train_dynamics_step(params, opt_state, obs, actions, next_obs, rng):
        loss_val, grads = jax.value_and_grad(dynamics_loss_fn)(params, rng, obs, actions, next_obs)
        updates, new_opt_state = dynamics_optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
    
    @jax.jit
    def train_reward_step(params, opt_state, obs, actions, rewards, rng):
        loss_val, grads = jax.value_and_grad(reward_loss_fn)(params, rng, obs, actions, rewards)
        updates, new_opt_state = reward_optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
    
    # Collect experience data
    observations, actions, next_observations, rewards = collect_experience(
        game, num_episodes=num_episodes_collect
    )
    
    # Training loop
    dynamics_losses = []
    reward_losses = []
    
    print("Training world model and reward model...")
    for epoch in range(num_epochs):
        epoch_dynamics_losses = []
        epoch_reward_losses = []
        
        # Create batches for this epoch
        for batch_idx, (obs_batch, action_batch, next_obs_batch) in enumerate(create_batches(
            observations, actions, next_observations, batch_size
        )):
            # Extract matching rewards for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(rewards))
            reward_batch = jnp.array(rewards[start_idx:end_idx])
            
            # Train dynamics model
            rng, step_rng_dynamics = jax.random.split(rng)
            dynamics_params, dynamics_opt_state, dynamics_loss = train_dynamics_step(
                dynamics_params, dynamics_opt_state, obs_batch, action_batch, next_obs_batch, step_rng_dynamics
            )
            epoch_dynamics_losses.append(dynamics_loss)
            
            # Train reward model
            rng, step_rng_reward = jax.random.split(rng)
            reward_params, reward_opt_state, reward_loss = train_reward_step(
                reward_params, reward_opt_state, obs_batch, action_batch, reward_batch, step_rng_reward
            )
            epoch_reward_losses.append(reward_loss)
        
        # Compute average losses for this epoch
        avg_dynamics_loss = jnp.mean(jnp.array(epoch_dynamics_losses))
        avg_reward_loss = jnp.mean(jnp.array(epoch_reward_losses))
        
        dynamics_losses.append(avg_dynamics_loss)
        reward_losses.append(avg_reward_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            eval_mse = evaluate_model(dynamics_params, game, model)
            print(f"Epoch {epoch}/{num_epochs}, Dynamics Loss: {avg_dynamics_loss:.6f}, "
                  f"Reward Loss: {avg_reward_loss:.6f}, Eval MSE: {eval_mse:.6f}")
    
    # Save trained models if path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'dynamics_params': dynamics_params,
                'reward_params': reward_params
            }, f)
    
    # Plot training losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dynamics_losses)
    plt.title('Dynamics Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    
    plt.subplot(1, 2, 2)
    plt.plot(reward_losses)
    plt.title('Reward Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    
    plt.tight_layout()
    plt.savefig('model_training_losses.png')
    plt.show()
    
    return dynamics_params, reward_params



def visualize_world_model_predictions(game, params, num_steps=100, render_every=5, model=None):
    """
    Visualize world model predictions compared to actual game frames.
    
    Args:
        game: The JaxSeaquest game environment
        params: Trained world model parameters
        num_steps: Number of steps to simulate
        render_every: How often to render frames (every N steps)
        model: The world model to use for predictions
    """
    print("Visualizing world model predictions...")
    
    # Initialize environment
    obs, state = game.reset()

    # Create a renderer for visualization
    renderer = SeaquestRenderer()
    
    flat_obs = game.obs_to_flat_array(obs)
    
    # JIT-compiled step function
    jitted_step = jax.jit(game.step)
    
    for step in range(num_steps):
        # Choose a random action
        action = jax.random.randint(jax.random.PRNGKey(step), (), 0, 18)
        
        # Get actual next observation from environment
        next_obs, next_state, reward, done, _ = jitted_step(state, action)
        next_flat_obs = game.obs_to_flat_array(next_obs)
        
        # Get prediction from world model
        pred_next_flat_obs = model.apply(params, None, flat_obs, action)
        
        # Render every N steps
        if step % render_every == 0:
            mse = jnp.mean(jnp.square(pred_next_flat_obs - next_flat_obs))
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Render the actual game state using the renderer
            plt.subplot(1, 3, 1)
            game_frame = renderer.render(state)
            plt.imshow(game_frame, interpolation='nearest')
            plt.title(f"Actual Game Frame (Step {step})")
            
            # Render actual observation (flattened)
            plt.subplot(1, 3, 2)
            plt.imshow([next_flat_obs], aspect='auto', cmap='viridis')
            plt.title(f"Actual Observation\nStep {step}")
            plt.colorbar()
            
            # Render predicted observation
            plt.subplot(1, 3, 3)
            plt.imshow([pred_next_flat_obs], aspect='auto', cmap='viridis')
            plt.title(f"Predicted Observation\nMSE: {mse:.6f}")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f"world_model_viz_step_{step}.png")
            plt.show()
            print(f"Step {step}, MSE: {mse:.6f}")
        
        # Update for next step
        obs, state = next_obs, next_state
        flat_obs = next_flat_obs
        
        if done:
            print("Episode ended")
            obs, state = game.reset()
            flat_obs = game.obs_to_flat_array(obs)


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
    save_path = "world_model.pkl"

    # Set the global model variable
    model = build_world_model()

    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            
            dynamics_params = saved_data['dynamics_params']
            reward_params = saved_data['reward_params'] 
    else:
        print("No existing model found. Training a new model...")
        dynamics_params, reward_params = train_world_model(
            game, 
            num_epochs=50, 
            batch_size=64, 
            num_episodes_collect=100,
            save_path=save_path
        )

    # Pass the model parameter to the visualization function
    # visualize_world_model_predictions(game, dynamics_params, num_steps=50, render_every=10, model=model)