import os
# Set XLA flags before importing JAX
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
# Make sure JAX_PLATFORM_NAME is not set to 'cpu'
if 'JAX_PLATFORM_NAME' in os.environ and os.environ['JAX_PLATFORM_NAME'] == 'cpu':
    del os.environ['JAX_PLATFORM_NAME']

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import pickle
import matplotlib.pyplot as plt
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Dict, List
from flax.training.train_state import TrainState
import distrax

# Check for GPU availability
devices = jax.devices()
if any(d.platform == 'gpu' for d in devices):
    print(f"GPU detected: {[d for d in devices if d.platform == 'gpu'][0]}")
    print(f"JAX is running on: {jax.devices()[0].platform.upper()}")
else:
    print("WARNING: No GPU detected! JAX will run on CPU which will be much slower.")
    print(f"Available devices: {devices}")

# Import your world model
from worldmodel import build_world_model, build_reward_model

# Import environment
import sys
from src.jaxatari.games.jax_seaquest import JaxSeaquest


class ActorCritic(nn.Module):
    """PPO actor-critic network with categorical action distribution."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray


def load_world_model(path):
    """Load a trained world model and reward model from file."""
    
    with open(path, 'rb') as f:
        saved_data = pickle.load(f)
    
    dynamics_model = build_world_model()
    reward_model = build_reward_model()
    
    # Check which format the saved data is in
    if 'dynamics_params' in saved_data and 'reward_params' in saved_data:
        # New format with both models
        return (dynamics_model, saved_data['dynamics_params']), (reward_model, saved_data['reward_params'])
    elif 'params' in saved_data:
        # Old format with just world model
        print("Found old model format with only dynamics model parameters.")
        print("Initializing reward model from scratch.")
        
        # Initialize reward model with random parameters
        dummy_obs, _ = JaxSeaquest().reset()
        dummy_flat_obs = JaxSeaquest().obs_to_flat_array(dummy_obs)
        dummy_action = jnp.array(0)
        
        reward_rng = jax.random.PRNGKey(42)
        reward_params = reward_model.init(reward_rng, dummy_flat_obs, dummy_action)
        
        return (dynamics_model, saved_data['params']), (reward_model, reward_params)
    else:
        raise KeyError("Saved model file has unexpected format")


def ppo_train(
    game,
    use_world_model=True,
    dynamics_model=None,
    dynamics_params=None,
    reward_model=None,
    reward_params=None,
    num_envs=1,
    num_steps=128,
    total_timesteps=500000,
    learning_rate=2.5e-4,
    anneal_lr=True,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    update_epochs=4,
    num_minibatches=4,
    activation="tanh",
):
    """
    Train a PPO agent using either the world model or the real environment.
    
    Args:
        game: The game environment
        use_world_model: Whether to use the world model or real environment
        dynamics_model: World model for state transitions
        dynamics_params: Parameters for dynamics model
        reward_model: Model for reward prediction
        reward_params: Parameters for reward model
        num_envs: Number of parallel environments
        num_steps: Number of steps to run in each environment per update
        total_timesteps: Total number of timesteps to train for
        learning_rate: Learning rate
        anneal_lr: Whether to anneal the learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_eps: PPO clip parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        update_epochs: Number of epochs to update policy per rollout
        num_minibatches: Number of minibatches per epoch
        activation: Activation function for the network
    """
    # Calculate derived parameters
    num_updates = total_timesteps // num_steps // num_envs
    minibatch_size = num_envs * num_steps // num_minibatches
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    # Set up learning rate schedule if needed
    def linear_schedule(count):
        frac = 1.0 - (count // (num_minibatches * update_epochs)) / num_updates
        return learning_rate * frac
    
    # Initialize network
    dummy_obs, _ = game.reset()
    dummy_flat_obs = game.obs_to_flat_array(dummy_obs)
    obs_size = dummy_flat_obs.shape[0]
    
    network = ActorCritic(action_dim=18, activation=activation)  # 18 actions in Seaquest
    
    rng, init_rng = jax.random.split(rng)
    network_params = network.init(init_rng, jnp.zeros((obs_size,)))
    
    # Create optimizer
    if anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5)
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5)
        )
    
    # Initialize training state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    # JIT-compile environment step
    jitted_step = jax.jit(game.step)
    
    # Initialize environment state and observations
    # For simplicity, we'll handle one environment at a time even though the PPO implementation
    # can support multiple parallel environments
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = game.reset()
    flat_obs = game.obs_to_flat_array(obs)
    
    # World model step function
    def world_model_step(obs, action):
        """Step using the world model."""
        next_obs = dynamics_model.apply(dynamics_params, None, obs, action)
        reward = reward_model.apply(reward_params, None, obs, action)
        # In the world model, we assume episodes don't terminate
        done = jnp.array(False)
        return next_obs, reward, done
    
    # Real environment step function
    def env_step(state, action):
        """Step using the real environment."""
        next_obs, next_state, reward, done, _ = jitted_step(state, action)
        flat_next_obs = game.obs_to_flat_array(next_obs)
        return next_state, flat_next_obs, reward, done
    
    # Storage for metrics
    episode_returns = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    entropy_values = []
    
    episode_return = 0
    episode_length = 0
    
    # Main training loop
    for update in range(num_updates):
        # Storage for rollout
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        rollout_log_probs = []
        rollout_next_obs = []
        
        # Collect rollout
        for step in range(num_steps):
            rollout_obs.append(flat_obs)
            
            # Get action from policy
            rng, action_rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, flat_obs)
            action = pi.sample(seed=action_rng)
            log_prob = pi.log_prob(action)
            
            # Execute action
            if use_world_model:
                next_obs, reward, done = world_model_step(flat_obs, action)
                env_state = None  # Not used with world model
            else:
                env_state, next_obs, reward, done = env_step(env_state, action)
            
            # Store transition
            rollout_actions.append(action)
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            rollout_values.append(value)
            rollout_log_probs.append(log_prob)
            rollout_next_obs.append(next_obs)
            
            # Update state
            flat_obs = next_obs
            
            # Track episode statistics
            episode_return += reward
            episode_length += 1
            
            if done:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                # Reset episode tracking
                episode_return = 0
                episode_length = 0
                
                # Reset environment if using real environment
                if not use_world_model:
                    rng, reset_rng = jax.random.split(rng)
                    obs, env_state = game.reset()
                    flat_obs = game.obs_to_flat_array(obs)
        
        # Convert lists to arrays
        obs_array = jnp.array(rollout_obs)
        action_array = jnp.array(rollout_actions)
        reward_array = jnp.array(rollout_rewards)
        done_array = jnp.array(rollout_dones)
        value_array = jnp.array(rollout_values)
        log_prob_array = jnp.array(rollout_log_probs)
        next_obs_array = jnp.array(rollout_next_obs)
        
        # Calculate returns and advantages
        # Get the value of the final state
        _, last_value = network.apply(train_state.params, flat_obs)
        
        # GAE calculation
        advantages = jnp.zeros(num_steps)
        returns = jnp.zeros(num_steps)
        
        # Manual implementation of GAE and returns calculation
        gae = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - done_array[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - done_array[t+1]
                next_value = value_array[t+1]
                
            delta = reward_array[t] + gamma * next_value * next_non_terminal - value_array[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            advantages = advantages.at[t].set(gae)
            returns = returns.at[t].set(gae + value_array[t])
            
        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # PPO update loop
        for epoch in range(update_epochs):
            # Shuffle the data
            rng, shuffle_rng = jax.random.split(rng)
            permutation = jax.random.permutation(shuffle_rng, num_steps)
            
            # Split into minibatches
            for start in range(0, num_steps, minibatch_size):
                end = start + minibatch_size
                batch_indices = permutation[start:end]
                
                # Get batch data
                batch_obs = obs_array[batch_indices]
                batch_actions = action_array[batch_indices]
                batch_log_probs = log_prob_array[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_values = value_array[batch_indices]
                
                # Compute PPO loss
                def loss_fn(params):
                    # Get new action probabilities and values
                    new_pi, new_values = jax.vmap(lambda o: network.apply(params, o))(batch_obs)
                    new_log_probs = new_pi.log_prob(batch_actions)
                    
                    # PPO clipped objective
                    ratio = jnp.exp(new_log_probs - batch_log_probs)
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
                    actor_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
                    
                    # Value function loss
                    value_pred_clipped = batch_values + jnp.clip(
                        new_values - batch_values, -clip_eps, clip_eps
                    )
                    value_loss1 = jnp.square(new_values - batch_returns)
                    value_loss2 = jnp.square(value_pred_clipped - batch_returns)
                    value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))
                    
                    # Entropy loss
                    entropy = jnp.mean(new_pi.entropy())
                    
                    # Total loss
                    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
                    
                    return total_loss, (actor_loss, value_loss, entropy)
                
                # Update model parameters
                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (_, (actor_loss, critic_loss, entropy)), grads = grad_fn(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                
                # Record metrics
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropy_values.append(entropy)
        
        # Log progress
        if (update % 10 == 0) or (update == num_updates - 1):
            mean_return = jnp.mean(jnp.array(episode_returns[-10:])) if episode_returns else 0
            print(f"Update: {update}/{num_updates}, Mean return: {mean_return:.2f}")
            print(f"Actor Loss: {jnp.mean(jnp.array(actor_losses[-10:])):.6f}, " +
                  f"Critic Loss: {jnp.mean(jnp.array(critic_losses[-10:])):.6f}, " +
                  f"Entropy: {jnp.mean(jnp.array(entropy_values[-10:])):.6f}")
    
    # Plot training metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_returns)
    plt.title('Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    
    plt.subplot(2, 2, 2)
    plt.plot(actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(critic_losses)
    plt.title('Critic Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 4)
    plt.plot(entropy_values)
    plt.title('Entropy')
    plt.xlabel('Update Step')
    plt.ylabel('Entropy')
    
    plt.tight_layout()
    model_type = "world_model" if use_world_model else "real_env"
    plt.savefig(f'ppo_training_metrics_{model_type}.png')
    plt.show()
    
    return train_state


def build_actor_critic(action_dim=18, activation="tanh"):
    """
    Build actor and critic networks for visualization/rendering.
    This function separates the actor and critic parts of the ActorCritic model.
    
    Args:
        action_dim: Number of possible actions
        activation: Activation function to use
    
    Returns:
        actor: Actor network
        critic: Critic network
    """
    class Actor(nn.Module):
        """Actor network that outputs action probabilities."""
        action_dim: int
        activation: str = "tanh"
        
        @nn.compact
        def __call__(self, x):
            if self.activation == "relu":
                activation_fn = nn.relu
            else:
                activation_fn = nn.tanh
                
            x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation_fn(x)
            x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation_fn(x)
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
            return logits
    
    class Critic(nn.Module):
        """Critic network that outputs state value estimates."""
        activation: str = "tanh"
        
        @nn.compact
        def __call__(self, x):
            if self.activation == "relu":
                activation_fn = nn.relu
            else:
                activation_fn = nn.tanh
                
            x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation_fn(x)
            x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation_fn(x)
            value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
            return jnp.squeeze(value, axis=-1)
    
    # Create actor and critic networks
    actor = Actor(action_dim=action_dim, activation=activation)
    critic = Critic(activation=activation)
    
    return actor, critic


if __name__ == "__main__":
    # Initialize game
    game = JaxSeaquest()
    
    # Load trained world model and reward model
    (dynamics_model, dynamics_params), (reward_model, reward_params) = load_world_model('world_model.pkl')
    
    # Choose whether to use world model or direct environment interaction
    use_world_model = False  # Set to False to use real environment
    
    # Train PPO agent
    train_state = ppo_train(
        game,
        use_world_model=use_world_model,
        dynamics_model=dynamics_model, 
        dynamics_params=dynamics_params,
        reward_model=reward_model, 
        reward_params=reward_params,
        num_steps=128,
        total_timesteps=50000,
        learning_rate=2.5e-4,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        update_epochs=4,
        num_minibatches=4
    )
    
    # Save trained policy
    model_type = "world_model" if use_world_model else "real_env"
    with open(f'ppo_agent_{model_type}.pkl', 'wb') as f:
        pickle.dump({
            'params': train_state.params,
        }, f)
    
    print(f"PPO training complete using {'world model' if use_world_model else 'real environment'}!")