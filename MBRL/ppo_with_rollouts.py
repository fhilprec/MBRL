import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import numpy as np
import distrax
from typing import Tuple, Any, Dict
from jax import random

from worldmodel import MODEL_ARCHITECTURE

def create_actor_critic_network(obs_shape: Tuple[int, ...], action_dim: int):
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
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
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
            critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
                critic
            )

            return pi, jnp.squeeze(critic, axis=-1)
    
    return ActorCritic(action_dim=action_dim)


def generate_imagined_rollouts(
    dynamics_params: Any,
    policy_params: Any, 
    network: nn.Module,
    initial_observations: jnp.ndarray,
    rollout_length: int,
    normalization_stats: Dict = None,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate imagined rollouts using the world model.
    
    Args:
        dynamics_params: Parameters of the trained world model
        policy_params: Parameters of the policy network
        network: The ActorCritic network
        initial_observations: Starting observations for rollouts [num_rollouts, obs_dim]
        rollout_length: Length of each rollout
        normalization_stats: Statistics for observation normalization
        key: Random key for sampling
        
    Returns:
        observations, actions, rewards, values, log_probs arrays
    """
    # Import world model functions (assuming they exist in worldmodel.py)
    from worldmodel import get_reward_from_observation
    
    if key is None:
        key = jax.random.PRNGKey(42)
    
    num_rollouts = initial_observations.shape[0]
    obs_dim = initial_observations.shape[1]
    
    # Initialize arrays to store rollout data
    observations = jnp.zeros((rollout_length, num_rollouts, obs_dim))
    actions = jnp.zeros((rollout_length, num_rollouts), dtype=jnp.int32)
    rewards = jnp.zeros((rollout_length, num_rollouts))
    values = jnp.zeros((rollout_length, num_rollouts))
    log_probs = jnp.zeros((rollout_length, num_rollouts))
    initial_lstm_state = None  # Initialize LSTM state if needed
    
    # Set initial observations
    current_obs = initial_observations
    observations = observations.at[0].set(current_obs)
    

    def rollout_step(carry, step_idx):
        current_obs, key, lstm_state = carry
        
        # Get action from policy
        key, subkey = jax.random.split(key)
        pi, value = network.apply(policy_params, current_obs)
        action = pi.sample(seed=subkey)
        log_prob = pi.log_prob(action)
        
        # Apply world model and get next LSTM state
        world_model = MODEL_ARCHITECTURE()
        next_obs, next_lstm_state = world_model.apply(
            dynamics_params,
            None,
            current_obs,
            jnp.array([action]),
            lstm_state,
        )

        reward = get_reward_from_observation(next_obs)

        # Store data
        step_data = {
            'obs': current_obs,
            'action': action,
            'reward': reward,
            'value': value,
            'log_prob': log_prob
        }
        
        return (next_obs, key, next_lstm_state), step_data

    # Run the rollout (assuming you have initial_lstm_state defined)
    key, subkey = jax.random.split(key)
    final_carry, rollout_data = jax.lax.scan(
        rollout_step, 
        (current_obs, subkey, initial_lstm_state), 
        jnp.arange(rollout_length)
    )
    
    # Extract arrays from rollout data
    observations = rollout_data['obs']
    actions = rollout_data['action'] 
    rewards = rollout_data['reward']
    values = rollout_data['value']
    log_probs = rollout_data['log_prob']
    
    return observations, actions, rewards, values, log_probs


def train_actor_critic(
    params: Any,
    network: nn.Module,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    log_probs: jnp.ndarray,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    key: jax.random.PRNGKey = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    num_minibatches: int = 4
) -> Tuple[Any, Dict]:
    """
    Train the actor-critic network on imagined rollouts.
    
    Args:
        params: Current network parameters
        network: The ActorCritic network
        observations: Rollout observations [rollout_length, num_rollouts, obs_dim]
        actions: Actions taken [rollout_length, num_rollouts]
        rewards: Rewards received [rollout_length, num_rollouts]
        values: Value estimates [rollout_length, num_rollouts]
        log_probs: Log probabilities of actions [rollout_length, num_rollouts]
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        key: Random key
        
    Returns:
        Updated parameters and training metrics
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )
    
    # Create train state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    
    # Calculate advantages using GAE
    def calculate_gae(rewards, values, gamma, gae_lambda):
        """Calculate Generalized Advantage Estimation."""
        rollout_length, num_rollouts = rewards.shape
        advantages = jnp.zeros_like(rewards)
        
        # Bootstrap value for the last step (assume 0 for terminal states)
        next_value = jnp.zeros(num_rollouts)
        gae = jnp.zeros(num_rollouts)
        
        def gae_step(carry, step_data):
            gae, next_value = carry
            reward, value, done = step_data
            
            # Assume done=False for simplicity in imagined rollouts
            done = jnp.zeros_like(reward)
            
            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * gae_lambda * (1 - done) * gae
            
            return (gae, value), gae
        
        # Reverse scan through the trajectory
        step_data = (rewards, values, jnp.zeros_like(rewards))
        _, advantages = jax.lax.scan(
            gae_step,
            (gae, next_value),
            step_data,
            reverse=True
        )
        
        return advantages
    
    advantages = calculate_gae(rewards, values, gamma, gae_lambda)
    targets = advantages + values
    
    # Flatten batch dimensions
    batch_size = observations.shape[0] * observations.shape[1]
    observations_flat = observations.reshape(batch_size, -1)
    actions_flat = actions.reshape(batch_size)
    advantages_flat = advantages.reshape(batch_size)
    targets_flat = targets.reshape(batch_size)
    old_log_probs_flat = log_probs.reshape(batch_size)
    old_values_flat = values.reshape(batch_size)
    
    # Normalize advantages
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
    
    def loss_fn(params, obs, actions, advantages, targets, old_log_probs, old_values):
        """Compute PPO loss."""
        # Forward pass
        pi, value = network.apply(params, obs)
        log_prob = pi.log_prob(actions)
        
        # Value loss
        value_pred_clipped = old_values + jnp.clip(
            value - old_values, -clip_eps, clip_eps
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        
        # Policy loss
        ratio = jnp.exp(log_prob - old_log_probs)
        loss_actor1 = ratio * advantages
        loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        
        # Entropy loss
        entropy = pi.entropy().mean()
        
        # Total loss
        total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy
        
        return total_loss, {
            'policy_loss': loss_actor,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss
        }
    
    # Training loop
    metrics_history = []
    
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # Shuffle data
        perm = jax.random.permutation(subkey, batch_size)
        obs_shuffled = observations_flat[perm]
        actions_shuffled = actions_flat[perm]
        advantages_shuffled = advantages_flat[perm] 
        targets_shuffled = targets_flat[perm]
        old_log_probs_shuffled = old_log_probs_flat[perm]
        old_values_shuffled = old_values_flat[perm]
        
        # Minibatch training
        minibatch_size = batch_size // num_minibatches
        epoch_metrics = []
        
        for i in range(num_minibatches):
            start_idx = i * minibatch_size
            end_idx = start_idx + minibatch_size
            
            batch_obs = obs_shuffled[start_idx:end_idx]
            batch_actions = actions_shuffled[start_idx:end_idx]
            batch_advantages = advantages_shuffled[start_idx:end_idx]
            batch_targets = targets_shuffled[start_idx:end_idx]
            batch_old_log_probs = old_log_probs_shuffled[start_idx:end_idx]
            batch_old_values = old_values_shuffled[start_idx:end_idx]
            
            # Compute gradients and update
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params,
                batch_obs,
                batch_actions, 
                batch_advantages,
                batch_targets,
                batch_old_log_probs,
                batch_old_values
            )
            
            train_state = train_state.apply_gradients(grads=grads)
            epoch_metrics.append(metrics)
        
        # Average metrics for this epoch
        epoch_avg_metrics = jax.tree_map(
            lambda *x: jnp.mean(jnp.array(x)),
            *epoch_metrics
        )
        metrics_history.append(epoch_avg_metrics)
    
    # Average final metrics
    final_metrics = jax.tree_map(
        lambda *x: jnp.mean(jnp.array(x)),
        *metrics_history
    )
    
    return train_state.params, final_metrics