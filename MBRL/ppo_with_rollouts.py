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
from jax import lax

from worldmodel import MODEL_ARCHITECTURE, get_reward_from_observation

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



import jax
import jax.numpy as jnp
from jax import lax
from typing import Any, Dict, Tuple
import flax.linen as nn


def generate_imagined_rollouts(
    dynamics_params: Any,
    policy_params: Any, 
    network: nn.Module,
    initial_observations: jnp.ndarray,
    rollout_length: int,
    normalization_stats: Dict,
    key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Extract normalization stats
    state_mean = normalization_stats['mean']
    state_std = normalization_stats['std']
    
    world_model = MODEL_ARCHITECTURE()
    
    def single_rollout(carry, x):
        """Process a single initial observation through a complete rollout."""
        key, cur_obs = carry
        
        def rollout_step(step_carry, step_x):
            """Single step of the rollout."""
            key, obs, lstm_state = step_carry
            
            # Sample action from policy
            key, subkey = jax.random.split(key)
            pi, value = network.apply(policy_params, obs)
            action = pi.sample(seed=subkey)
            log_prob = pi.log_prob(action)
            
            # Normalize observation for world model
            normalized_obs = (obs - state_mean) / state_std
            
            # Predict next state
            normalized_next_obs, new_lstm_state = world_model.apply(
                dynamics_params,
                None,
                normalized_obs,
                jnp.array([action]),
                lstm_state,
            )
            
            # Denormalize
            next_obs = jnp.round(normalized_next_obs * state_std + state_mean)
            next_obs = next_obs.squeeze()
            
            # Get reward
            reward = get_reward_from_observation(next_obs)
            
            # Prepare outputs for this step
            step_outputs = {
                'obs': next_obs,
                'reward': reward,
                'action': action,
                'value': value,
                'log_prob': log_prob
            }
            
            new_carry = (key, next_obs, new_lstm_state)
            return new_carry, step_outputs
        
        # Initialize rollout
        initial_reward = get_reward_from_observation(cur_obs)
        lstm_state = None
        
        # Run rollout steps
        init_carry = (key, cur_obs, lstm_state)
        final_carry, step_outputs = lax.scan(
            rollout_step, 
            init_carry, 
            None, 
            length=rollout_length
        )
        
        # Construct complete sequences
        # Prepend initial observation and reward
        observations = jnp.concatenate([
            cur_obs[None, ...], 
            step_outputs['obs']
        ], axis=0)
        
        rewards = jnp.concatenate([
            jnp.array([initial_reward]),
            step_outputs['reward']
        ])
        
        # Actions start with None (represented as zeros) for initial state
        initial_action = jnp.zeros_like(step_outputs['action'][0])
        actions = jnp.concatenate([
            initial_action[None, ...],
            step_outputs['action']
        ])
        
        # Values and log_probs start with 0.0 for initial state
        values = jnp.concatenate([
            jnp.array([0.0]),
            step_outputs['value']
        ])
        
        log_probs = jnp.concatenate([
            jnp.array([0.0]),
            step_outputs['log_prob']
        ])
        
        rollout_outputs = {
            'observations': observations,
            'rewards': rewards,
            'actions': actions,
            'values': values,
            'log_probs': log_probs
        }
        
        return final_carry[0], rollout_outputs  # Return updated key
    
    # Process all initial observations
    init_carry = key
    final_key, all_outputs = lax.scan(
        single_rollout,
        init_carry,
        initial_observations
    )
    
    return (
        all_outputs['observations'],
        all_outputs['rewards'], 
        all_outputs['actions'],
        all_outputs['values'],
        all_outputs['log_probs']
    )


# Alternative version that handles None LSTM state explicitly
def generate_imagined_rollouts_handle_none(
    dynamics_params: Any,
    policy_params: Any, 
    network: nn.Module,
    initial_observations: jnp.ndarray,
    rollout_length: int,
    normalization_stats: Dict,
    key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Extract normalization stats
    state_mean = normalization_stats['mean']
    state_std = normalization_stats['std']
    world_model = MODEL_ARCHITECTURE()
    
    def single_trajectory_rollout(cur_obs, subkey):
        """Generate a single trajectory starting from cur_obs."""
        
        def rollout_step(carry, x):
            key, obs, lstm_state_valid, lstm_state = carry
            
            # Sample action from policy
            key, action_key = jax.random.split(key)
            pi, value = network.apply(policy_params, obs)
            action = pi.sample(seed=action_key)
            log_prob = pi.log_prob(action)
            
            # Normalize observation for world model
            normalized_obs = (obs - state_mean) / state_std
            
            # Use None for first step, then use the actual state
            actual_lstm_state = None if not lstm_state_valid else lstm_state
            
            # Predict next state
            normalized_next_obs, new_lstm_state = world_model.apply(
                dynamics_params,
                None,
                normalized_obs,
                jnp.array([action]),
                actual_lstm_state,
            )
            
            # Denormalize
            next_obs = jnp.round(normalized_next_obs * state_std + state_mean)
            next_obs = next_obs.squeeze()
            
            # Get reward
            reward = get_reward_from_observation(next_obs)
            
            step_data = (next_obs, reward, action, value, log_prob)
            # After first step, we have a valid LSTM state
            new_carry = (key, next_obs, True, new_lstm_state)
            
            return new_carry, step_data
        
        # Initialize with dummy LSTM state structure
        initial_reward = get_reward_from_observation(cur_obs)
        dummy_lstm_state = jnp.zeros((1, 64))  # Adjust based on your LSTM hidden size
        
        # Use flag to indicate if LSTM state is valid
        init_carry = (subkey, cur_obs, False, dummy_lstm_state)
        
        # Run rollout
        _, trajectory_data = lax.scan(rollout_step, init_carry, None, length=rollout_length)
        
        # Unpack trajectory data
        next_obs_seq, rewards_seq, actions_seq, values_seq, log_probs_seq = trajectory_data
        
        # Build complete sequences including initial state
        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([initial_reward]), rewards_seq])
        
        # Initial action/value/log_prob are placeholders
        init_action = jnp.zeros_like(actions_seq[0])
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([jnp.array([0.0]), values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])
        
        return observations, rewards, actions, values, log_probs
    
    # Split keys for each trajectory
    num_trajectories = initial_observations.shape[0]
    keys = jax.random.split(key, num_trajectories)
    
    # Vectorize over all initial observations
    rollout_fn = jax.vmap(single_trajectory_rollout, in_axes=(0, 0))
    return rollout_fn(initial_observations, keys)

# Alternative version with explicit vmap for clearer parallelization
def generate_imagined_rollouts(
    dynamics_params: Any,
    policy_params: Any, 
    network: nn.Module,
    initial_observations: jnp.ndarray,
    rollout_length: int,
    normalization_stats: Dict,
    key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Extract normalization stats
    state_mean = normalization_stats['mean']
    state_std = normalization_stats['std']
    world_model = MODEL_ARCHITECTURE()
    
    def single_trajectory_rollout(cur_obs, subkey):
        """Generate a single trajectory starting from cur_obs."""
        
        def rollout_step(carry, x):
            key, obs, lstm_state = carry
            
            # Sample action from policy
            key, action_key = jax.random.split(key)
            pi, value = network.apply(policy_params, obs)
            action = pi.sample(seed=action_key)
            log_prob = pi.log_prob(action)
            
            # Normalize observation for world model
            normalized_obs = (obs - state_mean) / state_std
            
            # Predict next state
            normalized_next_obs, new_lstm_state = world_model.apply(
                dynamics_params,
                None,
                normalized_obs,
                jnp.array([action]),
                lstm_state,
            )
            
            # Denormalize
            next_obs = jnp.round(normalized_next_obs * state_std + state_mean)
            next_obs = next_obs.squeeze()
            
            # Get reward
            reward = get_reward_from_observation(next_obs)
            
            step_data = (next_obs, reward, action, value, log_prob)
            new_carry = (key, next_obs, new_lstm_state)
            
            return new_carry, step_data
        
        # Initialize LSTM state properly to maintain consistent pytree structure
        initial_reward = get_reward_from_observation(cur_obs)
        
        # Get initial LSTM state structure by doing a dummy forward pass
        dummy_normalized_obs = (cur_obs - state_mean) / state_std
        dummy_action = jnp.zeros(1)  # Adjust size based on your action space
        _, initial_lstm_state = world_model.apply(
            dynamics_params,
            None,
            dummy_normalized_obs,
            dummy_action,
            None,
        )
        
        # Use the properly structured LSTM state
        init_carry = (subkey, cur_obs, initial_lstm_state)
        
        # Run rollout
        _, trajectory_data = lax.scan(rollout_step, init_carry, None, length=rollout_length)
        
        # Unpack trajectory data
        next_obs_seq, rewards_seq, actions_seq, values_seq, log_probs_seq = trajectory_data
        
        # Build complete sequences including initial state
        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([initial_reward]), rewards_seq])
        
        # Initial action/value/log_prob are placeholders
        init_action = jnp.zeros_like(actions_seq[0])
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([jnp.array([0.0]), values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])
        
        return observations, rewards, actions, values, log_probs
    
    # Split keys for each trajectory
    num_trajectories = initial_observations.shape[0]
    keys = jax.random.split(key, num_trajectories)
    
    # Vectorize over all initial observations
    rollout_fn = jax.vmap(single_trajectory_rollout, in_axes=(0, 0))
    return rollout_fn(initial_observations, keys)
      

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
        epoch_avg_metrics = jax.tree.map(
            lambda *x: jnp.mean(jnp.array(x)),
            *epoch_metrics
        )
        metrics_history.append(epoch_avg_metrics)
    
    # Average final metrics
    final_metrics = jax.tree.map(
        lambda *x: jnp.mean(jnp.array(x)),
        *metrics_history
    )
    
    return train_state.params, final_metrics