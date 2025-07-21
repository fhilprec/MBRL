import os
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
import pygame
import time
import jax
import jax.numpy as jnp
from jax import lax, random
import optax
import haiku as hk
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle
from jaxatari.games.jax_seaquest import SeaquestRenderer, JaxSeaquest
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper
import gc
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import distrax




from worldmodel import collect_experience_sequential, train_world_model, get_reward_from_observation
from ppo_with_rollouts import train_actor_critic, create_actor_critic_network, generate_imagined_rollouts





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
    from worldmodel import predict_next_observation, get_reward_from_observation
    
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
    
    # Set initial observations
    current_obs = initial_observations
    observations = observations.at[0].set(current_obs)
    
    def rollout_step(carry, step_idx):
        current_obs, key = carry
        
        # Get action from policy
        key, subkey = jax.random.split(key)
        pi, value = network.apply(policy_params, current_obs)
        action = pi.sample(seed=subkey)
        log_prob = pi.log_prob(action)
        
        # Predict next observation using world model
        try:
            next_obs = predict_next_observation(
                dynamics_params, current_obs, action, normalization_stats
            )
        except:
            # Fallback: use a simple prediction or add noise to current obs
            key, subkey = jax.random.split(key)
            next_obs = current_obs + jax.random.normal(subkey, current_obs.shape) * 0.01
        

        reward = get_reward_from_observation(next_obs)
        
        # Store data
        step_data = {
            'obs': current_obs,
            'action': action,
            'reward': reward,
            'value': value,
            'log_prob': log_prob
        }
        
        return (next_obs, key), step_data
    
    # Run the rollout
    key, subkey = jax.random.split(key)
    final_carry, rollout_data = jax.lax.scan(
        rollout_step, 
        (current_obs, subkey), 
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



def main():

    iterations = 3
    frame_stack_size = 1
    
    # Initialize policy parameters and network
    policy_params = None
    network = None
    
    # Hyperparameters for policy training
    rollout_length = 50  # Length of imagined rollouts
    num_rollouts = 1000  # Number of rollouts per iteration
    policy_epochs = 20   # Number of policy training epochs
    learning_rate = 3e-4
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # Collect real experience
        game = JaxSeaquest()
        env = AtariWrapper(
            game, sticky_actions=False, episodic_life=False, frame_stack_size=frame_stack_size
        )
        env = FlattenObservationWrapper(env)

        obs, actions, rewards, _, boundaries = collect_experience_sequential(
            env, num_episodes=5, max_steps_per_episode=1e5, seed=i, policy_params=policy_params, network=network
        )

        next_obs = obs[1:] 
        obs = obs[:-1]  

        # Train world model
        print("Training world model...")
        dynamics_params, training_info = train_world_model(
            obs,
            actions,
            next_obs,
            rewards,
            episode_boundaries=boundaries,
            frame_stack_size=frame_stack_size,
            num_epochs=100,
        )
        normalization_stats = training_info.get("normalization_stats", None)

        # Train policy using imagined rollouts exclusively from the world model
        print("Training policy with imagined rollouts...")
        
        # Create or update actor-critic network
        if network is None:
            obs_shape = obs.shape[1:]  # Get observation shape
            action_dim = env.num_actions
            network = create_actor_critic_network(obs_shape, action_dim)
            
            # Initialize policy parameters if first iteration
            key = jax.random.PRNGKey(42)
            dummy_obs = jnp.zeros((1,) + obs_shape)
            policy_params = network.init(key, dummy_obs)
        
        # Generate imagined rollouts using the world model
        print("Generating imagined rollouts...")
        imagined_obs, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs = generate_imagined_rollouts(
            dynamics_params=dynamics_params,
            policy_params=policy_params,
            network=network,
            initial_observations=obs[:num_rollouts],  # Use real observations as starting points
            rollout_length=rollout_length,
            normalization_stats=normalization_stats,
            key=jax.random.PRNGKey(i * 1000)
        )
        
        # Train actor-critic on imagined rollouts
        print("Training actor-critic...")
        policy_params, training_metrics = train_actor_critic(
            params=policy_params,
            network=network,
            observations=imagined_obs,
            actions=imagined_actions,
            rewards=imagined_rewards,
            values=imagined_values,
            log_probs=imagined_log_probs,
            num_epochs=policy_epochs,
            learning_rate=learning_rate,
            key=jax.random.PRNGKey(i * 2000)
        )
        
        # Print training progress
        if training_metrics:
            print(f"Policy loss: {training_metrics.get('policy_loss', 'N/A'):.4f}")
            print(f"Value loss: {training_metrics.get('value_loss', 'N/A'):.4f}")
            print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")
        
        # Optional: Save checkpoints
        if i % 1 == 0:
            checkpoint = {
                'policy_params': policy_params,
                'dynamics_params': dynamics_params,
                'normalization_stats': normalization_stats,
                'iteration': i
            }
            with open(f'checkpoint_iter_{i}.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
        
        # Clean up memory
        gc.collect()
        
        print(f"Completed iteration {i+1}")
        print("-" * 50)


if __name__ == "__main__":    
    main()