import argparse
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Dict, Any
import numpy as np
import pickle
from jaxatari.games.jax_pong import JaxPong
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import distrax
from rtpt import RTPT
from jax import lax

from worldmodelPong import compare_real_vs_model, get_enhanced_reward
from model_architectures import *
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper

SEED = 42

# Let's add debugging to understand what's happening with your lambda returns

# Let's add debugging to understand what's happening with your lambda returns

# Let's add debugging to understand what's happening with your lambda returns

def debug_lambda_returns_computation(rewards, values, discounts, lambda_=0.95):
    """
    Debug version of lambda returns with detailed logging
    """
    print("=== DEBUGGING LAMBDA RETURNS ===")
    
    # Check input shapes and basic stats
    print(f"Rewards shape: {rewards.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Discounts shape: {discounts.shape}")
    
    print(f"\nRewards stats:")
    print(f"  Mean: {rewards.mean():.4f}")
    print(f"  Std: {rewards.std():.4f}")
    print(f"  Min: {rewards.min():.4f}")
    print(f"  Max: {rewards.max():.4f}")
    print(f"  Non-zero count: {jnp.sum(rewards != 0)}")
    
    print(f"\nValues stats:")
    print(f"  Mean: {values.mean():.4f}")
    print(f"  Std: {values.std():.4f}")
    print(f"  Min: {values.min():.4f}")
    print(f"  Max: {values.max():.4f}")
    
    print(f"\nDiscounts stats:")
    print(f"  Mean: {discounts.mean():.4f}")
    print(f"  Unique values: {jnp.unique(discounts)}")
    
    # Let's examine a single trajectory in detail
    print(f"\n=== SINGLE TRAJECTORY ANALYSIS ===")
    traj_idx = 0
    traj_rewards = rewards[:, traj_idx]  # Shape: (T,)
    traj_values = values[:, traj_idx]    # Shape: (T,)
    traj_discounts = discounts[:, traj_idx]  # Shape: (T,)
    
    print(f"Trajectory {traj_idx}:")
    print(f"  Rewards: {traj_rewards[:10]} ...")  # First 10 steps
    print(f"  Values: {traj_values[:10]} ...")
    print(f"  Discounts: {traj_discounts[:10]} ...")
    
    # Manually compute lambda returns for this trajectory
    T = len(traj_rewards)
    bootstrap = traj_values[-1]
    
    # Compute step by step
    lambda_returns = []
    
    # Start from the end and work backwards (this is how the paper does it)
    next_lambda_return = bootstrap
    
    for t in reversed(range(T-1)):  # T-2, T-3, ..., 0
        # V^λ_t = r_t + γ_t * [(1-λ) * v(s_{t+1}) + λ * V^λ_{t+1}]
        reward_t = traj_rewards[t]
        discount_t = traj_discounts[t]
        value_next = traj_values[t+1] if t+1 < T-1 else bootstrap
        
        lambda_return_t = reward_t + discount_t * (
            (1 - lambda_) * value_next + lambda_ * next_lambda_return
        )
        
        lambda_returns.append(lambda_return_t)
        next_lambda_return = lambda_return_t
        
        if t >= T-5:  # Show last few steps
            print(f"  Step {t}: r={reward_t:.3f}, γ={discount_t:.3f}, "
                  f"v_next={value_next:.3f}, λ_ret={lambda_return_t:.3f}")
    
    # Reverse to get chronological order
    lambda_returns = jnp.array(lambda_returns[::-1])
    
    print(f"\nLambda returns for trajectory {traj_idx}:")
    print(f"  Mean: {lambda_returns.mean():.4f}")
    print(f"  First 5: {lambda_returns[:5]}")
    print(f"  Last 5: {lambda_returns[-5:]}")
    
    return lambda_returns

# Also let's check if your lambda_return_dreamerv2 function matches the paper
def verify_lambda_return_implementation():
    """
    Verify the lambda return implementation against paper equation (4)
    """
    print("\n=== VERIFYING LAMBDA RETURN IMPLEMENTATION ===")
    
    # Create simple test case
    T = 5
    rewards = jnp.array([0.1, 0.2, -0.1, 0.3, 0.0])
    values = jnp.array([1.0, 1.1, 0.9, 1.2, 0.8])
    discounts = jnp.array([0.99, 0.99, 0.99, 0.99, 0.99])
    bootstrap = 0.5
    lambda_ = 0.95
    
    print(f"Test inputs:")
    print(f"  Rewards: {rewards}")
    print(f"  Values: {values}")
    print(f"  Discounts: {discounts}")
    print(f"  Bootstrap: {bootstrap}")
    
    # Manual computation following paper exactly
    # V^λ_t = r_t + γ_t * [(1-λ) * v(s_{t+1}) + λ * V^λ_{t+1}]
    
    # Start from end
    V_lambda = jnp.zeros(T)
    V_lambda = V_lambda.at[-1].set(bootstrap)  # Last value is bootstrap
    
    for t in reversed(range(T-1)):
        next_value = values[t+1] if t+1 < T-1 else bootstrap
        V_lambda = V_lambda.at[t].set(
            rewards[t] + discounts[t] * (
                (1 - lambda_) * next_value + lambda_ * V_lambda[t+1]
            )
        )
    
    print(f"Manual lambda returns: {V_lambda}")
    
    # Compare with your implementation  
    # Note: your function expects bootstrap as array, not scalar
    bootstrap_array = jnp.array([bootstrap])
    your_result = lambda_return_dreamerv2(
        rewards, values, discounts, bootstrap_array, lambda_=lambda_, axis=0
    )
    print(f"Your implementation: {your_result}")
    print(f"Difference: {jnp.abs(V_lambda - your_result).max():.6f}")

# Add this to your training function right after computing targets:
# debug_lambda_returns_computation(rewards, values, discounts)
# verify_lambda_return_implementation()




def flatten_obs(
    state, single_state: bool = False, is_list=False
) -> Tuple[jnp.ndarray, Any]:
    """
    Flatten the state PyTree into a single array.
    This is useful for debugging and visualization.
    """

    if type(state) == list:
        flat_states = []

        for s in state:
            flat_state, _ = jax.flatten_util.ravel_pytree(s)
            flat_states.append(flat_state)
        flat_states = jnp.stack(flat_states, axis=0)
        print(flat_states.shape)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener
    batch_shape = (
        state.player_x.shape[0]
        if hasattr(state, "player_x")
        else state.paddle_y.shape[0]
    )

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)
    return flat_state, unflattener


def create_dreamerv2_actor(action_dim: int):
    """Create DreamerV2 Actor network with ~1M parameters and ELU activations."""

    class DreamerV2Actor(nn.Module):
        action_dim: int

        @nn.compact
        def __call__(self, x):

            hidden_size = 512

            x = nn.Dense(
                hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = nn.elu(x)

            x = nn.Dense(
                hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = nn.elu(x)

            x = nn.Dense(
                hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = nn.elu(x)

            logits = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(x)

            return distrax.Categorical(logits=logits)

    return DreamerV2Actor(action_dim=action_dim)


def create_dreamerv2_critic():
    """Create DreamerV2 Critic network with ~1M parameters and ELU activations."""

    class DreamerV2Critic(nn.Module):

        @nn.compact
        def __call__(self, x):
            hidden_size = 512

            x = nn.Dense(
                hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = nn.elu(x)

            x = nn.Dense(
                hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = nn.elu(x)

            x = nn.Dense(
                hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = nn.elu(x)

            # value = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(x)
            value = nn.Dense(1, kernel_init=orthogonal(0.001), bias_init=constant(0.0))(x)

            return jnp.squeeze(value, axis=-1)

    return DreamerV2Critic()


def lambda_return_dreamerv2(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    discounts: jnp.ndarray,
    bootstrap: jnp.ndarray,
    lambda_: float = 0.95,
    axis: int = 0,
):

    next_values = jnp.concatenate([values[1:], jnp.array([bootstrap])], axis=axis)
    # next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=axis)

    def compute_target(carry, inputs):
        next_lambda_return = carry
        reward, discount, value, next_value = inputs

        target = reward + discount * (
            (1 - lambda_) * next_value + lambda_ * next_lambda_return
        )

        return target, target

    reversed_inputs = jax.tree.map(
        lambda x: jnp.flip(x, axis=axis), (rewards, discounts, values, next_values)
    )

    _, reversed_returns = lax.scan(compute_target, bootstrap, reversed_inputs)

    lambda_returns = jnp.flip(reversed_returns, axis=axis)

    return lambda_returns


def generate_imagined_rollouts(
    dynamics_params: Any,
    actor_params: Any,
    critic_params: Any,
    actor_network: nn.Module,
    critic_network: nn.Module,
    initial_observations: jnp.ndarray,
    rollout_length: int,
    normalization_stats: Dict,
    discount: float = 0.99,
    key: jax.random.PRNGKey = None,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Generate imagined rollouts following DreamerV2 approach."""

    if key is None:
        key = jax.random.PRNGKey(42)

    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]
    world_model = PongLSTM(10)

    def single_trajectory_rollout(cur_obs, subkey):
        """Generate a single trajectory starting from cur_obs."""

        def rollout_step(carry, x):
            key, obs, lstm_state = carry

            key, action_key = jax.random.split(key)
            pi = actor_network.apply(actor_params, obs)
            action = pi.sample(seed=action_key)
            log_prob = pi.log_prob(action)

            value = critic_network.apply(critic_params, obs)

            normalized_obs = (obs - state_mean) / state_std

            normalized_next_obs, new_lstm_state = world_model.apply(
                dynamics_params,
                None,
                normalized_obs,
                jnp.array([action]),
                lstm_state,
            )

            next_obs = normalized_next_obs * state_std + state_mean
            next_obs = next_obs.squeeze().astype(obs.dtype)

            reward = get_reward_from_ball_position(next_obs, frame_stack_size=4)

            print(f"Raw reward before tanh: {reward}")
            reward = jnp.tanh(reward * 0.1)
            print(f"Final reward after tanh: {reward}")

            discount_factor = jnp.array(discount)

            step_data = (next_obs, reward, discount_factor, action, value, log_prob)
            new_carry = (key, next_obs, new_lstm_state)

            return new_carry, step_data

        dummy_normalized_obs = (cur_obs - state_mean) / state_std
        dummy_action = jnp.zeros(1, dtype=jnp.int32)
        _, initial_lstm_state = world_model.apply(
            dynamics_params, None, dummy_normalized_obs, dummy_action, None
        )

        cur_obs = cur_obs.astype(jnp.float32)
        init_carry = (subkey, cur_obs, initial_lstm_state)

        _, trajectory_data = lax.scan(
            rollout_step, init_carry, None, length=rollout_length
        )

        (
            next_obs_seq,
            rewards_seq,
            discounts_seq,
            actions_seq,
            values_seq,
            log_probs_seq,
        ) = trajectory_data

        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([0.0]), rewards_seq])
        discounts = jnp.concatenate([jnp.array([discount]), discounts_seq])

        init_action = jnp.zeros_like(actions_seq[0])
        initial_value = critic_network.apply(critic_params, cur_obs)
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([initial_value[None, ...], values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])

        return observations, rewards, discounts, actions, values, log_probs

    num_trajectories = initial_observations.shape[0]
    keys = jax.random.split(key, num_trajectories)

    rollout_fn = jax.vmap(single_trajectory_rollout, in_axes=(0, 0))
    return rollout_fn(initial_observations, keys)


def generate_real_rollouts(
    dynamics_params: Any,
    actor_params: Any,
    critic_params: Any,
    actor_network: nn.Module,
    critic_network: nn.Module,
    rollout_length: int,
    initial_observations: jnp.ndarray,
    normalization_stats: Dict,
    discount: float = 0.99,
    key: jax.random.PRNGKey = None,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:

    from obs_state_converter import pong_flat_observation_to_state

    game = JaxPong()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=4,
    )
    dummy_obs, dummy_state = env.reset(jax.random.PRNGKey(0))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    if key is None:
        key = jax.random.PRNGKey(42)

    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]

    def single_trajectory_rollout(cur_obs, subkey):
        """Generate a single trajectory starting from cur_obs."""

        # Convert observation to pong state
        pong_state = pong_flat_observation_to_state(
            cur_obs, unflattener, frame_stack_size=4
        )
        
        # Create proper wrapper state structure
        # The AtariWrapper expects a state with env_state attribute
        current_state = dummy_state.replace(env_state=pong_state)

        def rollout_step(carry, x):
            key, obs, state = carry

            key, action_key = jax.random.split(key)
            pi = actor_network.apply(actor_params, obs)
            action = pi.sample(seed=action_key)
            log_prob = pi.log_prob(action)

            value = critic_network.apply(critic_params, obs)
            
            next_obs, next_state, reward, done, _ = env.step(state, action)
            next_obs, _ = flatten_obs(next_obs, single_state=True)
            
            # Ensure dtype consistency - convert to float32 to match input
            next_obs = next_obs.astype(jnp.float32)

            # reward = get_enhanced_reward(next_obs, action, frame_stack_size=4)
            reward = get_simple_dense_reward(next_obs, action, frame_stack_size=4)





            discount_factor = jnp.array(discount)

            step_data = (next_obs, reward, discount_factor, action, value, log_prob)
            new_carry = (key, next_obs, next_state)

            return new_carry, step_data

        cur_obs = cur_obs.astype(jnp.float32)
        init_carry = (subkey, cur_obs, current_state)

        _, trajectory_data = lax.scan(
            rollout_step, init_carry, None, length=rollout_length
        )

        (
            next_obs_seq,
            rewards_seq,
            discounts_seq,
            actions_seq,
            values_seq,
            log_probs_seq,
        ) = trajectory_data

        action_counts = jnp.bincount(actions_seq, length=6)
        print(f"Action 3 (LEFT): {action_counts[3]}, Action 4 (RIGHTFIRE): {action_counts[4]}")


        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([0.0]), rewards_seq])
        discounts = jnp.concatenate([jnp.array([discount]), discounts_seq])

        init_action = jnp.zeros_like(actions_seq[0])
        initial_value = critic_network.apply(critic_params, cur_obs)
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([initial_value[None, ...], values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])

        return observations, rewards, discounts, actions, values, log_probs

    num_trajectories = initial_observations.shape[0]
    keys = jax.random.split(key, num_trajectories)

    rollout_fn = jax.vmap(single_trajectory_rollout, in_axes=(0, 0))
    return rollout_fn(initial_observations, keys)


def train_dreamerv2_actor_critic(
    actor_params: Any,
    critic_params: Any,
    actor_network: nn.Module,
    critic_network: nn.Module,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    values: jnp.ndarray,
    log_probs: jnp.ndarray,
    num_epochs: int = 10,
    actor_lr: float = 8e-5,
    critic_lr: float = 8e-5,
    key: jax.random.PRNGKey = None,
    lambda_: float = 0.95,
    entropy_scale: float = 1e-3,
    use_reinforce: bool = True,
    target_update_freq: int = 100,
    max_grad_norm: float = 100.0,
) -> Tuple[Any, Any, Any, Dict]:
    """Train DreamerV2 actor and critic networks following the paper exactly."""

    if key is None:
        key = jax.random.PRNGKey(42)

    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )

    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    actor_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_params,
        tx=actor_tx,
    )

    critic_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_params,
        tx=critic_tx,
    )

    target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)
    update_counter = 0

    batch_size = observations.shape[0] * observations.shape[1]
    T, B = rewards.shape[:2]

    def compute_trajectory_targets(traj_rewards, traj_values, traj_discounts):
        """Compute λ-returns for a single trajectory."""

        bootstrap = traj_values[-1]

        targets = lambda_return_dreamerv2(
            traj_rewards[:-1],
            traj_values[:-1],
            traj_discounts[:-1],
            bootstrap,
            lambda_=lambda_,
            axis=0,
        )
        return targets

    targets = jax.vmap(compute_trajectory_targets, in_axes=(1, 1, 1), out_axes=1)(
        rewards, values, discounts
    )




    print(
        f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    )
    # With this:
    # debug_lambda_returns_computation(rewards, values, discounts)
    print(f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")
    # exit()

    observations_flat = observations[:-1].reshape((T - 1) * B, -1)
    actions_flat = actions[:-1].reshape((T - 1) * B)
    targets_flat = targets.reshape((T - 1) * B)
    values_flat = values[:-1].reshape((T - 1) * B)
    old_log_probs_flat = log_probs[:-1].reshape((T - 1) * B)

    # def critic_loss_fn(critic_params, obs, targets):
    #     """DreamerV2 critic loss with squared error."""
    #     predicted_values = critic_network.apply(critic_params, obs)

    #     loss = jnp.mean((predicted_values - targets) ** 2)
    #     return loss, {"critic_loss": loss, "critic_mean": jnp.mean(predicted_values)}
    def critic_loss_fn(critic_params, obs, targets):
        predicted_values = critic_network.apply(critic_params, obs)
        
        # Clip to reasonable range
        predicted_values = jnp.clip(predicted_values, -5.0, 5.0)
        
        loss = jnp.mean((predicted_values - targets) ** 2)
        return loss, {"critic_loss": loss, "critic_mean": jnp.mean(predicted_values)}

    def actor_loss_fn(actor_params, obs, actions, targets, values, old_log_probs):
        """DreamerV2 actor loss with inaction penalty."""
        pi = actor_network.apply(actor_params, obs)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()

        if use_reinforce:
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            reinforce_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(advantages))
            policy_loss = reinforce_loss
        else:
            policy_loss = -jnp.mean(targets)

        # Add inaction penalty
        # Penalize all actions except 3 (LEFT) and 4 (RIGHTFIRE)
        mask = jnp.array([1, 1, 1, 0, 0, 1], dtype=pi.probs.dtype)
        noop_penalty = 1 * jnp.sum(pi.probs * mask)
        
        entropy_loss = -entropy_scale * jnp.mean(entropy)
        
        total_loss = policy_loss + entropy_loss + noop_penalty

        return total_loss, {
            "actor_loss": total_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "noop_penalty": noop_penalty,
            "entropy": jnp.mean(entropy),
            "advantages_mean": jnp.mean(targets - values) if use_reinforce else 0.0,
        }
    
    metrics_history = []

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)

        perm = jax.random.permutation(subkey, (T - 1) * B)
        obs_shuffled = observations_flat[perm]
        actions_shuffled = actions_flat[perm]
        targets_shuffled = targets_flat[perm]
        values_shuffled = values_flat[perm]
        old_log_probs_shuffled = old_log_probs_flat[perm]

        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_state.params, obs_shuffled, targets_shuffled)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(
            actor_state.params,
            obs_shuffled,
            actions_shuffled,
            targets_shuffled,
            values_shuffled,
            old_log_probs_shuffled,
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        update_counter += 1
        if update_counter % target_update_freq == 0:
            target_critic_params = jax.tree.map(lambda x: x.copy(), critic_state.params)
            print(f"Updated target critic at step {update_counter}")

        epoch_metrics = {**critic_metrics, **actor_metrics}
        metrics_history.append(epoch_metrics)

        print(
            f"Epoch {epoch}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
            f"Entropy: {actor_metrics['entropy']:.4f}"
        )

    final_metrics = jax.tree.map(lambda *x: jnp.mean(jnp.array(x)), *metrics_history)

    return actor_state.params, critic_state.params, target_critic_params, final_metrics


def evaluate_real_performance(actor_network, actor_params, obs_shape, num_episodes=5):
    """Evaluate the trained policy in the real Pong environment."""
    from jaxatari.games.jax_pong import JaxPong

    env = JaxPong()
    env = AtariWrapper(
        env, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    total_rewards = []

    rng = jax.random.PRNGKey(0)

    for episode in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, state = env.reset(reset_key)
        episode_reward = 0
        done = False
        step_count = 0

        print(f"Episode {episode + 1}:")

        while not done and step_count < 1000:

            obs_tensor, _ = flatten_obs(obs, single_state=True)

            pi = actor_network.apply(actor_params, obs_tensor)
            action = pi.sample(seed=reset_key)
            print(action)

            obs, state, reward, done, _ = env.step(state, action)
            episode_reward += reward
            step_count += 1

            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward: {episode_reward:.3f}")

        total_rewards.append(episode_reward)
        print(f"  Final reward: {episode_reward:.3f}, Steps: {step_count}")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\nEvaluation Results:")
    print(f"Mean episode reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Episode rewards: {total_rewards}")

    return total_rewards


def main():

    training_runs = 1

    action_dim = 6
    rollout_length = 45 #only 45 when using real rollouts since they wait for a couple of frames to start
    num_rollouts = 1600
    policy_epochs = 20
    actor_lr = 1e-4
    critic_lr = 3e-4
    lambda_ = 0.95
    entropy_scale = 1e-1
    discount = 0.99

    for i in range(training_runs):

        parser = argparse.ArgumentParser(description="DreamerV2 Pong agent")

        actor_network = create_dreamerv2_actor(action_dim)
        critic_network = create_dreamerv2_critic()

        if os.path.exists("world_model_PongLSTM_pong.pkl"):
            with open("world_model_PongLSTM_pong.pkl", "rb") as f:
                saved_data = pickle.load(f)
                dynamics_params = saved_data["dynamics_params"]
                normalization_stats = saved_data.get("normalization_stats", None)

            print(f"Loading existing model from experience_data_LSTM_pong_0.pkl...")
            with open("experience_data_LSTM_pong_0.pkl", "rb") as f:
                saved_data = pickle.load(f)
                obs = saved_data["obs"]
        else:
            print("Train Model first")
            exit()

        obs_shape = obs.shape[1:]
        key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

        actor_params = None
        critic_params = None
        target_critic_params = None

        if os.path.exists("actor_params.pkl"):
            try:
                with open("actor_params.pkl", "rb") as f:
                    saved_data = pickle.load(f)
                    actor_params = saved_data.get("params", saved_data)
                    print("Loaded existing actor parameters")
            except Exception as e:
                print(f"Error loading actor params: {e}. Reinitializing...")
                key, subkey = jax.random.split(key)
                actor_params = actor_network.init(subkey, dummy_obs)
        else:
            key, subkey = jax.random.split(key)
            actor_params = actor_network.init(subkey, dummy_obs)
            print("Initialized new actor parameters")

        if os.path.exists("critic_params.pkl"):
            try:
                with open("critic_params.pkl", "rb") as f:
                    saved_data = pickle.load(f)
                    critic_params = saved_data.get("params", saved_data)
                    print("Loaded existing critic parameters")
            except Exception as e:
                print(f"Error loading critic params: {e}. Reinitializing...")
                key, subkey = jax.random.split(key)
                critic_params = critic_network.init(subkey, dummy_obs)
        else:
            key, subkey = jax.random.split(key)
            critic_params = critic_network.init(subkey, dummy_obs)
            print("Initialized new critic parameters")

        actor_param_count = sum(x.size for x in jax.tree.leaves(actor_params))
        critic_param_count = sum(x.size for x in jax.tree.leaves(critic_params))
        print(f"Actor parameters: {actor_param_count:,}")
        print(f"Critic parameters: {critic_param_count:,}")


        shuffled_obs = jax.random.permutation(jax.random.PRNGKey(SEED), obs)

        # evaluate_real_performance(
        #     actor_network, actor_params, obs_shape, num_episodes=5
        # )
        # exit()

        print("Generating imagined rollouts...")
        (
            imagined_obs,
            imagined_rewards,
            imagined_discounts,
            imagined_actions,
            imagined_values,
            imagined_log_probs,
        ) = generate_real_rollouts(
            dynamics_params=dynamics_params,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_network=actor_network,
            critic_network=critic_network,
            initial_observations=shuffled_obs[:num_rollouts],
            rollout_length=rollout_length,
            normalization_stats=normalization_stats,
            discount=discount,
            key=jax.random.PRNGKey(SEED),
        )

        parser.add_argument("--render", type=int, help="Number of rollouts to render")
        args = parser.parse_args()

        if args.render:
            visualization_offset = 50
            for i in range(int(args.render)):
                compare_real_vs_model(
                    steps_into_future=0,
                    obs=imagined_obs[i + visualization_offset],
                    actions=imagined_actions[i + visualization_offset],
                    frame_stack_size=4,
                )

        print(
            f"Reward stats: min={jnp.min(imagined_rewards):.4f}, max={jnp.max(imagined_rewards):.4f}"
        )
        print(
            f"Non-zero rewards: {jnp.sum(imagined_rewards != 0.0)} / {imagined_rewards.size}"
        )
        print(f"Imagined rollouts shape: {imagined_obs.shape}")

        print("Training DreamerV2 actor-critic...")
        actor_params, critic_params, target_critic_params, training_metrics = (
            train_dreamerv2_actor_critic(
                actor_params=actor_params,
                critic_params=critic_params,
                actor_network=actor_network,
                critic_network=critic_network,
                observations=imagined_obs,
                actions=imagined_actions,
                rewards=imagined_rewards,
                discounts=imagined_discounts,
                values=imagined_values,
                log_probs=imagined_log_probs,
                num_epochs=policy_epochs,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                key=jax.random.PRNGKey(2000),
                lambda_=lambda_,
                entropy_scale=entropy_scale,
                use_reinforce=True,
            )
        )

        if training_metrics:
            print(f"Final training metrics:")
            for key, value in training_metrics.items():
                print(f"  {key}: {value:.4f}")
            print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")

        def save_model_checkpoints(actor_params, critic_params, target_critic_params):
            """Save parameters with consistent structure"""
            with open("actor_params.pkl", "wb") as f:
                pickle.dump({"params": actor_params}, f)
            with open("critic_params.pkl", "wb") as f:
                pickle.dump({"params": critic_params}, f)
            with open("target_critic_params.pkl", "wb") as f:
                pickle.dump({"params": target_critic_params}, f)
            print("Saved actor, critic, and target critic parameters")

        save_model_checkpoints(actor_params, critic_params, target_critic_params)


if __name__ == "__main__":
    rtpt = RTPT(name_initials="FH", experiment_name="DreamerV2Agent", max_iterations=3)
    rtpt.start()
    main()
