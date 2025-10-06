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


def lambda_return_dreamerv2(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    discounts: jnp.ndarray,
    bootstrap: jnp.ndarray,
    lambda_: float = 0.95,
    axis: int = 0,
):
    """Your existing implementation"""
    next_values = jnp.concatenate([values[1:], jnp.array([bootstrap])], axis=axis)

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


def compute_trajectory_targets(traj_rewards, traj_values, traj_discounts, lambda_=0.95):
    """Your trajectory target computation"""
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


def manual_lambda_returns_reference(
    rewards, values, discounts, bootstrap, lambda_=0.95
):
    """
    Reference implementation following DreamerV2 paper equation (4):
    V^λ_t = r_t + γ_t * [(1-λ) * v(s_{t+1}) + λ * V^λ_{t+1}]

    Computed backwards from the end.
    """
    T = len(rewards)
    lambda_returns = jnp.zeros(T)

    next_lambda_return = bootstrap

    for t in reversed(range(T)):

        if t == T - 1:
            next_value = bootstrap
        else:
            next_value = values[t + 1]

        lambda_return = rewards[t] + discounts[t] * (
            (1 - lambda_) * next_value + lambda_ * next_lambda_return
        )

        lambda_returns = lambda_returns.at[t].set(lambda_return)
        next_lambda_return = lambda_return

    return lambda_returns


def test_simple_case():
    """Test with a very simple 3-step trajectory"""
    print("=== TEST 1: Simple 3-step trajectory ===")

    rewards = jnp.array([1.0, 0.5, 0.0])
    values = jnp.array([2.0, 1.5, 1.0])
    discounts = jnp.array([0.9, 0.9, 0.9])
    lambda_ = 0.8

    print(f"Input:")
    print(f"  Rewards: {rewards}")
    print(f"  Values: {values}")
    print(f"  Discounts: {discounts}")
    print(f"  Lambda: {lambda_}")
    print(f"  Bootstrap (last value): {values[-1]}")

    your_targets = compute_trajectory_targets(rewards, values, discounts, lambda_)

    ref_targets = manual_lambda_returns_reference(
        rewards[:-1], values[:-1], discounts[:-1], values[-1], lambda_
    )

    print(f"\nResults:")
    print(f"  Your targets:      {your_targets}")
    print(f"  Reference targets: {ref_targets}")
    print(f"  Max difference:    {jnp.abs(your_targets - ref_targets).max():.8f}")
    print(f"  Match? {jnp.allclose(your_targets, ref_targets, atol=1e-6)}")

    print(f"\n  Manual step-by-step calculation:")
    bootstrap = values[-1]

    v_lambda_1 = rewards[1] + discounts[1] * bootstrap
    print(f"    t=1: {rewards[1]} + {discounts[1]} * {bootstrap} = {v_lambda_1}")

    blended_next = (1 - lambda_) * values[1] + lambda_ * v_lambda_1
    v_lambda_0 = rewards[0] + discounts[0] * blended_next
    print(
        f"    t=0: {rewards[0]} + {discounts[0]} * ({1-lambda_} * {values[1]} + {lambda_} * {v_lambda_1})"
    )
    print(f"         = {rewards[0]} + {discounts[0]} * {blended_next} = {v_lambda_0}")

    manual_result = jnp.array([v_lambda_0, v_lambda_1])
    print(f"  Manual result:     {manual_result}")

    return jnp.allclose(your_targets, ref_targets, atol=1e-6)


def test_edge_cases():
    """Test edge cases"""
    print("\n=== TEST 2: Edge cases ===")

    print("All zeros:")
    rewards = jnp.zeros(5)
    values = jnp.zeros(5)
    discounts = jnp.ones(5) * 0.99

    your_result = compute_trajectory_targets(rewards, values, discounts)
    ref_result = manual_lambda_returns_reference(
        rewards[:-1], values[:-1], discounts[:-1], values[-1]
    )

    print(f"  Your result: {your_result}")
    print(f"  Reference:   {ref_result}")
    print(f"  Match? {jnp.allclose(your_result, ref_result, atol=1e-6)}")

    print("\nLambda=0 (1-step TD):")
    rewards = jnp.array([1.0, 2.0, 0.5])
    values = jnp.array([1.0, 1.5, 2.0])
    discounts = jnp.array([0.9, 0.9, 0.9])

    your_result = compute_trajectory_targets(rewards, values, discounts, lambda_=0.0)

    expected = rewards[:-1] + discounts[:-1] * values[1:]

    print(f"  Your result: {your_result}")
    print(f"  Expected:    {expected}")
    print(f"  Match? {jnp.allclose(your_result, expected, atol=1e-6)}")

    print("\nLambda=1 (Monte Carlo):")
    your_result = compute_trajectory_targets(rewards, values, discounts, lambda_=1.0)

    bootstrap = values[-1]

    mc_0 = rewards[0] + discounts[0] * (rewards[1] + discounts[1] * bootstrap)
    mc_1 = rewards[1] + discounts[1] * bootstrap
    expected = jnp.array([mc_0, mc_1])

    print(f"  Your result: {your_result}")
    print(f"  Expected:    {expected}")
    print(f"  Match? {jnp.allclose(your_result, expected, atol=1e-6)}")


def test_realistic_pong_data():
    """Test with realistic Pong-like data"""
    print("\n=== TEST 3: Realistic Pong data ===")

    T = 10
    rewards = jnp.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 0.8, 0.5, 0.2, 0.0])
    values = jnp.array([1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 2.0, 1.5, 1.0, 0.8])
    discounts = jnp.full(T, 0.95)

    print(f"Trajectory length: {T}")
    print(f"Rewards: {rewards}")
    print(f"Values: {values}")

    your_result = compute_trajectory_targets(rewards, values, discounts)
    ref_result = manual_lambda_returns_reference(
        rewards[:-1], values[:-1], discounts[:-1], values[-1]
    )

    print(f"\nTargets shape: {your_result.shape} (should be {T-1})")
    print(f"Your targets: {your_result}")
    print(f"Reference:    {ref_result}")
    print(f"Max diff: {jnp.abs(your_result - ref_result).max():.8f}")
    print(f"Match? {jnp.allclose(your_result, ref_result, atol=1e-6)}")

    print(f"\nSanity checks:")
    print(f"  All finite? {jnp.all(jnp.isfinite(your_result))}")
    print(f"  Target range: [{your_result.min():.3f}, {your_result.max():.3f}]")
    print(f"  Target mean: {your_result.mean():.3f}")


def run_all_tests():
    """Run all tests"""
    print("Testing compute_trajectory_targets function")
    print("=" * 60)

    test1_pass = test_simple_case()
    test_edge_cases()
    test_realistic_pong_data()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Simple case test: {'PASS' if test1_pass else 'FAIL'}")
    print("\nIf all tests pass, your compute_trajectory_targets function is correct.")
    print("If any test fails, there's a bug in the implementation.")


SEED = 42


def debug_lambda_returns_computation(rewards, values, discounts, lambda_=0.95):
    """
    Debug version of lambda returns with detailed logging
    """
    print("=== DEBUGGING LAMBDA RETURNS ===")

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

    print(f"\n=== SINGLE TRAJECTORY ANALYSIS ===")
    traj_idx = 0
    traj_rewards = rewards[:, traj_idx]
    traj_values = values[:, traj_idx]
    traj_discounts = discounts[:, traj_idx]

    print(f"Trajectory {traj_idx}:")
    print(f"  Rewards: {traj_rewards[:10]} ...")
    print(f"  Values: {traj_values[:10]} ...")
    print(f"  Discounts: {traj_discounts[:10]} ...")

    T = len(traj_rewards)
    bootstrap = traj_values[-1]

    lambda_returns = []

    next_lambda_return = bootstrap

    for t in reversed(range(T - 1)):

        reward_t = traj_rewards[t]
        discount_t = traj_discounts[t]
        value_next = traj_values[t + 1] if t + 1 < T - 1 else bootstrap

        lambda_return_t = reward_t + discount_t * (
            (1 - lambda_) * value_next + lambda_ * next_lambda_return
        )

        lambda_returns.append(lambda_return_t)
        next_lambda_return = lambda_return_t

        if t >= T - 5:
            print(
                f"  Step {t}: r={reward_t:.3f}, γ={discount_t:.3f}, "
                f"v_next={value_next:.3f}, λ_ret={lambda_return_t:.3f}"
            )

    lambda_returns = jnp.array(lambda_returns[::-1])

    print(f"\nLambda returns for trajectory {traj_idx}:")
    print(f"  Mean: {lambda_returns.mean():.4f}")
    print(f"  First 5: {lambda_returns[:5]}")
    print(f"  Last 5: {lambda_returns[-5:]}")

    return lambda_returns


def verify_lambda_return_implementation():
    """
    Verify the lambda return implementation against paper equation (4)
    """
    print("\n=== VERIFYING LAMBDA RETURN IMPLEMENTATION ===")

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

    V_lambda = jnp.zeros(T)
    V_lambda = V_lambda.at[-1].set(bootstrap)

    for t in reversed(range(T - 1)):
        next_value = values[t + 1] if t + 1 < T - 1 else bootstrap
        V_lambda = V_lambda.at[t].set(
            rewards[t]
            + discounts[t] * ((1 - lambda_) * next_value + lambda_ * V_lambda[t + 1])
        )

    print(f"Manual lambda returns: {V_lambda}")

    bootstrap_array = jnp.array([bootstrap])
    your_result = lambda_return_dreamerv2(
        rewards, values, discounts, bootstrap_array, lambda_=lambda_, axis=0
    )
    print(f"Your implementation: {your_result}")
    print(f"Difference: {jnp.abs(V_lambda - your_result).max():.6f}")


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
    class DreamerV2Actor(nn.Module):
        action_dim: int

        @nn.compact
        def __call__(self, x):
            hidden_size = 64
            
            # Add layer normalization for stability
            x = nn.LayerNorm()(x)
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            x = nn.elu(x)
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            x = nn.elu(x)
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            x = nn.elu(x)
            
            # Initialize with small values and neutral bias for better exploration
            logits = nn.Dense(
                self.action_dim, 
                kernel_init=orthogonal(0.01),  # Small initialization
                bias_init=constant(0.0)        # Neutral bias
            )(x)
            
            return distrax.Categorical(logits=logits)
    
    return DreamerV2Actor(action_dim=action_dim)


def create_dreamerv2_critic():
    """Create DreamerV2 Critic network with distributional output."""

    class DreamerV2Critic(nn.Module):

        @nn.compact
        def __call__(self, x):
            hidden_size = 64

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

            mean = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
            log_std = nn.Dense(
                1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(x)

            mean = jnp.squeeze(mean, axis=-1)
            log_std = jnp.squeeze(log_std, axis=-1)

            log_std = jnp.clip(log_std, -10, 2)

            return distrax.Normal(mean, jnp.exp(log_std))

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

            value_dist = critic_network.apply(critic_params, obs)
            value = value_dist.mean()

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

            reward = improved_pong_reward(next_obs, action, frame_stack_size=4)

            # No transformation needed - reward function already returns proper range
            print(f"Raw reward: {reward}")

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
        initial_value_dist = critic_network.apply(critic_params, cur_obs)
        initial_value = initial_value_dist.mean()
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

        pong_state = pong_flat_observation_to_state(
            cur_obs, unflattener, frame_stack_size=4
        )

        current_state = dummy_state.replace(env_state=pong_state)

        def rollout_step(carry, x):
            key, obs, state = carry

            key, action_key = jax.random.split(key)
            pi = actor_network.apply(actor_params, obs)
            action = pi.sample(seed=action_key)
            log_prob = pi.log_prob(action)

            value_dist = critic_network.apply(critic_params, obs)
            value = value_dist.mean()
            next_obs, next_state, reward, done, _ = env.step(state, action)
            next_obs, _ = flatten_obs(next_obs, single_state=True)

            next_obs = next_obs.astype(jnp.float32)

            # Use consistent reward function as imagined rollouts
            # This ensures real and imagined experiences have same reward structure
            reward = improved_pong_reward(next_obs, action, frame_stack_size=4)
            
            # Optional: Add scoring bonus (but keep positive-only structure)
            old_score_diff = obs[-5] - obs[-1]     # Previous player_score - opponent_score  
            new_score_diff = next_obs[-5] - next_obs[-1]  # Current player_score - opponent_score
            score_change = new_score_diff - old_score_diff
            
            # Add large bonus for scoring (positive only)
            scoring_bonus = jnp.where(
                score_change >= 1.0,  # Player scored
                2.0,  # Large positive bonus
                0.0   # No penalty for opponent scoring
            )
            
            reward = reward + scoring_bonus
            
            # Clip to reasonable positive range
            reward = jnp.clip(reward, 0.0, 2.1)  # Keep all rewards positive


            # jax.debug.print("obs[-5] : {} , obs[-1] : {}, next_obs[-5] : {}, next_obs[-1] : {}, reward : {}", obs[-5], obs[-1], next_obs[-5], next_obs[-1], reward)
            # jax.debug.print("obs[-1] : {}", obs[-1])
            # jax.debug.print("next_obs[-5] : {}", next_obs[-5])
            # jax.debug.print("next_obs[-1] : {}", next_obs[-1])
            # jax.debug.print("REWARD : {}", reward)

            discount_factor = jnp.array(discount)

            step_data = (next_obs, reward, discount_factor, action, value, log_prob, done)
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
            dones_seq,
        ) = trajectory_data

        action_counts = jnp.bincount(actions_seq, length=6)
        # print(
        #     f"Action 3 (LEFT): {action_counts[3]}, Action 4 (RIGHTFIRE): {action_counts[4]}"
        # )

        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([0.0]), rewards_seq])
        discounts = jnp.concatenate([jnp.array([discount]), discounts_seq])


        init_action = jnp.zeros_like(actions_seq[0])
        initial_value_dist = critic_network.apply(critic_params, cur_obs)
        initial_value = initial_value_dist.mean()
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([initial_value[None, ...], values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])

        return observations, rewards, discounts, actions, values, log_probs, dones_seq

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
    target_update_freq: int = 100,
    max_grad_norm: float = 100.0,
    target_kl=0.01,
    early_stopping_patience=5,
) -> Tuple[Any, Any, Any, Dict]:
    """Train DreamerV2 actor and critic networks following the paper exactly."""

    if key is None:
        key = jax.random.PRNGKey(42)

    actor_schedule = optax.exponential_decay(
        init_value=actor_lr,
        transition_steps=100,
        decay_rate=0.99,
        end_value=actor_lr * 0.1
    )
    
    critic_schedule = optax.exponential_decay(
        init_value=critic_lr,
        transition_steps=100,
        decay_rate=0.99,
        end_value=critic_lr * 0.1
    )
    
    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(actor_schedule, eps=1e-5),  # Use schedule instead of fixed lr
    )
    
    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(critic_schedule, eps=1e-5),  # Use schedule instead of fixed lr
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

    best_loss = float("inf")
    patience_counter = 0

    update_counter = 0

    batch_size = observations.shape[0] * observations.shape[1]
    T, B = rewards.shape[:2]

    def compute_trajectory_targets(traj_rewards, traj_values, traj_discounts):
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

    # In your training loop:
    targets = jax.vmap(compute_trajectory_targets, in_axes=(1, 1, 1), out_axes=1)(
        rewards, values, discounts
    )

    # DON'T normalize targets - this destroys the reward signal!
    # The critic needs to learn the actual value scale, not normalized targets
    targets_for_training = targets
    
    print(f"Raw targets - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")
    print(f"Targets range: [{targets.min():.4f}, {targets.max():.4f}]")

    print(
        f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    )
    # print(
    #     f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    # )
    # print(
    #     f"Lambda returns stats - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    # )
    # print(
    #     f"Lambda returns stats - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    # )

    observations_flat = observations[:-1].reshape((T - 1) * B, -1)
    actions_flat = actions[:-1].reshape((T - 1) * B)
    targets_flat = targets_for_training.reshape((T - 1) * B)
    values_flat = values[:-1].reshape((T - 1) * B)
    old_log_probs_flat = log_probs[:-1].reshape((T - 1) * B)

    """"
    Dreamerv2 code
     def critic_loss(self, seq, target):
        
        

        
        dist = self.critic(seq['feat'][:-1])
        target = tf.stop_gradient(target)
        weight = tf.stop_gradient(seq['weight'])
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {'critic': dist.mode().mean()}
        return critic_loss, metrics
    
    """
    def critic_loss_fn(critic_params, obs, targets):
        dist = critic_network.apply(critic_params, obs)
        
        # Use the proper DreamerV2 distributional loss
        log_prob = dist.log_prob(targets)
        critic_loss = -jnp.mean(log_prob)
        
        return critic_loss, {
            "critic_loss": critic_loss,
            "predictions_mean": jnp.mean(dist.mean()),
            "predictions_std": jnp.std(dist.mean()),
            "targets_mean": jnp.mean(targets),
            "targets_std": jnp.std(targets),
        }
    """ def actor_loss(self, seq, target):
    
    
    
    
    
    
    
    
    
    metrics = {}
    
    
    
    
    policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics"""


    def actor_loss_fn(actor_params, obs, actions, targets, values, old_log_probs):
        pi = actor_network.apply(actor_params, obs)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()
        
        # DreamerV2 uses targets directly as advantages (no baseline subtraction in some variants)
        # Since targets already include value bootstrapping via lambda returns
        advantages = jax.lax.stop_gradient(targets)
        
        # Less aggressive advantage normalization - only if std is reasonable
        adv_mean = jnp.mean(advantages)
        adv_std = jnp.std(advantages) + 1e-8
        
        # Only normalize if advantages have meaningful variation
        advantages = jnp.where(
            adv_std > 0.1,  # Only normalize if there's meaningful variation
            (advantages - adv_mean) / adv_std,
            advantages  # Keep raw advantages if they're too uniform
        )
        
        # REINFORCE-style policy gradient loss
        policy_loss = -jnp.mean(log_prob * advantages)
        
        # Entropy regularization
        entropy_bonus = entropy_scale * jnp.mean(entropy)
        
        total_loss = policy_loss - entropy_bonus
        
        return total_loss, {
            "policy_loss": policy_loss,
            "entropy": jnp.mean(entropy),
            "advantages_mean": adv_mean,
            "advantages_std": adv_std,
            "targets_mean": jnp.mean(targets),
        }
    

    metrics_history = []

    for epoch in range(num_epochs):

        key, subkey = jax.random.split(key)

        # Proper data shuffling for better training
        perm = jax.random.permutation(subkey, (T - 1) * B)
        obs_shuffled = observations_flat[perm]
        actions_shuffled = actions_flat[perm]
        targets_shuffled = targets_flat[perm]
        values_shuffled = values_flat[perm]
        old_log_probs_shuffled = old_log_probs_flat[perm]

        # if epoch == 0:
        #     current_preds_dist = critic_network.apply(
        #         critic_state.params, obs_shuffled[:100]
        #     )
        #     current_preds = current_preds_dist.mean()
        #     sample_targets = targets_shuffled[:100]

        #     print(f"\nCritic debugging (epoch {epoch}):")
        #     print(
        #         f"  Target stats: mean={sample_targets.mean():.3f}, std={sample_targets.std():.3f}"
        #     )
        #     print(
        #         f"  Prediction stats: mean={current_preds.mean():.3f}, std={current_preds.std():.3f}"
        #     )
        #     print(
        #         f"  Scale difference: {abs(sample_targets.mean() - current_preds.mean()):.3f}"
        #     )

        old_pi = actor_network.apply(actor_params, obs_shuffled)
        old_log_probs_new = old_pi.log_prob(actions_shuffled)

        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_state.params, obs_shuffled, targets_shuffled)

        # critic_grad_norm = jnp.sqrt(
        #     sum(jnp.sum(g**2) for g in jax.tree.leaves(critic_grads))
        # )
        # if critic_grad_norm > 10.0:
        #     print(
        #         f"Warning: Large critic gradients ({critic_grad_norm:.2f}) at epoch {epoch}"
        #     )

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

        new_pi = actor_network.apply(actor_state.params, obs_shuffled)
        new_log_probs = new_pi.log_prob(actions_shuffled)
        kl_div = jnp.mean(old_log_probs_new - new_log_probs)

        # if kl_div > target_kl:
        #     print(
        #         f"Early stopping at epoch {epoch}: KL divergence {kl_div:.6f} > {target_kl}"
        #     )
        #     break

        update_counter += 1

        epoch_metrics = {**critic_metrics, **actor_metrics}
        metrics_history.append(epoch_metrics)

        print(
            f"Epoch {epoch}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
            f"Entropy: {actor_metrics['entropy']:.4f}, Adv Mean: {actor_metrics['advantages_mean']:.4f}"
        )

        # More detailed training diagnostics
        if epoch % 1 == 0:  # Print every epoch for debugging
            print(f"  Targets - Mean: {actor_metrics['targets_mean']:.4f}, Std: {actor_metrics['advantages_std']:.4f}")
            print(f"  Values - Mean: {jnp.mean(values):.4f}, Std: {jnp.std(values):.4f}")
            print(f"  Rewards - Mean: {jnp.mean(rewards):.4f}, Non-zero: {jnp.sum(rewards != 0.0)}")

        total_loss = actor_loss + critic_loss
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: No improvement for {early_stopping_patience} epochs"
            )
            break

    return actor_state.params, critic_state.params


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
            temperature = 0.1
            scaled_logits = pi.logits / temperature
            scaled_pi = distrax.Categorical(logits=scaled_logits)
            action = scaled_pi.sample(seed=jax.random.PRNGKey(step_count))
            if step_count % 100 == 0:
                obs_flat, _ = flatten_obs(obs, single_state=True)
                training_reward = improved_pong_reward(
                    obs_flat, action, frame_stack_size=4
                )
                print(f"  Training reward would be: {training_reward:.3f}")

                obs_flat, _ = flatten_obs(obs, single_state=True)

                last_obs = obs_flat[(4 - 1) :: 4]

                player_y = last_obs[1]
                ball_y = last_obs[9]
                print(
                    f"  Player Y: {player_y:.2f}, Ball Y: {ball_y:.2f}, Distance: {abs(ball_y-player_y):.2f}"
                )

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


def analyze_policy_behavior(actor_network, actor_params, observations):
    """Analyze what the trained policy is doing"""

    print(observations.shape)

    sample_obs = observations.reshape(-1, observations.shape[-1])[:1000]
    print(sample_obs.shape)

    pi = actor_network.apply(actor_params, sample_obs)
    action_probs = jnp.mean(pi.probs, axis=0)

    print("\n=== POLICY ANALYSIS ===")
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    for i in range(len(action_names)):
        prob_val = float(action_probs[i])
        print(f"Action {i} ({action_names[i]}): {prob_val:.3f}")

    entropy_val = float(jnp.mean(pi.entropy()))
    movement_prob = float(action_probs[3] + action_probs[4])
    most_likely_idx = int(jnp.argmax(action_probs))

    print(f"Entropy: {entropy_val:.3f}")
    print(f"Favors movement: {movement_prob:.3f}")
    print(f"Most likely action: {most_likely_idx} ({action_names[most_likely_idx]})")

    return action_probs


def main():

    training_runs = 30

    training_params = {
        "action_dim": 6,
        "rollout_length": 60,      # Much shorter rollouts for stability
        "num_rollouts": 128,       # Smaller batch for stability  
        "policy_epochs": 20,        # Fewer epochs to prevent overfitting
        "actor_lr": 3e-5,          # Lower learning rate
        "critic_lr": 1e-4,         # Lower critic learning rate
        "lambda_": 0.95,           # Good value
        "entropy_scale": 3e-3,     # Higher entropy for exploration
        "discount": 0.99,          # Good for Pong
        "max_grad_norm": 10.0,     # Lower gradient clipping
        "target_kl": 0.01,         # Much more conservative
        "early_stopping_patience": 5,  # Shorter patience
    }

    for i in range(training_runs):

        parser = argparse.ArgumentParser(description="DreamerV2 Pong agent")

        actor_network = create_dreamerv2_actor(training_params["action_dim"])
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

        # Use different key for each training iteration
        iteration_key = jax.random.PRNGKey(SEED + i * 1000)
        shuffled_obs = jax.random.permutation(iteration_key, obs)

        print("Generating imagined rollouts...")
        (
            imagined_obs,
            imagined_rewards,
            imagined_discounts,
            imagined_actions,
            imagined_values,
            imagined_log_probs,
            dones_seq
        ) = generate_real_rollouts(
            dynamics_params=dynamics_params,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_network=actor_network,
            critic_network=critic_network,
            initial_observations=shuffled_obs[: training_params["num_rollouts"]],
            rollout_length=training_params["rollout_length"],
            normalization_stats=normalization_stats,
            discount=training_params["discount"],
            key=jax.random.split(iteration_key)[1],
        )

        # print(jnp.sum(dones_seq))
        # exit()

        print(imagined_rewards[:1000])

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
        actor_params, critic_params = train_dreamerv2_actor_critic(
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
            num_epochs=training_params["policy_epochs"],
            actor_lr=training_params["actor_lr"],
            critic_lr=training_params["critic_lr"],
            key=jax.random.PRNGKey(2000 + i * 100),
            lambda_=training_params["lambda_"],
            entropy_scale=training_params["entropy_scale"],
            target_kl=training_params["target_kl"],
            early_stopping_patience=training_params["early_stopping_patience"],
        )

        # Enhanced training monitoring
        mean_reward = jnp.mean(imagined_rewards)
        score_rewards = jnp.sum(jnp.abs(imagined_rewards) >= 0.9)  # Count actual scoring events
        non_zero_rewards = jnp.sum(imagined_rewards != 0.0)
        
        print(f"\n=== TRAINING RUN {i+1} SUMMARY ===")
        print(f"Mean reward: {mean_reward:.4f}")
        print(f"Scoring events: {score_rewards} / {imagined_rewards.size}")
        print(f"Non-zero rewards: {non_zero_rewards} / {imagined_rewards.size}")
        print(f"Reward range: [{jnp.min(imagined_rewards):.3f}, {jnp.max(imagined_rewards):.3f}]")

        def save_model_checkpoints(actor_params, critic_params):
            """Save parameters with consistent structure"""
            with open("actor_params.pkl", "wb") as f:
                pickle.dump({"params": actor_params}, f)
            with open("critic_params.pkl", "wb") as f:
                pickle.dump({"params": critic_params}, f)
            print("Saved actor, critic parameters")

        action_probs = analyze_policy_behavior(actor_network, actor_params, imagined_obs)
        
        # Check for balanced exploration
        min_action_prob = jnp.min(action_probs)
        max_action_prob = jnp.max(action_probs)
        exploration_balance = min_action_prob / max_action_prob
        print(f"Exploration balance: {exploration_balance:.3f} (higher is better)")

        save_model_checkpoints(actor_params, critic_params)
        
        # # Stop early if we're getting good scoring events and balanced exploration
        # if score_rewards > 0 and exploration_balance > 0.3:
        #     print("Good learning progress detected - early stopping criteria met!")
        #     print(f"Score events: {score_rewards}, Exploration balance: {exploration_balance:.3f}")
        #     break


if __name__ == "__main__":
    rtpt = RTPT(name_initials="FH", experiment_name="DreamerV2Agent", max_iterations=3)
    rtpt.start()
    main()
