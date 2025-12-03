import argparse
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
#print current python user
import getpass
user = getpass.getuser()

if user == 'fhilprecht':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
import gc





from worldmodelPong import compare_real_vs_model, print_full_array, collect_experience_sequential, calculate_score_based_reward
from model_architectures import *
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper

import random

MODEL_SCALE_FACTOR = 5

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
    Flatten the state PyTree into a single array and remove score features.
    This is useful for debugging and visualization.

    NOTE: Removes last 8 features (score_player and score_enemy for 4 frames).
    Original: 56 features, Output: 48 features

    IMPORTANT: If input is already a flat array with 48 features (from world model),
    returns it unchanged.
    """

    # If already a flat array (from world model predictions)
    if isinstance(state, jnp.ndarray):
        if state.ndim == 1 and state.shape[0] == 48:
            # Already processed, return as-is
            return state, None
        elif state.ndim == 2 and state.shape[-1] == 48:
            # Batch of already-processed observations
            return state, None

    if type(state) == list:
        flat_states = []

        for s in state:
            # Check if already flat
            if isinstance(s, jnp.ndarray) and s.shape[0] == 48:
                flat_states.append(s)
            else:
                flat_state, _ = jax.flatten_util.ravel_pytree(s)
                # Remove last 8 features (scores)
                flat_states.append(flat_state[:-8])
        flat_states = jnp.stack(flat_states, axis=0)
        print(flat_states.shape)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        # Remove last 8 features (scores)
        return flat_state[:-8], unflattener

    batch_shape = (
        state.player_x.shape[0]
        if hasattr(state, "player_x")
        else state.paddle_y.shape[0]
    )

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)
    # Remove last 8 features (scores) from each batch element
    return flat_state[..., :-8], unflattener


def create_dreamerv2_actor(action_dim: int):
    """Create DreamerV2 Actor network with ~1M parameters and ELU activations."""

    class DreamerV2Actor(nn.Module):
        action_dim: int

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

            logits = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
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
                1, kernel_init=orthogonal(0.01), bias_init=constant(-1.0)  # Initialize to lower std
            )(x)

            mean = jnp.squeeze(mean, axis=-1)
            log_std = jnp.squeeze(log_std, axis=-1)

            # Tighter clipping for more stable learning: std in [0.05, 1.0]
            log_std = jnp.clip(log_std, -3.0, 0.0)

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
    reward_predictor_params: Any = None,
    model_scale_factor: int = MODEL_SCALE_FACTOR,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Generate imagined rollouts following DreamerV2 approach (OPTIMIZED & JIT-compiled)."""

    if key is None:
        key = jax.random.PRNGKey(42)

    # CRITICAL FIX: Use normalization stats from world model training!
    # The world model was trained on normalized data, so we must normalize during rollouts too
    if normalization_stats:
        state_mean = normalization_stats["mean"]
        state_std = normalization_stats["std"]
    else:
        state_mean = 0
        state_std = 1

    # print(state_mean)
    # print(state_std)

    # Use PongMLPDeep - retrained with use_deep=True
    world_model = PongMLPDeep(model_scale_factor)

    num_trajectories = initial_observations.shape[0]

    # DIAGNOSTIC: Check initial observations and normalization
    print("\n=== INITIAL OBSERVATION DIAGNOSTIC ===")
    print(f"Number of trajectories: {num_trajectories}")
    print(f"First initial obs (raw): player_y={initial_observations[0][1::4][-1]:.2f}, ball_x={initial_observations[0][8::4][-1]:.2f}, ball_y={initial_observations[0][9::4][-1]:.2f}")
    print(f"State mean sample: {state_mean if isinstance(state_mean, (int, float)) else state_mean[1::4][-1] if hasattr(state_mean, '__getitem__') else 'N/A'}")
    print(f"State std sample: {state_std if isinstance(state_std, (int, float)) else state_std[1::4][-1] if hasattr(state_std, '__getitem__') else 'N/A'}")
    normalized_test = (initial_observations[0] - state_mean) / state_std
    print(f"After normalization: player_y={normalized_test[1::4][-1]:.4f}, ball_x={normalized_test[8::4][-1]:.4f}, ball_y={normalized_test[9::4][-1]:.4f}")
    print("=" * 50)

    # Pre-compute initial LSTM states for all trajectories at once (OPTIMIZATION 1)
    print(f"Initializing LSTM states for {num_trajectories} trajectories...")
    dummy_action = jnp.zeros(1, dtype=jnp.int32)
    def init_lstm_state(obs):
        normalized_obs = (obs - state_mean) / state_std
        _, lstm_state = world_model.apply(
            dynamics_params, jax.random.PRNGKey(0), normalized_obs, dummy_action, None
        )
        return lstm_state

    initial_lstm_states = jax.vmap(init_lstm_state)(initial_observations)
    print("LSTM states initialized!")

    # JIT-compiled rollout function (OPTIMIZATION 2)
    @jax.jit
    def single_trajectory_rollout(cur_obs, subkey, initial_lstm_state):
        """Generate a single trajectory starting from cur_obs."""

        def rollout_step(carry, step_idx):
            key, obs, lstm_state = carry

            key, action_key = jax.random.split(key)
            pi = actor_network.apply(actor_params, obs)

            # # Add exploration noise to prevent policy collapse during imagination
            # key, noise_key = jax.random.split(key)
            # exploration_noise = jax.random.normal(noise_key, pi.logits.shape) * 0.5
            # pi_exploratory = distrax.Categorical(logits=pi.logits + exploration_noise)

            # action = pi_exploratory.sample(seed=action_key)
            action = pi.sample(seed=action_key)
            # action = jnp.asarray(4, dtype=jnp.int32) #overwrite for testing out
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
            next_obs = jnp.round(next_obs).squeeze().astype(obs.dtype)



            improved_reward = improved_pong_reward(next_obs, action, frame_stack_size=4)

            reward_predictor_reward = 0.0
            if reward_predictor_params is not None:
                print("REWARD PREDICTOR USED IN ROLLOUTS")
                reward_model = RewardPredictorMLPTransition(1)
                # RewardPredictorMLPPositionOnly expects (current_state, action, next_state)
                predicted_reward = reward_model.apply(
                    reward_predictor_params,
                    None,
                    obs[None, :],      # current state
                    jnp.array([action]),  # action (needs to be array)
                    next_obs[None, :]  # next state (IMAGINED - may have errors!)
                )
                # Clip and round to match real rollout behavior: {-1, 0, +1}
                predicted_reward_clipped = jnp.round(jnp.clip(jnp.squeeze(predicted_reward*(4/3)/2), -1.0, 1.0))

                # Apply confidence weighting: lower confidence for later steps
                reward_predictor_reward = predicted_reward_clipped # * confidence deactivate confidence for now

            # Combine rewards: hand-crafted (always) + predicted (confidence-weighted)
            # Hand-crafted reward provides stable gradient signal throughout
            # Predicted reward adds sparse score information when confident
            # reward = improved_reward
            reward = improved_reward

            # DEBUG: Print reward components for first trajectory, first few steps
            # def debug_print_rewards(traj_idx, step_idx, improved_rew, predictor_rew, total_rew):
            #     jax.debug.print(
            #         "Traj {traj}, Step {step}: improved={imp:.4f}, predictor={pred:.4f}, total={tot:.4f}",
            #         traj=traj_idx, step=step_idx, imp=improved_rew, pred=predictor_rew, tot=total_rew
            #     )

            # # Only print for trajectory 0, steps 0-5 to avoid spam
            # jax.lax.cond(
            #     step_idx < 6,
            #     lambda _: debug_print_rewards(0, step_idx, improved_reward, reward_predictor_reward, reward),
            #     lambda _: None,
            #     None
            # )

            discount_factor = jnp.array(discount)



            step_data = (next_obs, action, reward, discount_factor, value, log_prob)
            new_carry = (key, next_obs, new_lstm_state)

            return new_carry, step_data

        cur_obs = cur_obs.astype(jnp.float32)
        init_carry = (subkey, cur_obs, initial_lstm_state)

        # Pass step indices to rollout_step for confidence weighting
        step_indices = jnp.arange(rollout_length)
        _, trajectory_data = lax.scan(
            rollout_step, init_carry, step_indices
        )

        (
            next_obs_seq,
            actions_seq,
            rewards_seq,
            discounts_seq,
            values_seq,
            log_probs_seq,
        ) = trajectory_data

        # OPTIMIZATION 3: Avoid concatenations, build arrays directly in correct shape
        # Initial values
        initial_value_dist = critic_network.apply(critic_params, cur_obs)
        initial_value = initial_value_dist.mean()

        # Build full sequences directly (avoid concat overhead)
        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq], axis=0)
        actions = jnp.concatenate([jnp.zeros_like(actions_seq[0])[None, ...], actions_seq], axis=0)
        rewards = jnp.concatenate([jnp.array([0.0]), rewards_seq], axis=0)
        discounts = jnp.concatenate([jnp.array([discount]), discounts_seq], axis=0)
        values = jnp.concatenate([initial_value[None, ...], values_seq], axis=0)
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq], axis=0)

        total_steps = len(observations)

        return observations, actions, rewards, discounts, values, log_probs, total_steps

    keys = jax.random.split(key, num_trajectories)

    # OPTIMIZATION 4: Use vmap with pre-computed LSTM states
    rollout_fn = jax.vmap(single_trajectory_rollout, in_axes=(0, 0, 0))

    # IMPORTANT: Process in batches to avoid excessive JIT compilation time
    # First batch triggers compilation, subsequent batches reuse compiled code
    batch_size = 100  # Process 100 trajectories at a time
    num_batches = (num_trajectories + batch_size - 1) // batch_size

    print(f"Generating {num_trajectories} imagined trajectories in {num_batches} batches...")
    print("First batch will trigger JIT compilation (may take 30-60 seconds)...")

    all_observations = []
    all_actions = []
    all_rewards = []
    all_discounts = []
    all_values = []
    all_log_probs = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_trajectories)

        # Properly slice the LSTM state pytree
        batch_lstm_states = jax.tree_util.tree_map(
            lambda x: x[start_idx:end_idx],
            initial_lstm_states
        )

        batch_obs, batch_actions, batch_rewards, batch_discounts, batch_values, batch_log_probs, total_steps = rollout_fn(
            initial_observations[start_idx:end_idx],
            keys[start_idx:end_idx],
            batch_lstm_states
        )

        all_observations.append(batch_obs)
        all_actions.append(batch_actions)
        all_rewards.append(batch_rewards)
        all_discounts.append(batch_discounts)
        all_values.append(batch_values)
        all_log_probs.append(batch_log_probs)

        if batch_idx == 0:
            print(f"First batch complete! Remaining batches will be faster...")
        elif (batch_idx + 1) % 10 == 0:
            print(f"Completed {batch_idx + 1}/{num_batches} batches...")

    # Concatenate all batches
    observations = jnp.concatenate(all_observations, axis=0)
    actions = jnp.concatenate(all_actions, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)
    discounts = jnp.concatenate(all_discounts, axis=0)
    values = jnp.concatenate(all_values, axis=0)
    log_probs = jnp.concatenate(all_log_probs, axis=0)



    # OPTIMIZATION 5: Direct transpose to correct format (T, B, ...)
    return (
        jnp.transpose(observations[:, :-1], (1, 0, 2)),  # (B, T, F) -> (T, B, F)
        jnp.transpose(actions[:, 1:], (1, 0)),           # (B, T) -> (T, B)
        jnp.transpose(rewards[:, 1:], (1, 0)),           # (B, T) -> (T, B)
        jnp.transpose(discounts[:, 1:], (1, 0)),         # (B, T) -> (T, B)
        jnp.transpose(values, (1, 0)),                   # (B, T+1) -> (T+1, B)
        jnp.transpose(log_probs[:, 1:], (1, 0)),         # (B, T) -> (T, B)
        0
    )


def run_single_episode(episode_key, actor_params, actor_network, env, max_steps=10000, reward_predictor_params=None, model_scale_factor=MODEL_SCALE_FACTOR, use_score_reward=False, inference=False):
    """Run one complete episode using JAX scan with masking."""
    reset_key, step_key = jax.random.split(episode_key)
    obs, state = env.reset(reset_key)

    def step_fn(carry, _):
        rng, obs, state, done = carry

        # Skip if already done
        def continue_step(_):
            # Get action
            rng_new, action_key = jax.random.split(rng)
            flat_obs, _ = flatten_obs(obs, single_state=True)
            pi = actor_network.apply(actor_params, flat_obs)
            if not inference:
                action = pi.sample(seed=action_key)
            else:
                action = pi.mode()

            # Step environment
            next_obs, next_state, reward, next_done, _ = env.step(state, action)
            next_flat_obs = flatten_obs(next_obs, single_state=True)[0]

            # Compute score-based reward from state (not observations!)
            # Observations no longer contain scores (stripped to 48 features)
            # State is AtariState wrapper, actual game state is in env_state field
            old_player_score = state.env_state.player_score
            old_enemy_score = state.env_state.enemy_score
            new_player_score = next_state.env_state.player_score
            new_enemy_score = next_state.env_state.enemy_score

            score_reward = (new_player_score - old_player_score) - (new_enemy_score - old_enemy_score)
            # Clip to prevent weird edge cases
            score_reward = jnp.where(jnp.abs(score_reward) > 1, 0.0, score_reward)

            # Choose reward based on flag
            if use_score_reward:
                # For evaluation: use actual score difference from game state
                final_reward = jnp.array(score_reward, dtype=jnp.float32)
            else:
                # For training: use improved pong reward (hand-crafted from positions)
                improved_reward = improved_pong_reward(next_flat_obs, action, frame_stack_size=4)

                # HYBRID REWARD: For real rollouts, we can trust the predictor more since
                # next_state comes from the REAL environment (no compounding errors)
                # Still apply slight confidence weighting for consistency, but higher baseline
                reward_predictor_reward = jnp.array(0.0, dtype=jnp.float32)
                if reward_predictor_params is not None:
                    reward_model = RewardPredictorMLPTransition(1)
                    # RewardPredictorMLPPositionOnly expects (current_state, action, next_state)
                    rng_reward = jax.random.PRNGKey(0)
                    predicted_reward = reward_model.apply(
                        reward_predictor_params,
                        rng_reward,
                        flat_obs[None, :],       # current state
                        jnp.array([action]),     # action (needs to be array)
                        next_flat_obs[None, :]   # next state (REAL - no model errors!)
                    )
                    # Clip and round to match real rollout behavior: {-1, 0, +1}
                    predicted_reward_clipped = jnp.round(jnp.clip(jnp.squeeze(predicted_reward*(4/3)/2), -1.0, 1.0))



                else:
                    reward_predictor_reward = jnp.array(0.0, dtype=jnp.float32)

                # Combine rewards: hand-crafted (always) + predicted (confidence-weighted)
                final_reward = jnp.array(improved_pong_reward(next_flat_obs, action, frame_stack_size=4), dtype=jnp.float32)

            # Store transition with valid mask (valid = not done BEFORE this step)
            transition = (flat_obs, state, action, final_reward, ~done)

            return (rng_new, next_flat_obs, next_state, next_done), transition
        
        def skip_step(_):
            # Return dummy data when done - ENSURE EXACT TYPE MATCHING
            flat_obs, _ = flatten_obs(obs, single_state=True)
            
            # Make sure dummy values match the exact types from continue_step
            dummy_action = jnp.array(0, dtype=jnp.int32)  # Match action type
            dummy_reward = jnp.array(0.0, dtype=jnp.float32)  # Match reward type  
            dummy_valid = jnp.array(False, dtype=jnp.bool_)  # Match valid mask type
            
            dummy_transition = (
                flat_obs,  # Same obs type
                state,     # Same state type
                dummy_action,  # int32
                dummy_reward,  # float32
                dummy_valid    # bool
            )
            return (rng, flat_obs, state, done), dummy_transition
        
        return jax.lax.cond(done, skip_step, continue_step, None)
    
    flattened_init_obs = flatten_obs(obs, single_state=True)[0]

    initial_carry = (step_key, flattened_init_obs, state, jnp.array(False))
    _, transitions = jax.lax.scan(step_fn, initial_carry, None, length=max_steps)
    
    observations, states, actions, rewards, valid_mask = transitions
    
    # Filter to only valid steps
    episode_length = jnp.sum(valid_mask)
    
    return observations, actions, rewards, valid_mask, states, episode_length


def generate_real_rollouts(
    dynamics_params: Any,
    actor_params: Any,
    critic_params: Any,
    actor_network: nn.Module,
    critic_network: nn.Module,
    rollout_length: int,
    normalization_stats: Dict,
    discount: float = 0.99,
    num_episodes: int = 30,
    key: jax.random.PRNGKey = None,
    initial_observations = None,
    num_rollouts: int = 3000,
    reward_predictor_params: Any = None,
    model_scale_factor: int = MODEL_SCALE_FACTOR,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Collect episodes with JAX vmap and reshape to (rollout_length, num_rollouts, features)."""
    
    # Create environment
    game = JaxPong()
    env = AtariWrapper(game, sticky_actions=False, episodic_life=False, frame_stack_size=4)
    env = FlattenObservationWrapper(env)
    
    # Generate episode keys
    episode_keys = jax.random.split(key, num_episodes)
    
    print(f"Collecting {num_episodes} episodes with vmap...")
    
    # Run episodes in parallel with vmap
    vmapped_episode_fn = jax.vmap(
        lambda k: run_single_episode(k, actor_params, actor_network, env, reward_predictor_params=reward_predictor_params, model_scale_factor=model_scale_factor),
        in_axes=0
    )

    # After getting the vmapped results
    observations, actions, rewards, valid_mask, states, episode_length = vmapped_episode_fn(episode_keys)

    # Process each episode separately to extract only valid steps
    all_valid_obs = []
    all_valid_actions = []
    all_valid_rewards = []
    all_valid_states = []

    for ep_idx in range(num_episodes):
        ep_length = int(episode_length[ep_idx])
        
        # Extract valid steps for this episode
        valid_obs = observations[ep_idx, :ep_length]
        valid_actions = actions[ep_idx, :ep_length]
        valid_rewards = rewards[ep_idx, :ep_length]
        
        all_valid_obs.append(valid_obs)
        all_valid_actions.append(valid_actions)
        all_valid_rewards.append(valid_rewards)
        
    # Concatenate all valid episodes
    all_obs = jnp.concatenate(all_valid_obs, axis=0)
    all_actions = jnp.concatenate(all_valid_actions, axis=0)
    all_rewards = jnp.concatenate(all_valid_rewards, axis=0)

    print(f"Total valid steps: {len(all_obs)}")

    # Now you can sample rollouts from the concatenated valid data
    total_steps = len(all_obs)
    num_valid_starts = total_steps - rollout_length
    if num_valid_starts < num_rollouts:
        num_rollouts = num_valid_starts
        print(f"Reduced num_rollouts to {num_rollouts} due to insufficient data")

    # Sample random rollout start positions
    rng = jax.random.PRNGKey(42)
    start_indices = jax.random.choice(rng, num_valid_starts, shape=(num_rollouts,), replace=False)

    # Create rollouts by slicing
    obs_rollouts = jnp.stack([all_obs[i:i+rollout_length+1] for i in start_indices])
    actions_rollouts = jnp.stack([all_actions[i:i+rollout_length] for i in start_indices])
    rewards_rollouts = jnp.stack([all_rewards[i:i+rollout_length] for i in start_indices])

    # print(f"Rollout shapes: obs={obs_rollouts.shape}, actions={actions_rollouts.shape}, rewards={rewards_rollouts.shape}")
    
    # Compute values and log_probs
    all_obs_for_critic = obs_rollouts.reshape(-1, obs_rollouts.shape[-1])
    value_dists = critic_network.apply(critic_params, all_obs_for_critic)
    values = value_dists.mean().reshape(num_rollouts, rollout_length + 1)
    
    all_actions_flat = actions_rollouts.reshape(-1)
    all_obs_flat = obs_rollouts[:, :-1].reshape(-1, obs_rollouts.shape[-1])
    pis = actor_network.apply(actor_params, all_obs_flat)
    log_probs = pis.log_prob(all_actions_flat).reshape(num_rollouts, rollout_length)
    
    discounts = jnp.full_like(rewards_rollouts, discount)

    # print(f"Rollouts shape: (, {num_rollouts}, {rollout_length+1}, {obs_rollouts.shape[-1]})")
    # print(f"Before transpose - rewards: {rewards_rollouts.shape}, values: {values.shape}")

    # Transpose to (T, B, ...) format as expected by training function
    # Current format: (B, T, ...) where B=num_rollouts, T=rollout_length
    # Need format: (T, B, ...) where T=rollout_length, B=num_rollouts
    # NOTE: values needs T+1 timesteps for bootstrapping, others need T timesteps

    total_valid_steps = int(total_steps)
    print(f"Total valid steps: {total_valid_steps}")

    return (
        jnp.transpose(obs_rollouts[:, :-1], (1, 0, 2)),  # (B, T, F) -> (T, B, F)
        jnp.transpose(actions_rollouts, (1, 0)),         # (B, T) -> (T, B)
        jnp.transpose(rewards_rollouts, (1, 0)),         # (B, T) -> (T, B)
        jnp.transpose(discounts, (1, 0)),                # (B, T) -> (T, B)
        jnp.transpose(values, (1, 0)),                   # (B, T+1) -> (T+1, B) - KEEP all values!
        jnp.transpose(log_probs, (1, 0)),                # (B, T) -> (T, B)
        total_valid_steps,
    )


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

    best_loss = float("inf")
    patience_counter = 0

    update_counter = 0

    batch_size = observations.shape[0] * observations.shape[1]
    T, B = rewards.shape[:2]

    def compute_trajectory_targets(traj_rewards, traj_values, traj_discounts):
        """Compute λ-returns for a single trajectory.

        Args:
            traj_rewards: (T,) rewards for timesteps 0 to T-1
            traj_values: (T+1,) values for timesteps 0 to T
            traj_discounts: (T,) discount factors

        Returns:
            targets: (T,) lambda returns for timesteps 0 to T-1
        """
        bootstrap = traj_values[-1]  # V(s_T) for bootstrapping

        targets = lambda_return_dreamerv2(
            traj_rewards,          # (T,) - all T rewards
            traj_values[:-1],      # (T,) - V(s_0) to V(s_{T-1})
            traj_discounts,        # (T,) - all T discounts
            bootstrap,             # V(s_T) - bootstrap value
            lambda_=lambda_,
            axis=0,
        )
        return targets


    # Vmap over batch dimension (axis 1) - compute targets for each trajectory separately
    # rewards shape: (T, B), vmap over B to get 3000 trajectories of length T
    targets = jax.vmap(compute_trajectory_targets, in_axes=(1, 1, 1), out_axes=1)(
        rewards, values, discounts
    )
    targets_mean = targets.mean()
    targets_std = targets.std()
    targets_normalized = (targets - targets_mean) / (targets_std + 1e-8)

    # Use all T timesteps for training
    observations_flat = observations.reshape(T * B, -1)
    actions_flat = actions.reshape(T * B)
    targets_flat = targets.reshape(T * B)
    values_flat = values[:-1].reshape(T * B)  # values has T+1 timesteps
    old_log_probs_flat = log_probs.reshape(T * B)

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
        loss = -jnp.mean(dist.log_prob(targets))
        return loss, {"critic_loss": loss, "critic_mean": dist.mean()}

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

    def actor_loss_fn(
        actor_params,
        obs,
        actions,
        targets,
        values,
        old_log_probs,
        actor_grad="both",
        mix_ratio=0.1,
        debug=False,
    ):
        pi = actor_network.apply(actor_params, obs)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()

        advantages = targets - values

        # if debug:
        #     jax.debug.print("Raw advantages: mean={mean}, std={std}, min={min}, max={max}",
        #                   mean=advantages.mean(), std=advantages.std(),
        #                   min=advantages.min(), max=advantages.max())
        #     jax.debug.print("Actions: shape={shape}, dtype={dtype}, min={min}, max={max}, first_10={first}",
        #                   shape=actions.shape, dtype=actions.dtype,
        #                   min=actions.min(), max=actions.max(),
        #                   first=actions[:10])
        #     jax.debug.print("Pi logits: shape={shape}, min={min}, max={max}, first_sample={first}",
        #                   shape=pi.logits.shape,
        #                   min=pi.logits.min(), max=pi.logits.max(),
        #                   first=pi.logits[0])
        #     jax.debug.print("Pi probs: first_sample={first}",
        #                   first=pi.probs[0])
        #     jax.debug.print("Log probs: mean={mean}, std={std}, min={min}, max={max}, first_10={first}",
        #                   mean=log_prob.mean(), std=log_prob.std(),
        #                   min=log_prob.min(), max=log_prob.max(),
        #                   first=log_prob[:10])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # REINFORCE objective: maximize log_prob * advantages
        # We want to MAXIMIZE this, so the loss should be NEGATIVE
        reinforce_obj = log_prob * jax.lax.stop_gradient(advantages)

        # if debug:
        #     jax.debug.print("Normalized advantages: mean={mean}, std={std}",
        #                   mean=advantages.mean(), std=advantages.std())
        #     jax.debug.print("Reinforce obj: mean={mean}, std={std}, sum={sum}",
        #                   mean=reinforce_obj.mean(), std=reinforce_obj.std(),
        #                   sum=reinforce_obj.sum())

        # Entropy bonus encourages exploration (we want to MAXIMIZE entropy)
        entropy_bonus = entropy_scale * entropy

        # Total objective to MAXIMIZE (higher is better)
        total_objective = reinforce_obj + entropy_bonus

        # Loss to MINIMIZE (negative of what we want to maximize)
        actor_loss = -jnp.mean(total_objective)

        return actor_loss, {
            "actor_loss": actor_loss,
            "objective": jnp.mean(reinforce_obj),
            "entropy": jnp.mean(entropy),
            "advantages_mean": jnp.mean(advantages),
            "advantages_std": jnp.std(advantages),
            "mix_ratio": mix_ratio,
        }

    metrics_history = []

    for epoch in range(num_epochs):

        key, subkey = jax.random.split(key)

        perm = jax.random.permutation(subkey, T * B)
        obs_shuffled = observations_flat[perm]
        actions_shuffled = actions_flat[perm]
        targets_shuffled = targets_flat[perm]
        values_shuffled = values_flat[perm]
        old_log_probs_shuffled = old_log_probs_flat[perm]

        if epoch == 0:
            print(f"Training data shapes: obs={obs_shuffled.shape}, targets={targets_shuffled.shape}, values={values_shuffled.shape}")

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
            debug=False,  # Disable debug now that issue is fixed
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        new_pi = actor_network.apply(actor_state.params, obs_shuffled)
        new_log_probs = new_pi.log_prob(actions_shuffled)
        kl_div = jnp.mean(old_log_probs_new - new_log_probs)

        if kl_div > target_kl:
            print(
                f"Early stopping at epoch {epoch}: KL divergence {kl_div:.6f} > {target_kl}"
            )
            break

        update_counter += 1

        epoch_metrics = {**critic_metrics, **actor_metrics}
        metrics_history.append(epoch_metrics)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
                f"Entropy: {actor_metrics['entropy']:.4f}, KL: {kl_div:.6f}, "
                f"Reinforce Obj: {actor_metrics['objective']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}"
            )

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


def evaluate_real_performance(actor_network, actor_params, num_episodes=10, render=False, reward_predictor_params=None, model_scale_factor=MODEL_SCALE_FACTOR):
    """Evaluate the trained policy in the real Pong environment using JAX scan."""
    from jaxatari.games.jax_pong import JaxPong

    env = JaxPong()
    env = AtariWrapper(
        env, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )

    # Use a random seed if not provided
    seed = int(__import__('time').time() * 1000) % (2**31)
    rng = jax.random.PRNGKey(seed)

    # Generate episode keys
    episode_keys = jax.random.split(rng, num_episodes)

    print(f"Evaluating {num_episodes} episodes with vmap...")

    # Run episodes in parallel with vmap (reusing the existing run_single_episode function)
    # Use use_score_reward=True for evaluation to get actual Pong score
    vmapped_episode_fn = jax.vmap(
        lambda k: run_single_episode(k, actor_params, actor_network, env, reward_predictor_params=reward_predictor_params, model_scale_factor=model_scale_factor, use_score_reward=True, inference=True),
        in_axes=0
    )

    observations, actions, rewards, valid_mask, states, episode_lengths = vmapped_episode_fn(episode_keys)

    # Extract episode rewards by summing valid rewards for each episode
    total_rewards = []
    all_obs = []
    all_actions = []

    for ep_idx in range(num_episodes):
        ep_length = int(episode_lengths[ep_idx])

        # Sum rewards for valid steps
        ep_reward = float(jnp.sum(rewards[ep_idx, :ep_length]))
        total_rewards.append(ep_reward)

        print(f"Episode {ep_idx + 1}: Final reward: {ep_reward:.3f}, Steps: {ep_length}")

        # Collect observations and actions for rendering if needed
        if render:
            valid_obs = observations[ep_idx, :ep_length]
            valid_actions = actions[ep_idx, :ep_length]
            all_obs.append(valid_obs)
            all_actions.append(valid_actions)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\nEvaluation Results:")
    print(f"Mean episode reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Episode rewards: {total_rewards}")

    # Render collected episodes if requested
    if render and len(all_obs) > 0:
        # Concatenate all episodes for rendering
        obs_array = jnp.concatenate(all_obs, axis=0)
        actions_array = jnp.concatenate(all_actions, axis=0)

        print(f"\nRendering {len(obs_array)} frames from {num_episodes} episodes...")

        compare_real_vs_model(
            steps_into_future=0,
            obs=obs_array,
            num_steps=obs_array.shape[0],
            actions=actions_array,
            frame_stack_size=4,
            clock_speed=50,
            model_scale_factor=model_scale_factor,
            reward_predictor_params=None,
        )

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

    training_runs = 100000
    model_scale_factor = MODEL_SCALE_FACTOR  # Same as in worldmodelPong.py

    training_params = {
        "action_dim": 6,
        "rollout_length": 7,  # Reduced from 6 to 4 - errors compound too fast by step 3
        "num_rollouts": 30000,
        "policy_epochs": 10,  # Max epochs, KL will stop earlier
        "actor_lr": 8e-5,  # Reduced significantly for smaller policy updates
        "critic_lr": 5e-4,  # Moderate critic learning rate
        "lambda_": 0.95,
        "entropy_scale": 0.01,  # Maintain exploration
        "discount": 0.95,
        "max_grad_norm": 0.5,  # Tight gradient clipping
        "target_kl": 0.15,  # Slightly relaxed to allow 2-3 epochs
        "early_stopping_patience": 100,
        "retrain_interval": 50,  # Retrain world model every 50 iterations
        "wm_sample_size": 500,  # Number of samples to collect for world model training
        "wm_train_epochs": 20,  # Increased from 10 - better world model accuracy
    }

    parser = argparse.ArgumentParser(description="DreamerV2 Pong agent")
    parser.add_argument("--eval", type=bool, help="Specifies whether to run evaluation", default=0)
    parser.add_argument("--render", type=int, help="Specifies whether to run rendering", default=0)
    parser.add_argument("--rollout_style", type=str, help="Specifies whether to use 'model' or 'real' rollouts", default="real")
    args = parser.parse_args()


    rollout_func = None
    prefix = ""
    if args.rollout_style not in ["model", "real"]:
        print("Invalid rollout_style argument. Use 'model' or 'real'.")
        exit()
    else:
        print(f"Using '{args.rollout_style}' rollouts for training.")
        if args.rollout_style == "model":
            rollout_func = generate_imagined_rollouts
            prefix = "imagined_"
        if args.rollout_style == "real":
            rollout_func = generate_real_rollouts
            prefix = "real_"

    model_exists = False

    # Load starting iteration counter from actor params if it exists
    start_iteration = 0
    if os.path.exists(f"{prefix}actor_params.pkl"):
        try:
            with open(f"{prefix}actor_params.pkl", "rb") as f:
                saved_data = pickle.load(f)
                start_iteration = saved_data.get("iteration", 0)
                print(f"Resuming from iteration {start_iteration}")
        except Exception as e:
            print(f"Could not load iteration counter: {e}. Starting from 0")
            start_iteration = 0

    for i in range(start_iteration, start_iteration + training_runs):
        print(f"This is the {i}th iteration training the actor-critic")

        actor_network = create_dreamerv2_actor(training_params["action_dim"])
        critic_network = create_dreamerv2_critic()

        # Try to load MLP world model first, fall back to LSTM model
        model_path = None
        loaded_model_scale_factor = MODEL_SCALE_FACTOR  # Default
        loaded_use_deep = True  # Default
        loaded_model_type = "PongMLPDeep"  # Default

        if os.path.exists("worldmodel_mlp.pkl"):
            model_path = "worldmodel_mlp.pkl"
            print(f"Loading MLP world model from {model_path}...")
        elif os.path.exists("world_model_PongLSTM_checkpoint.pkl"):
            model_path = "world_model_PongLSTM_checkpoint.pkl"
            print(f"Loading LSTM world model from {model_path}...")

        if model_path:
            with open(model_path, "rb") as f:
                saved_data = pickle.load(f)
                dynamics_params = saved_data.get("params", saved_data.get("dynamics_params"))

                # Handle both old and new normalization formats
                if "normalization_stats" in saved_data:
                    # New format: checkpoint["normalization_stats"]["mean/std"]
                    normalization_stats = saved_data["normalization_stats"]
                elif "mean" in saved_data and "std" in saved_data:
                    # Old format: checkpoint["mean/std"] directly
                    normalization_stats = {
                        "mean": saved_data["mean"],
                        "std": saved_data["std"]
                    }
                    print("Detected old normalization format, converting to new format")
                else:
                    normalization_stats = None
                    print("Warning: No normalization stats found in checkpoint")

                # Load model architecture info (update outer scope variables)
                loaded_model_scale_factor = saved_data.get("model_scale_factor", loaded_model_scale_factor)
                loaded_use_deep = saved_data.get("use_deep", loaded_use_deep)
                loaded_model_type = saved_data.get("model_type", loaded_model_type)

                print(f"Model info from checkpoint: type={loaded_model_type}, scale={loaded_model_scale_factor}, use_deep={loaded_use_deep}")

                model_exists = True
                del saved_data
                gc.collect()

            # Load standalone reward predictor
            reward_predictor_path = "reward_predictor_standalone.pkl"
            if os.path.exists(reward_predictor_path):
                print(f"Loading standalone reward predictor from {reward_predictor_path}...")
                with open(reward_predictor_path, "rb") as f:
                    reward_data = pickle.load(f)
                    reward_predictor_params = reward_data["params"]
            else:
                print(f"Warning: No standalone reward predictor found at {reward_predictor_path}")
                reward_predictor_params = None

            # Always use experience_mlp.pkl for imagined rollouts
            experience_path = "experience_mlp.pkl"

            if not os.path.exists(experience_path):
                print(f"ERROR: {experience_path} not found!")
                print("Run: python MBRL/worldmodel_mlp.py collect 100")
                return

            print(f"Loading experience data from {experience_path}...")
            with open(experience_path, "rb") as f:
                saved_data = pickle.load(f)
                obs = saved_data["obs"]
                del saved_data
                gc.collect()
                print("Loaded model and experience data, freed saved_data from memory")
        # else:
        #     print("Train Model first")
        #     exit()

        # obs_shape = obs.shape[1:]
        key = jax.random.PRNGKey(42)
        # dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

            # Create environment
        game = JaxPong()
        env = AtariWrapper(game, sticky_actions=False, episodic_life=False, frame_stack_size=4)
        env = FlattenObservationWrapper(env)

        dummy_obs = flatten_obs(env.reset(jax.random.PRNGKey(0))[0], single_state=True)[0]

        actor_params = None
        critic_params = None

        if os.path.exists(f"{prefix}actor_params.pkl"):
            try:
                with open(f"{prefix}actor_params.pkl", "rb") as f:
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

        if os.path.exists(f"{prefix}critic_params.pkl"):
            try:
                with open(f"{prefix}critic_params.pkl", "rb") as f:
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
        # print(f"Actor parameters: {actor_param_count:,}")
        # print(f"Critic parameters: {critic_param_count:,}")

       

        if args.eval:
            # Use render parameter if provided, otherwise default to False
            render_eval = bool(args.render)
            evaluate_real_performance(actor_network, actor_params, render=render_eval, reward_predictor_params=reward_predictor_params, model_scale_factor=loaded_model_scale_factor, num_episodes=100)
            exit()


        #stuff to make it run without a model
        # if not model_exists:
        #     obs = jax.numpy.array(dummy_obs, dtype=jnp.float32)
        #     dynamics_params = None
        #     reward_predictor_params = None
        #     normalization_stats = None

        key, shuffle_key = jax.random.split(key)
        shuffled_obs = jax.random.permutation(shuffle_key, obs)

        # Free the original obs array, we only need shuffled_obs
        del obs
        gc.collect()


        print("Generating imagined rollouts...")

        (
            imagined_obs,
            imagined_actions,   # FIX: Corrected order to match return statement
            imagined_rewards,
            imagined_discounts,
            imagined_values,
            imagined_log_probs,
            total_valid_steps
        ) = rollout_func(
            dynamics_params=dynamics_params,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_network=actor_network,
            critic_network=critic_network,
            initial_observations=shuffled_obs[: training_params["num_rollouts"]],
            rollout_length=training_params["rollout_length"],
            normalization_stats=normalization_stats,
            discount=training_params["discount"],
            key=jax.random.PRNGKey(SEED),
            reward_predictor_params=reward_predictor_params,
            model_scale_factor=loaded_model_scale_factor,
        )

    






        if args.render:
            visualization_offset = 0

            # Render multiple imagined rollouts sequentially (one rollout after another)
            n_rollouts = int(args.render)
            n_rollouts = min(n_rollouts, imagined_obs.shape[1])
            if n_rollouts <= 0:
                print("No rollouts selected for rendering (args.render <= 0)")
            else:
                # imagined_obs has shape (T, B, F). Select first n_rollouts -> (T, B_sel, F)
                sel_obs = imagined_obs[:, :n_rollouts, :]

                # Reorder to (B_sel, T, F) so frames of each rollout are contiguous when flattened
                sel_obs = jnp.transpose(sel_obs, (1, 0, 2))

                # Flatten to (B_sel * T, F): rollout0 frames, then rollout1 frames, ...
                obs = sel_obs.reshape(sel_obs.shape[0] * sel_obs.shape[1], sel_obs.shape[2])

                # Prepare matching actions: imagined_actions has shape (T, B)
                sel_actions = imagined_actions[:, :n_rollouts]  # (T, B_sel)
                sel_actions = jnp.transpose(sel_actions, (1, 0)).reshape(-1)  # (B_sel * T,)

                compare_real_vs_model(
                    steps_into_future=0,
                    obs=obs,
                    num_steps=obs.shape[0],
                    actions=sel_actions,
                    frame_stack_size=4,
                    clock_speed=5,
                    model_scale_factor=loaded_model_scale_factor,
                    reward_predictor_params=None,
                    calc_score_based_reward=True,
                    rollout_length=training_params["rollout_length"],
                )

        print("\n=== REWARD STATISTICS ===")
        print(f"Reward stats: min={jnp.min(imagined_rewards):.4f}, max={jnp.max(imagined_rewards):.4f}, mean={jnp.mean(imagined_rewards):.4f}")
        print(f"Non-zero rewards: {jnp.sum(imagined_rewards != 0.0)} / {imagined_rewards.size}")

        # Show reward distribution
        rewards_flat = imagined_rewards.flatten()
        print(f"Rewards > 2.0: {jnp.sum(rewards_flat > 2.0)}")
        print(f"Rewards > 5.0: {jnp.sum(rewards_flat > 5.0)}")
        print(f"Rewards > 10.0: {jnp.sum(rewards_flat > 10.0)}")
        print(f"Rewards in [-1, 1]: {jnp.sum((rewards_flat >= -1.0) & (rewards_flat <= 1.0))}")
        print(f"Imagined rollouts shape: {imagined_obs.shape}")
        print("=" * 50)

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
            key=jax.random.PRNGKey(2000),
            lambda_=training_params["lambda_"],
            entropy_scale=training_params["entropy_scale"],
            target_kl=training_params["target_kl"],
            early_stopping_patience=training_params["early_stopping_patience"],
        )

        print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")

        def save_model_checkpoints(actor_params, critic_params, iteration, prefix=prefix):
            """Save parameters with consistent structure including iteration counter"""
            with open(f"{prefix}actor_params.pkl", "wb") as f:
                pickle.dump({"params": actor_params, "iteration": iteration + 1}, f)
            with open(f"{prefix}critic_params.pkl", "wb") as f:
                pickle.dump({"params": critic_params, "iteration": iteration + 1}, f)
            print(f"Saved actor, critic parameters (iteration {iteration})")

        action_probs = analyze_policy_behavior(actor_network, actor_params, imagined_obs)

        # Compute scalar metrics for logging
        mean_reward = float(jnp.mean(imagined_rewards))
        try:
            p = np.array(action_probs)
        except Exception:
            p = np.asarray(action_probs)

        # simple entropy and movement metrics
        entropy_val = -float(np.sum(p * np.log(p + 1e-12)))
        movement_prob = float(p[3] + p[4]) if p.size > 4 else 0.0
        most_likely = int(np.argmax(p))

        # Append a single-line log entry to training_log
        log_line = (
            f"iter={i}, mean_reward={mean_reward:.6f}, total_valid_steps={total_valid_steps}, "
            f"action_probs={p.tolist()}, entropy={entropy_val:.6f}, movement_prob={movement_prob:.6f}, "
            f"most_likely={most_likely}\n"
        )

        with open(f"{prefix}training_log", "a") as lf:
            lf.write(log_line)

        save_model_checkpoints(actor_params, critic_params, i, prefix=prefix)



        if i % training_params["retrain_interval"] == 0:
            # evaluate_real_performance(actor_network, actor_params, num_episodes=3, render=False, reward_predictor_params=reward_predictor_params, model_scale_factor=loaded_model_scale_factor)
            # and print result into training_log
            eval_rewards = evaluate_real_performance(actor_network, actor_params, num_episodes=10, render=False, reward_predictor_params=reward_predictor_params, model_scale_factor=loaded_model_scale_factor)
            eval_mean = float(np.mean(eval_rewards))
            
            eval_std = float(np.std(eval_rewards))
            with open(f"{prefix}training_log", "a") as lf:
                lf.write(f"eval_mean_reward={eval_mean:.6f}, eval_std_reward={eval_std:.6f}\n")
            if eval_mean >= 14.0:
                print(f"Achieved eval mean reward of {eval_mean:.2f}, stopping training early!")
                break
                # Retrain worldmodel every retrain_interval training runs
        if i % training_params["retrain_interval"] == 0 and rollout_func == generate_imagined_rollouts:   #activate this later
            print(f"\n{'='*60}")
            print(f"RETRAINING WORLDMODEL AFTER {i} TRAINING RUNS")
            print(f"{'='*60}\n")

            # Determine which actor to use based on rollout_style
            # Map 'model' -> 'imagined' for the actor filename
            actor_type = "imagined" if args.rollout_style == "model" else args.rollout_style

            # Collect fresh experience with trained actor
            print(f"Collecting fresh experience with {actor_type} actor...")
            os.system(f"python MBRL/worldmodel_mlp.py collect {training_params['wm_sample_size']} {actor_type}")

            # Retrain worldmodel
            print("Retraining worldmodel...")
            os.system(f"python MBRL/worldmodel_mlp.py train {training_params['wm_train_epochs']}")

            # Reload the updated worldmodel and get training error
            if os.path.exists("worldmodel_mlp.pkl"):
                with open("worldmodel_mlp.pkl", "rb") as f:
                    saved_data = pickle.load(f)
                    dynamics_params = saved_data.get("params", saved_data.get("dynamics_params"))
                    if "normalization_stats" in saved_data:
                        normalization_stats = saved_data["normalization_stats"]
                    # Get the training loss from the checkpoint
                    wm_training_loss = saved_data.get("loss", None)
                    print("Reloaded updated worldmodel!")
                    if wm_training_loss is not None:
                        print(f"World model training loss: {wm_training_loss:.6f}")

            print("Worldmodel retraining complete!")
            print(f"{'='*60}\n")
            with open(f"{prefix}training_log", "a") as lf:
                if wm_training_loss is not None:
                    lf.write(f"-------------------------------------- Retrained Model (loss={wm_training_loss:.6f}) --------------------------------------\n")
                else:
                    lf.write("-------------------------------------- Retrained Model --------------------------------------\n")



if __name__ == "__main__":
    rtpt = RTPT(name_initials="FH", experiment_name="OCActorCritic", max_iterations=3)

    rtpt.start()
    main()
