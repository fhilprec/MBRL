import argparse
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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





from worldmodelPong import compare_real_vs_model, get_enhanced_reward, print_full_array, collect_experience_sequential
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


def run_single_episode(episode_key, actor_params, actor_network, env, max_steps=10000):
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
            action = pi.sample(seed=action_key)
            
            # Step environment
            next_obs, next_state, reward, next_done, _ = env.step(state, action)
            next_flat_obs = flatten_obs(next_obs, single_state=True)[0]
            
            reward = jnp.array(improved_pong_reward(next_flat_obs, action, frame_stack_size=4), dtype=jnp.float32)


            old_score =  flat_obs[-5]-flat_obs[-1]
            new_score =  next_flat_obs[-5]-next_flat_obs[-1]

            score_reward = new_score - old_score
            score_reward = jnp.array(jnp.where(jnp.abs(score_reward) > 1, 0.0, score_reward))

            reward = reward + score_reward * 2 # to make actual score really important

            # Store transition with valid mask (valid = not done BEFORE this step)
            transition = (flat_obs, state, action, reward, ~done)
            
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
        lambda k: run_single_episode(k, actor_params, actor_network, env),
        in_axes=0
    )
    
    # After getting the vmapped results
    observations, actions, rewards, valid_mask, states, episode_length = vmapped_episode_fn(episode_keys)

    print(f"Episode lengths: {episode_length}")

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
        
        print(f"Episode {ep_idx+1}: {ep_length} valid steps")

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

    print(f"Rollout shapes: obs={obs_rollouts.shape}, actions={actions_rollouts.shape}, rewards={rewards_rollouts.shape}")
    
    # Compute values and log_probs
    all_obs_for_critic = obs_rollouts.reshape(-1, obs_rollouts.shape[-1])
    value_dists = critic_network.apply(critic_params, all_obs_for_critic)
    values = value_dists.mean().reshape(num_rollouts, rollout_length + 1)
    
    all_actions_flat = actions_rollouts.reshape(-1)
    all_obs_flat = obs_rollouts[:, :-1].reshape(-1, obs_rollouts.shape[-1])
    pis = actor_network.apply(actor_params, all_obs_flat)
    log_probs = pis.log_prob(all_actions_flat).reshape(num_rollouts, rollout_length)
    
    discounts = jnp.full_like(rewards_rollouts, discount)

    print(f"Rollouts shape: (, {num_rollouts}, {rollout_length+1}, {obs_rollouts.shape[-1]})")
    print(f"Before transpose - rewards: {rewards_rollouts.shape}, values: {values.shape}")

    # Transpose to (T, B, ...) format as expected by training function
    # Current format: (B, T, ...) where B=num_rollouts, T=rollout_length
    # Need format: (T, B, ...) where T=rollout_length, B=num_rollouts
    # NOTE: values needs T+1 timesteps for bootstrapping, others need T timesteps
    return (
        jnp.transpose(obs_rollouts[:, :-1], (1, 0, 2)),  # (B, T, F) -> (T, B, F)
        jnp.transpose(actions_rollouts, (1, 0)),         # (B, T) -> (T, B)
        jnp.transpose(rewards_rollouts, (1, 0)),         # (B, T) -> (T, B)
        jnp.transpose(discounts, (1, 0)),                # (B, T) -> (T, B)
        jnp.transpose(values, (1, 0)),                   # (B, T+1) -> (T+1, B) - KEEP all values!
        jnp.transpose(log_probs, (1, 0)),                # (B, T) -> (T, B)
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

    # print(
    #     f"Normalized targets - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    # )

    print(
        f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    )
    print(f"Targets shape: {targets.shape}, Values shape: {values.shape}")
    print(f"Raw advantages - Mean: {(targets - values[:-1]).mean():.4f}, Std: {(targets - values[:-1]).std():.4f}")
    # print(
    #     f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    # )
    # print(
    #     f"Lambda returns stats - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    # )
    # print(
    #     f"Lambda returns stats - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    # )

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

        if debug:
            jax.debug.print("Raw advantages: mean={mean}, std={std}, min={min}, max={max}",
                          mean=advantages.mean(), std=advantages.std(),
                          min=advantages.min(), max=advantages.max())
            jax.debug.print("Actions: shape={shape}, dtype={dtype}, min={min}, max={max}, first_10={first}",
                          shape=actions.shape, dtype=actions.dtype,
                          min=actions.min(), max=actions.max(),
                          first=actions[:10])
            jax.debug.print("Pi logits: shape={shape}, min={min}, max={max}, first_sample={first}",
                          shape=pi.logits.shape,
                          min=pi.logits.min(), max=pi.logits.max(),
                          first=pi.logits[0])
            jax.debug.print("Pi probs: first_sample={first}",
                          first=pi.probs[0])
            jax.debug.print("Log probs: mean={mean}, std={std}, min={min}, max={max}, first_10={first}",
                          mean=log_prob.mean(), std=log_prob.std(),
                          min=log_prob.min(), max=log_prob.max(),
                          first=log_prob[:10])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # REINFORCE objective: maximize log_prob * advantages
        # We want to MAXIMIZE this, so the loss should be NEGATIVE
        reinforce_obj = log_prob * jax.lax.stop_gradient(advantages)

        if debug:
            jax.debug.print("Normalized advantages: mean={mean}, std={std}",
                          mean=advantages.mean(), std=advantages.std())
            jax.debug.print("Reinforce obj: mean={mean}, std={std}, sum={sum}",
                          mean=reinforce_obj.mean(), std=reinforce_obj.std(),
                          sum=reinforce_obj.sum())

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


def evaluate_real_performance(actor_network, actor_params, num_episodes=5):
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

        while not done and step_count < 10000:

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

    training_runs = 1000

    training_params = {
        "action_dim": 6,
        "rollout_length": 20,
        "num_rollouts": 3000,
        "policy_epochs": 10,  # Max epochs, KL will stop earlier
        "actor_lr": 8e-5,  # Reduced significantly for smaller policy updates
        "critic_lr": 5e-4,  # Moderate critic learning rate
        "lambda_": 0.95,
        "entropy_scale": 0.01,  # Maintain exploration
        "discount": 0.95,
        "max_grad_norm": 0.5,  # Tight gradient clipping
        "target_kl": 0.15,  # Slightly relaxed to allow 2-3 epochs
        "early_stopping_patience": 100,
    }

    for i in range(training_runs):
        print(f"This is the {i}th iteration training the actor-critic")
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

        parser.add_argument("--eval", type=int, help="Specifies whether to run evaluation", default=0)
        args = parser.parse_args()

        if args.eval:
            evaluate_real_performance(actor_network, actor_params,  num_episodes=5)
            exit()

        #stuff to make it run without a model
        obs = jax.numpy.array(dummy_obs, dtype=jnp.float32)
        dynamics_params = None
        normalization_stats = None
        shuffled_obs = jax.random.permutation(jax.random.PRNGKey(SEED), obs)

        print("Generating imagined rollouts...")
        (
            imagined_obs,
            imagined_actions,   # FIX: Corrected order to match return statement
            imagined_rewards,
            imagined_discounts,
            imagined_values,
            imagined_log_probs,
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
            key=jax.random.PRNGKey(SEED),
        )

        # print(jnp.sum(dones_seq))
        # exit()

        

       

        print(imagined_obs.shape)

        if args.render:
            visualization_offset = 0
            for i in range(int(args.render)):

                compare_real_vs_model(
                    steps_into_future=0,
                    obs=imagined_obs[:, i, :],
                    actions=imagined_actions[:, i],
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
            key=jax.random.PRNGKey(2000),
            lambda_=training_params["lambda_"],
            entropy_scale=training_params["entropy_scale"],
            target_kl=training_params["target_kl"],
            early_stopping_patience=training_params["early_stopping_patience"],
        )

        print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")

        def save_model_checkpoints(actor_params, critic_params):
            """Save parameters with consistent structure"""
            with open("actor_params.pkl", "wb") as f:
                pickle.dump({"params": actor_params}, f)
            with open("critic_params.pkl", "wb") as f:
                pickle.dump({"params": critic_params}, f)
            print("Saved actor, critic parameters")

        analyze_policy_behavior(actor_network, actor_params, imagined_obs)

        save_model_checkpoints(actor_params, critic_params)


if __name__ == "__main__":
    rtpt = RTPT(name_initials="FH", experiment_name="OCActorCritic", max_iterations=3)

    rtpt.start()
    main()
