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
        traj_rewards[:-1],  # Remove last reward
        traj_values[:-1],   # Remove last value  
        traj_discounts[:-1], # Remove last discount
        bootstrap,
        lambda_=lambda_,
        axis=0,
    )
    return targets

def manual_lambda_returns_reference(rewards, values, discounts, bootstrap, lambda_=0.95):
    """
    Reference implementation following DreamerV2 paper equation (4):
    V^λ_t = r_t + γ_t * [(1-λ) * v(s_{t+1}) + λ * V^λ_{t+1}]
    
    Computed backwards from the end.
    """
    T = len(rewards)
    lambda_returns = jnp.zeros(T)
    
    # Start from the end, work backwards
    next_lambda_return = bootstrap
    
    for t in reversed(range(T)):
        # Get next value: v(s_{t+1})
        if t == T - 1:
            next_value = bootstrap  # Last timestep uses bootstrap
        else:
            next_value = values[t + 1]
        
        # Compute λ-return: V^λ_t = r_t + γ_t * [(1-λ) * v(s_{t+1}) + λ * V^λ_{t+1}]
        lambda_return = rewards[t] + discounts[t] * (
            (1 - lambda_) * next_value + lambda_ * next_lambda_return
        )
        
        lambda_returns = lambda_returns.at[t].set(lambda_return)
        next_lambda_return = lambda_return
    
    return lambda_returns

def test_simple_case():
    """Test with a very simple 3-step trajectory"""
    print("=== TEST 1: Simple 3-step trajectory ===")
    
    # Simple trajectory: 3 timesteps
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
    
    # Your implementation
    your_targets = compute_trajectory_targets(rewards, values, discounts, lambda_)
    
    # Reference implementation
    # For targets, we use rewards[:-1], values[:-1], discounts[:-1] with bootstrap=values[-1]
    ref_targets = manual_lambda_returns_reference(
        rewards[:-1], values[:-1], discounts[:-1], values[-1], lambda_
    )
    
    print(f"\nResults:")
    print(f"  Your targets:      {your_targets}")
    print(f"  Reference targets: {ref_targets}")
    print(f"  Max difference:    {jnp.abs(your_targets - ref_targets).max():.8f}")
    print(f"  Match? {jnp.allclose(your_targets, ref_targets, atol=1e-6)}")
    
    # Manual step-by-step verification for reference
    print(f"\n  Manual step-by-step calculation:")
    bootstrap = values[-1]  # 1.0
    
    # Working backwards for rewards[:-1] = [1.0, 0.5], values[:-1] = [2.0, 1.5]
    # t=1: V^λ_1 = r_1 + γ_1 * [(1-λ) * bootstrap + λ * bootstrap] = 0.5 + 0.9 * 1.0 = 1.4
    v_lambda_1 = rewards[1] + discounts[1] * bootstrap
    print(f"    t=1: {rewards[1]} + {discounts[1]} * {bootstrap} = {v_lambda_1}")
    
    # t=0: V^λ_0 = r_0 + γ_0 * [(1-λ) * v_1 + λ * V^λ_1] = 1.0 + 0.9 * [0.2 * 1.5 + 0.8 * 1.4]
    blended_next = (1 - lambda_) * values[1] + lambda_ * v_lambda_1
    v_lambda_0 = rewards[0] + discounts[0] * blended_next
    print(f"    t=0: {rewards[0]} + {discounts[0]} * ({1-lambda_} * {values[1]} + {lambda_} * {v_lambda_1})")
    print(f"         = {rewards[0]} + {discounts[0]} * {blended_next} = {v_lambda_0}")
    
    manual_result = jnp.array([v_lambda_0, v_lambda_1])
    print(f"  Manual result:     {manual_result}")
    
    return jnp.allclose(your_targets, ref_targets, atol=1e-6)

def test_edge_cases():
    """Test edge cases"""
    print("\n=== TEST 2: Edge cases ===")
    
    # Test with all zeros
    print("All zeros:")
    rewards = jnp.zeros(5)
    values = jnp.zeros(5)
    discounts = jnp.ones(5) * 0.99
    
    your_result = compute_trajectory_targets(rewards, values, discounts)
    ref_result = manual_lambda_returns_reference(rewards[:-1], values[:-1], discounts[:-1], values[-1])
    
    print(f"  Your result: {your_result}")
    print(f"  Reference:   {ref_result}")
    print(f"  Match? {jnp.allclose(your_result, ref_result, atol=1e-6)}")
    
    # Test with lambda=0 (should just be 1-step TD targets)
    print("\nLambda=0 (1-step TD):")
    rewards = jnp.array([1.0, 2.0, 0.5])
    values = jnp.array([1.0, 1.5, 2.0])
    discounts = jnp.array([0.9, 0.9, 0.9])
    
    your_result = compute_trajectory_targets(rewards, values, discounts, lambda_=0.0)
    # With lambda=0: V^λ_t = r_t + γ_t * v(s_{t+1})
    expected = rewards[:-1] + discounts[:-1] * values[1:]  # 1-step TD targets
    
    print(f"  Your result: {your_result}")
    print(f"  Expected:    {expected}")
    print(f"  Match? {jnp.allclose(your_result, expected, atol=1e-6)}")
    
    # Test with lambda=1 (Monte Carlo returns)
    print("\nLambda=1 (Monte Carlo):")
    your_result = compute_trajectory_targets(rewards, values, discounts, lambda_=1.0)
    # With lambda=1: should be discounted sum of rewards + bootstrap
    bootstrap = values[-1]
    # MC return for t=0: r_0 + γ_0 * r_1 + γ_0 * γ_1 * bootstrap
    # MC return for t=1: r_1 + γ_1 * bootstrap
    mc_0 = rewards[0] + discounts[0] * (rewards[1] + discounts[1] * bootstrap)
    mc_1 = rewards[1] + discounts[1] * bootstrap
    expected = jnp.array([mc_0, mc_1])
    
    print(f"  Your result: {your_result}")
    print(f"  Expected:    {expected}")
    print(f"  Match? {jnp.allclose(your_result, expected, atol=1e-6)}")

def test_realistic_pong_data():
    """Test with realistic Pong-like data"""
    print("\n=== TEST 3: Realistic Pong data ===")
    
    # Simulate a trajectory where player gets closer to ball, then hits it
    T = 10
    rewards = jnp.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 0.8, 0.5, 0.2, 0.0])
    values = jnp.array([1.0, 1.2, 1.5, 1.8, 2.2, 2.5, 2.0, 1.5, 1.0, 0.8])
    discounts = jnp.full(T, 0.95)
    
    print(f"Trajectory length: {T}")
    print(f"Rewards: {rewards}")
    print(f"Values: {values}")
    
    your_result = compute_trajectory_targets(rewards, values, discounts)
    ref_result = manual_lambda_returns_reference(rewards[:-1], values[:-1], discounts[:-1], values[-1])
    
    print(f"\nTargets shape: {your_result.shape} (should be {T-1})")
    print(f"Your targets: {your_result}")
    print(f"Reference:    {ref_result}")
    print(f"Max diff: {jnp.abs(your_result - ref_result).max():.8f}")
    print(f"Match? {jnp.allclose(your_result, ref_result, atol=1e-6)}")
    
    # Check that targets are reasonable
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
            # value = nn.Dense(1, kernel_init=orthogonal(0.001), bias_init=constant(0.0))(x)
            value = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(35.0))(x)

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

            reward = simple_movement_reward(next_obs, frame_stack_size=4)

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
            reward = simple_movement_reward(next_obs, action, frame_stack_size=4)
            




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

# 4. Fix the final metrics calculation
def safe_final_metrics(metrics_history):
    """Safely compute final metrics even if history is empty"""
    if not metrics_history:
        return {
            "actor_loss": 0.0,
            "policy_loss": 0.0, 
            "entropy_loss": 0.0,
            "noop_penalty": 0.0,
            "entropy": 0.0,
            "advantages_mean": 0.0,
            "critic_loss": 0.0,
            "critic_mean": 0.0,
        }
    
    # Get all keys from the first metric dict
    all_keys = metrics_history[0].keys()
    final_metrics = {}
    
    for key in all_keys:
        values = [m[key] for m in metrics_history]
        final_metrics[key] = jnp.mean(jnp.array(values))
    
    return final_metrics

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
    target_kl=0.01, early_stopping_patience=5
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

    best_loss = float('inf')
    patience_counter = 0

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




    #Comment this in to prove compute_trajectory_targets is working
    # print("Starting Tests for compute_trajectory_targets function...")
    # run_all_tests()


    




    print(
        f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    )
    print(f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")

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
        
        # Use reasonable clipping bounds based on your actual data
        predicted_values = jnp.clip(predicted_values, -50.0, 50.0)
        targets = jnp.clip(targets, -50.0, 50.0)
        
        # Keep the Huber loss but with larger threshold
        diff = predicted_values - targets
        huber_loss = jnp.where(
            jnp.abs(diff) < 5.0,  # Larger threshold
            0.5 * diff**2,
            5.0 * jnp.abs(diff) - 12.5
        )
        
        loss = jnp.mean(huber_loss)
        
        # Keep the L2 regularization
        l2_reg = 1e-6 * sum(jnp.sum(p**2) for p in jax.tree.leaves(critic_params))
        
        total_loss = loss + l2_reg
        
        return total_loss, {
            "critic_loss": total_loss, 
            "critic_mean": jnp.mean(predicted_values),
            "critic_std": jnp.std(predicted_values)
        }

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
        # mask = jnp.array([1, 1, 1, 0, 0, 1], dtype=pi.probs.dtype)
        noop_penalty = 0
        # noop_penalty = 1 * jnp.sum(pi.probs * mask)

        # movement_mask = jnp.array([0.01, 0.01, 0.01, 1.0, 1.0, 0.01])  # Heavily favor movement
        # movement_penalty = -jnp.sum(pi.probs * (1 - movement_mask)) * 2.0  

        entropy_loss = -entropy_scale * jnp.mean(entropy)

        total_loss = policy_loss + entropy_loss 

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

               # Debug critic predictions vs targets
        if epoch == 0:  # Only on first epoch
            current_preds = critic_network.apply(critic_state.params, obs_shuffled[:100])  # Sample 100
            sample_targets = targets_shuffled[:100]
            
            print(f"\nCritic debugging (epoch {epoch}):")
            print(f"  Target stats: mean={sample_targets.mean():.3f}, std={sample_targets.std():.3f}")
            print(f"  Prediction stats: mean={current_preds.mean():.3f}, std={current_preds.std():.3f}")
            print(f"  Scale difference: {abs(sample_targets.mean() - current_preds.mean()):.3f}")
            
            # Check initial loss
            initial_loss = jnp.mean((current_preds - sample_targets) ** 2)
            print(f"  Initial MSE loss: {initial_loss:.3f}")


        # Compute old policy distribution for KL divergence
        old_pi = actor_network.apply(actor_params, obs_shuffled)
        old_log_probs_new = old_pi.log_prob(actions_shuffled)

        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_state.params, obs_shuffled, targets_shuffled)


        # Check for gradient explosion
        critic_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(critic_grads)))
        if critic_grad_norm > 10.0:
            print(f"Warning: Large critic gradients ({critic_grad_norm:.2f}) at epoch {epoch}")
        




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

         # Compute KL divergence for monitoring
        new_pi = actor_network.apply(actor_state.params, obs_shuffled)
        new_log_probs = new_pi.log_prob(actions_shuffled)
        kl_div = jnp.mean(old_log_probs_new - new_log_probs)

        # Early stopping based on KL divergence
        if kl_div > target_kl:
            print(f"Early stopping at epoch {epoch}: KL divergence {kl_div:.6f} > {target_kl}")
            break

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

        # Track best performance
        total_loss = actor_loss + critic_loss
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: No improvement for {early_stopping_patience} epochs")
            break
        
        # # Enhanced logging
        # if epoch % 5 == 0:
        #     print(f"Epoch {epoch}:")
        #     print(f"  Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        #     print(f"  Entropy: {actor_metrics['entropy']:.4f}, KL Div: {kl_div:.6f}")
        #     print(f"  Advantage Mean: {actor_metrics.get('advantages_mean', 0):.4f}")
        #     print(f"  Critic Pred Mean: {critic_metrics['critic_mean']:.4f}")
        #     print(f"  Target Mean: {jnp.mean(targets_shuffled):.4f}")

        final_metrics = safe_final_metrics(metrics_history)
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
            temperature = 0.1  # Lower = more deterministic
            scaled_logits = pi.logits / temperature
            scaled_pi = distrax.Categorical(logits=scaled_logits)
            action = scaled_pi.sample(seed=jax.random.PRNGKey(step_count))
            if step_count % 100 == 0:
                obs_flat, _ = flatten_obs(obs, single_state=True)
                training_reward = simple_movement_reward(obs_flat, action, frame_stack_size=4)
                print(f"  Training reward would be: {training_reward:.3f}")

                obs_flat, _ = flatten_obs(obs, single_state=True)

                last_obs = obs_flat[(4 - 1)::4]
                # Extract positions (adjust indices as needed)
                player_y = last_obs[1]
                ball_y = last_obs[9] 
                print(f"  Player Y: {player_y:.2f}, Ball Y: {ball_y:.2f}, Distance: {abs(ball_y-player_y):.2f}")

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

    # Sample some observations - flatten to 2D first
    sample_obs = observations.reshape(-1, observations.shape[-1])[:1000]  # Take first 1000 obs
    print(sample_obs.shape)
    
    # Get action probabilities
    pi = actor_network.apply(actor_params, sample_obs)
    action_probs = jnp.mean(pi.probs, axis=0)
    
    print("\n=== POLICY ANALYSIS ===")
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    
    for i in range(len(action_names)):
        prob_val = float(action_probs[i])  # Index into the array first
        print(f"Action {i} ({action_names[i]}): {prob_val:.3f}")
    
    entropy_val = float(jnp.mean(pi.entropy()))
    movement_prob = float(action_probs[3] + action_probs[4])
    most_likely_idx = int(jnp.argmax(action_probs))
    
    print(f"Entropy: {entropy_val:.3f}")
    print(f"Favors movement: {movement_prob:.3f}")
    print(f"Most likely action: {most_likely_idx} ({action_names[most_likely_idx]})")
    
    return action_probs

def main():

    training_runs = 3

    training_params = {
        'action_dim': 6,
        'rollout_length': 45,
        'num_rollouts': 1600,
        'policy_epochs': 50,      # More epochs since we're stopping early
        'actor_lr': 5e-6,         # Even lower
        'critic_lr': 1e-6,        # Even lower  
        'lambda_': 0.95,
        'entropy_scale': 5e-4,    # Much lower entropy regularization
        'discount': 0.95,
        'max_grad_norm': 0.1,     # Extremely aggressive clipping
        'target_kl': 0.2,         # More lenient to allow longer training
        'early_stopping_patience': 15
    }


    for i in range(training_runs):

        parser = argparse.ArgumentParser(description="DreamerV2 Pong agent")

        actor_network = create_dreamerv2_actor(training_params['action_dim'])
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
            initial_observations=shuffled_obs[:training_params['num_rollouts']],
            rollout_length=training_params['rollout_length'],
            normalization_stats=normalization_stats,
            discount=training_params['discount'],
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
                num_epochs=training_params['policy_epochs'],
                actor_lr=training_params['actor_lr'],
                critic_lr=training_params['critic_lr'],
                key=jax.random.PRNGKey(2000),
                lambda_=training_params['lambda_'],
                entropy_scale=training_params['entropy_scale'],
                target_kl=training_params['target_kl'],
                early_stopping_patience=training_params['early_stopping_patience'],
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

        analyze_policy_behavior(actor_network, actor_params, imagined_obs)

        save_model_checkpoints(actor_params, critic_params, target_critic_params)
    
    # evaluate_real_performance(
    #         actor_network, actor_params, obs_shape, num_episodes=1
    #     )
    # exit()


if __name__ == "__main__":
    rtpt = RTPT(name_initials="FH", experiment_name="DreamerV2Agent", max_iterations=3)
    rtpt.start()
    main()
