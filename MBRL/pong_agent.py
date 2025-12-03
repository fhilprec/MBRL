import argparse
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import getpass

user = getpass.getuser()

if user == "fhilprecht":
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


from worldmodel_mlp import (
    compare_real_vs_model
)
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


SEED = 42


def flatten_obs(
    state, single_state: bool = False, is_list=False
) -> Tuple[jnp.ndarray, Any]:

    if isinstance(state, jnp.ndarray):
        if state.ndim == 1 and state.shape[0] == 48:

            return state, None
        elif state.ndim == 2 and state.shape[-1] == 48:

            return state, None

    if type(state) == list:
        flat_states = []

        for s in state:

            if isinstance(s, jnp.ndarray) and s.shape[0] == 48:
                flat_states.append(s)
            else:
                flat_state, _ = jax.flatten_util.ravel_pytree(s)

                flat_states.append(flat_state[:-8])
        flat_states = jnp.stack(flat_states, axis=0)
        print(flat_states.shape)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)

        return flat_state[:-8], unflattener

    batch_shape = (
        state.player_x.shape[0]
        if hasattr(state, "player_x")
        else state.paddle_y.shape[0]
    )

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)

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
                1,
                kernel_init=orthogonal(0.01),
                bias_init=constant(-1.0),
            )(x)

            mean = jnp.squeeze(mean, axis=-1)
            log_std = jnp.squeeze(log_std, axis=-1)

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

    if normalization_stats:
        state_mean = normalization_stats["mean"]
        state_std = normalization_stats["std"]
    else:
        state_mean = 0
        state_std = 1

    world_model = PongMLPDeep(model_scale_factor)

    num_trajectories = initial_observations.shape[0]

    dummy_action = jnp.zeros(1, dtype=jnp.int32)

    def init_lstm_state(obs):
        normalized_obs = (obs - state_mean) / state_std
        _, lstm_state = world_model.apply(
            dynamics_params, jax.random.PRNGKey(0), normalized_obs, dummy_action, None
        )
        return lstm_state

    initial_lstm_states = jax.vmap(init_lstm_state)(initial_observations)

    @jax.jit
    def single_trajectory_rollout(cur_obs, subkey, initial_lstm_state):
        """Generate a single trajectory starting from cur_obs."""

        def rollout_step(carry, step_idx):
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
            next_obs = jnp.round(next_obs).squeeze().astype(obs.dtype)

            reward = improved_pong_reward(next_obs, action, frame_stack_size=4)

            discount_factor = jnp.array(discount)

            step_data = (next_obs, action, reward, discount_factor, value, log_prob)
            new_carry = (key, next_obs, new_lstm_state)

            return new_carry, step_data

        cur_obs = cur_obs.astype(jnp.float32)
        init_carry = (subkey, cur_obs, initial_lstm_state)

        step_indices = jnp.arange(rollout_length)
        _, trajectory_data = lax.scan(rollout_step, init_carry, step_indices)

        (
            next_obs_seq,
            actions_seq,
            rewards_seq,
            discounts_seq,
            values_seq,
            log_probs_seq,
        ) = trajectory_data

        initial_value_dist = critic_network.apply(critic_params, cur_obs)
        initial_value = initial_value_dist.mean()

        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq], axis=0)
        actions = jnp.concatenate(
            [jnp.zeros_like(actions_seq[0])[None, ...], actions_seq], axis=0
        )
        rewards = jnp.concatenate([jnp.array([0.0]), rewards_seq], axis=0)
        discounts = jnp.concatenate([jnp.array([discount]), discounts_seq], axis=0)
        values = jnp.concatenate([initial_value[None, ...], values_seq], axis=0)
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq], axis=0)

        total_steps = len(observations)

        return observations, actions, rewards, discounts, values, log_probs, total_steps

    keys = jax.random.split(key, num_trajectories)

    rollout_fn = jax.vmap(single_trajectory_rollout, in_axes=(0, 0, 0))

    batch_size = 100
    num_batches = (num_trajectories + batch_size - 1) // batch_size

    all_observations = []
    all_actions = []
    all_rewards = []
    all_discounts = []
    all_values = []
    all_log_probs = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_trajectories)

        batch_lstm_states = jax.tree_util.tree_map(
            lambda x: x[start_idx:end_idx], initial_lstm_states
        )

        (
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_discounts,
            batch_values,
            batch_log_probs,
            total_steps,
        ) = rollout_fn(
            initial_observations[start_idx:end_idx],
            keys[start_idx:end_idx],
            batch_lstm_states,
        )

        all_observations.append(batch_obs)
        all_actions.append(batch_actions)
        all_rewards.append(batch_rewards)
        all_discounts.append(batch_discounts)
        all_values.append(batch_values)
        all_log_probs.append(batch_log_probs)

    observations = jnp.concatenate(all_observations, axis=0)
    actions = jnp.concatenate(all_actions, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)
    discounts = jnp.concatenate(all_discounts, axis=0)
    values = jnp.concatenate(all_values, axis=0)
    log_probs = jnp.concatenate(all_log_probs, axis=0)

    return (
        jnp.transpose(observations[:, :-1], (1, 0, 2)),
        jnp.transpose(actions[:, 1:], (1, 0)),
        jnp.transpose(rewards[:, 1:], (1, 0)),
        jnp.transpose(discounts[:, 1:], (1, 0)),
        jnp.transpose(values, (1, 0)),
        jnp.transpose(log_probs[:, 1:], (1, 0)),
        0,
    )


def run_single_episode(
    episode_key,
    actor_params,
    actor_network,
    env,
    max_steps=10000,
    reward_predictor_params=None,
    model_scale_factor=MODEL_SCALE_FACTOR,
    use_score_reward=False,
    inference=False,
):
    """Run one complete episode using JAX scan with masking."""
    reset_key, step_key = jax.random.split(episode_key)
    obs, state = env.reset(reset_key)

    def step_fn(carry, _):
        rng, obs, state, done = carry

        def continue_step(_):

            rng_new, action_key = jax.random.split(rng)
            flat_obs, _ = flatten_obs(obs, single_state=True)
            pi = actor_network.apply(actor_params, flat_obs)
            if not inference:
                action = pi.sample(seed=action_key)
            else:

                temperature = 0.05
                scaled_logits = pi.logits / temperature
                pi_temp = distrax.Categorical(logits=scaled_logits)
                action = pi_temp.sample(seed=action_key)

            next_obs, next_state, reward, next_done, _ = env.step(state, action)
            next_flat_obs = flatten_obs(next_obs, single_state=True)[0]

            old_player_score = state.env_state.player_score
            old_enemy_score = state.env_state.enemy_score
            new_player_score = next_state.env_state.player_score
            new_enemy_score = next_state.env_state.enemy_score

            score_reward = (new_player_score - old_player_score) - (
                new_enemy_score - old_enemy_score
            )

            score_reward = jnp.where(jnp.abs(score_reward) > 1, 0.0, score_reward)

            if use_score_reward:

                final_reward = jnp.array(score_reward, dtype=jnp.float32)
            else:

                improved_reward = improved_pong_reward(
                    next_flat_obs, action, frame_stack_size=4
                )

                reward_predictor_reward = jnp.array(0.0, dtype=jnp.float32)
                if reward_predictor_params is not None:
                    reward_model = RewardPredictorMLPTransition(1)

                    rng_reward = jax.random.PRNGKey(0)
                    predicted_reward = reward_model.apply(
                        reward_predictor_params,
                        rng_reward,
                        flat_obs[None, :],
                        jnp.array([action]),
                        next_flat_obs[None, :],
                    )

                    predicted_reward_clipped = jnp.round(
                        jnp.clip(jnp.squeeze(predicted_reward * (4 / 3) / 2), -1.0, 1.0)
                    )

                else:
                    reward_predictor_reward = jnp.array(0.0, dtype=jnp.float32)

                final_reward = jnp.array(
                    improved_pong_reward(next_flat_obs, action, frame_stack_size=4),
                    dtype=jnp.float32,
                )

            transition = (flat_obs, state, action, final_reward, ~done)

            return (rng_new, next_flat_obs, next_state, next_done), transition

        def skip_step(_):

            flat_obs, _ = flatten_obs(obs, single_state=True)

            dummy_action = jnp.array(0, dtype=jnp.int32)
            dummy_reward = jnp.array(0.0, dtype=jnp.float32)
            dummy_valid = jnp.array(False, dtype=jnp.bool_)

            dummy_transition = (
                flat_obs,
                state,
                dummy_action,
                dummy_reward,
                dummy_valid,
            )
            return (rng, flat_obs, state, done), dummy_transition

        return jax.lax.cond(done, skip_step, continue_step, None)

    flattened_init_obs = flatten_obs(obs, single_state=True)[0]

    initial_carry = (step_key, flattened_init_obs, state, jnp.array(False))
    _, transitions = jax.lax.scan(step_fn, initial_carry, None, length=max_steps)

    observations, states, actions, rewards, valid_mask = transitions

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
    initial_observations=None,
    num_rollouts: int = 3000,
    reward_predictor_params: Any = None,
    model_scale_factor: int = MODEL_SCALE_FACTOR,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Collect episodes with JAX vmap and reshape to (rollout_length, num_rollouts, features)."""

    game = JaxPong()
    env = AtariWrapper(
        game, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    env = FlattenObservationWrapper(env)

    episode_keys = jax.random.split(key, num_episodes)

    vmapped_episode_fn = jax.vmap(
        lambda k: run_single_episode(
            k,
            actor_params,
            actor_network,
            env,
            reward_predictor_params=reward_predictor_params,
            model_scale_factor=model_scale_factor,
        ),
        in_axes=0,
    )

    observations, actions, rewards, valid_mask, states, episode_length = (
        vmapped_episode_fn(episode_keys)
    )

    all_valid_obs = []
    all_valid_actions = []
    all_valid_rewards = []
    all_valid_states = []

    for ep_idx in range(num_episodes):
        ep_length = int(episode_length[ep_idx])

        valid_obs = observations[ep_idx, :ep_length]
        valid_actions = actions[ep_idx, :ep_length]
        valid_rewards = rewards[ep_idx, :ep_length]

        all_valid_obs.append(valid_obs)
        all_valid_actions.append(valid_actions)
        all_valid_rewards.append(valid_rewards)

    all_obs = jnp.concatenate(all_valid_obs, axis=0)
    all_actions = jnp.concatenate(all_valid_actions, axis=0)
    all_rewards = jnp.concatenate(all_valid_rewards, axis=0)

    total_steps = len(all_obs)
    num_valid_starts = total_steps - rollout_length
    if num_valid_starts < num_rollouts:
        num_rollouts = num_valid_starts

    rng = jax.random.PRNGKey(42)
    start_indices = jax.random.choice(
        rng, num_valid_starts, shape=(num_rollouts,), replace=False
    )

    obs_rollouts = jnp.stack(
        [all_obs[i : i + rollout_length + 1] for i in start_indices]
    )
    actions_rollouts = jnp.stack(
        [all_actions[i : i + rollout_length] for i in start_indices]
    )
    rewards_rollouts = jnp.stack(
        [all_rewards[i : i + rollout_length] for i in start_indices]
    )

    all_obs_for_critic = obs_rollouts.reshape(-1, obs_rollouts.shape[-1])
    value_dists = critic_network.apply(critic_params, all_obs_for_critic)
    values = value_dists.mean().reshape(num_rollouts, rollout_length + 1)

    all_actions_flat = actions_rollouts.reshape(-1)
    all_obs_flat = obs_rollouts[:, :-1].reshape(-1, obs_rollouts.shape[-1])
    pis = actor_network.apply(actor_params, all_obs_flat)
    log_probs = pis.log_prob(all_actions_flat).reshape(num_rollouts, rollout_length)

    discounts = jnp.full_like(rewards_rollouts, discount)

    total_valid_steps = int(total_steps)

    return (
        jnp.transpose(obs_rollouts[:, :-1], (1, 0, 2)),
        jnp.transpose(actions_rollouts, (1, 0)),
        jnp.transpose(rewards_rollouts, (1, 0)),
        jnp.transpose(discounts, (1, 0)),
        jnp.transpose(values, (1, 0)),
        jnp.transpose(log_probs, (1, 0)),
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

        bootstrap = traj_values[-1]

        targets = lambda_return_dreamerv2(
            traj_rewards,
            traj_values[:-1],
            traj_discounts,
            bootstrap,
            lambda_=lambda_,
            axis=0,
        )
        return targets

    targets = jax.vmap(compute_trajectory_targets, in_axes=(1, 1, 1), out_axes=1)(
        rewards, values, discounts
    )
    targets_mean = targets.mean()
    targets_std = targets.std()
    targets_normalized = (targets - targets_mean) / (targets_std + 1e-8)

    observations_flat = observations.reshape(T * B, -1)
    actions_flat = actions.reshape(T * B)
    targets_flat = targets.reshape(T * B)
    values_flat = values[:-1].reshape(T * B)
    old_log_probs_flat = log_probs.reshape(T * B)

    def critic_loss_fn(critic_params, obs, targets):
        dist = critic_network.apply(critic_params, obs)
        loss = -jnp.mean(dist.log_prob(targets))
        return loss, {"critic_loss": loss, "critic_mean": dist.mean()}

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

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        reinforce_obj = log_prob * jax.lax.stop_gradient(advantages)

        entropy_bonus = entropy_scale * entropy

        total_objective = reinforce_obj + entropy_bonus

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

        old_pi = actor_network.apply(actor_params, obs_shuffled)
        old_log_probs_new = old_pi.log_prob(actions_shuffled)

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
            debug=False,
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


def evaluate_real_performance(
    actor_network,
    actor_params,
    num_episodes=10,
    render=False,
    reward_predictor_params=None,
    model_scale_factor=MODEL_SCALE_FACTOR,
):
    """Evaluate the trained policy in the real Pong environment using JAX scan."""
    from jaxatari.games.jax_pong import JaxPong

    env = JaxPong()
    env = AtariWrapper(
        env, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )

    seed = int(__import__("time").time() * 1000) % (2**31)
    rng = jax.random.PRNGKey(seed)

    episode_keys = jax.random.split(rng, num_episodes)

    vmapped_episode_fn = jax.vmap(
        lambda k: run_single_episode(
            k,
            actor_params,
            actor_network,
            env,
            reward_predictor_params=reward_predictor_params,
            model_scale_factor=model_scale_factor,
            use_score_reward=True,
            inference=True,
        ),
        in_axes=0,
    )

    observations, actions, rewards, valid_mask, states, episode_lengths = (
        vmapped_episode_fn(episode_keys)
    )

    total_rewards = []
    all_obs = []
    all_actions = []

    for ep_idx in range(num_episodes):
        ep_length = int(episode_lengths[ep_idx])

        ep_reward = float(jnp.sum(rewards[ep_idx, :ep_length]))
        total_rewards.append(ep_reward)

        print(
            f"Episode {ep_idx + 1}: Final reward: {ep_reward:.3f}, Steps: {ep_length}"
        )

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

    if render and len(all_obs) > 0:

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

    sample_obs = observations.reshape(-1, observations.shape[-1])[:1000]

    pi = actor_network.apply(actor_params, sample_obs)
    action_probs = jnp.mean(pi.probs, axis=0)

    return action_probs


def main():

    training_runs = 100000
    model_scale_factor = MODEL_SCALE_FACTOR

    training_params = {
        "action_dim": 6,
        "rollout_length": 7,
        "num_rollouts": 30000,
        "policy_epochs": 10,
        "actor_lr": 8e-5,
        "critic_lr": 5e-4,
        "lambda_": 0.95,
        "entropy_scale": 0.01,
        "discount": 0.95,
        "max_grad_norm": 0.5,
        "target_kl": 0.15,
        "early_stopping_patience": 100,
        "retrain_interval": 50,
        "wm_sample_size": 500,
        "wm_train_epochs": 20,
    }

    parser = argparse.ArgumentParser(description="DreamerV2 Pong agent")
    parser.add_argument(
        "--eval", type=bool, help="Specifies whether to run evaluation", default=0
    )
    parser.add_argument(
        "--render", type=int, help="Specifies whether to run rendering", default=0
    )
    parser.add_argument(
        "--rollout_style",
        type=str,
        help="Specifies whether to use 'model' or 'real' rollouts",
        default="real",
    )
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

        actor_network = create_dreamerv2_actor(training_params["action_dim"])
        critic_network = create_dreamerv2_critic()

        model_path = None
        loaded_model_scale_factor = MODEL_SCALE_FACTOR
        loaded_use_deep = True
        loaded_model_type = "PongMLPDeep"

        model_path = "worldmodel_mlp.pkl"

        if model_path:
            with open(model_path, "rb") as f:
                saved_data = pickle.load(f)
                dynamics_params = saved_data.get(
                    "params", saved_data.get("dynamics_params")
                )

                normalization_stats = saved_data["normalization_stats"]

                loaded_model_scale_factor = saved_data.get(
                    "model_scale_factor", loaded_model_scale_factor
                )
                loaded_use_deep = saved_data.get("use_deep", loaded_use_deep)
                loaded_model_type = saved_data.get("model_type", loaded_model_type)

                model_exists = True
                del saved_data
                gc.collect()

            reward_predictor_params = None

            experience_path = "experience_mlp.pkl"

            if not os.path.exists(experience_path):
                print(f"ERROR: {experience_path} not found!")
                print("Run: python MBRL/worldmodel_mlp.py collect 100")
                return

            with open(experience_path, "rb") as f:
                saved_data = pickle.load(f)
                obs = saved_data["obs"]
                del saved_data
                gc.collect()

        key = jax.random.PRNGKey(42)

        game = JaxPong()
        env = AtariWrapper(
            game, sticky_actions=False, episodic_life=False, frame_stack_size=4
        )
        env = FlattenObservationWrapper(env)

        dummy_obs = flatten_obs(env.reset(jax.random.PRNGKey(0))[0], single_state=True)[
            0
        ]

        actor_params = None
        critic_params = None

        if os.path.exists(f"{prefix}actor_params.pkl"):
            try:
                with open(f"{prefix}actor_params.pkl", "rb") as f:
                    saved_data = pickle.load(f)
                    actor_params = saved_data.get("params", saved_data)
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

        if args.eval:

            render_eval = bool(args.render)
            evaluate_real_performance(
                actor_network,
                actor_params,
                render=render_eval,
                reward_predictor_params=reward_predictor_params,
                model_scale_factor=loaded_model_scale_factor,
                num_episodes=1,
            )
            exit()

        key, shuffle_key = jax.random.split(key)
        shuffled_obs = jax.random.permutation(shuffle_key, obs)

        del obs
        gc.collect()

        print(
            f"Generating imagined rollouts of shape {(training_params['rollout_length'], training_params['num_rollouts'])}"
        )

        (
            imagined_obs,
            imagined_actions,
            imagined_rewards,
            imagined_discounts,
            imagined_values,
            imagined_log_probs,
            total_valid_steps,
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

            n_rollouts = int(args.render)
            n_rollouts = min(n_rollouts, imagined_obs.shape[1])
            if n_rollouts <= 0:
                print("No rollouts selected for rendering (args.render <= 0)")
            else:

                sel_obs = imagined_obs[:, :n_rollouts, :]

                sel_obs = jnp.transpose(sel_obs, (1, 0, 2))

                obs = sel_obs.reshape(
                    sel_obs.shape[0] * sel_obs.shape[1], sel_obs.shape[2]
                )

                sel_actions = imagined_actions[:, :n_rollouts]
                sel_actions = jnp.transpose(sel_actions, (1, 0)).reshape(-1)

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

        def save_model_checkpoints(
            actor_params, critic_params, iteration, prefix=prefix
        ):
            """Save parameters with consistent structure including iteration counter"""
            with open(f"{prefix}actor_params.pkl", "wb") as f:
                pickle.dump({"params": actor_params, "iteration": iteration + 1}, f)
            with open(f"{prefix}critic_params.pkl", "wb") as f:
                pickle.dump({"params": critic_params, "iteration": iteration + 1}, f)

        action_probs = analyze_policy_behavior(
            actor_network, actor_params, imagined_obs
        )

        mean_reward = float(jnp.mean(imagined_rewards))
        try:
            p = np.array(action_probs)
        except Exception:
            p = np.asarray(action_probs)

        entropy_val = -float(np.sum(p * np.log(p + 1e-12)))
        movement_prob = float(p[3] + p[4]) if p.size > 4 else 0.0
        most_likely = int(np.argmax(p))

        log_line = (
            f"iter={i}, mean_reward={mean_reward:.6f}, total_valid_steps={total_valid_steps}, "
            f"action_probs={p.tolist()}, entropy={entropy_val:.6f}, movement_prob={movement_prob:.6f}, "
            f"most_likely={most_likely}\n"
        )

        with open(f"{prefix}training_log", "a") as lf:
            lf.write(log_line)

        save_model_checkpoints(actor_params, critic_params, i, prefix=prefix)

        if i % training_params["retrain_interval"] == 0:

            eval_rewards = evaluate_real_performance(
                actor_network,
                actor_params,
                num_episodes=10,
                render=False,
                reward_predictor_params=reward_predictor_params,
                model_scale_factor=loaded_model_scale_factor,
            )
            eval_mean = float(np.mean(eval_rewards))

            eval_std = float(np.std(eval_rewards))
            with open(f"{prefix}training_log", "a") as lf:
                lf.write(
                    f"eval_mean_reward={eval_mean:.6f}, eval_std_reward={eval_std:.6f}\n"
                )
            if eval_mean >= 19.0:
                print(
                    f"Achieved eval mean reward of {eval_mean:.2f}, stopping training early!"
                )
                break

        if (
            i % training_params["retrain_interval"] == 0
            and rollout_func == generate_imagined_rollouts
        ):
            print(f"\n{'='*60}")
            print(f"RETRAINING WORLDMODEL AFTER {i} TRAINING RUNS")
            print(f"{'='*60}\n")

            actor_type = (
                "imagined" if args.rollout_style == "model" else args.rollout_style
            )

            print(f"Collecting fresh experience with {actor_type} actor...")
            os.system(
                f"python MBRL/worldmodel_mlp.py collect {training_params['wm_sample_size']} {actor_type}"
            )

            print("Retraining worldmodel...")
            os.system(
                f"python MBRL/worldmodel_mlp.py train {training_params['wm_train_epochs']}"
            )

            if os.path.exists("worldmodel_mlp.pkl"):
                with open("worldmodel_mlp.pkl", "rb") as f:
                    saved_data = pickle.load(f)
                    dynamics_params = saved_data.get(
                        "params", saved_data.get("dynamics_params")
                    )
                    if "normalization_stats" in saved_data:
                        normalization_stats = saved_data["normalization_stats"]

                    wm_training_loss = saved_data.get("loss", None)
                    print("Reloaded updated worldmodel!")
                    if wm_training_loss is not None:
                        print(f"World model training loss: {wm_training_loss:.6f}")

            print("Worldmodel retraining complete!")
            print(f"{'='*60}\n")
            with open(f"{prefix}training_log", "a") as lf:
                if wm_training_loss is not None:
                    lf.write(
                        f"-------------------------------------- Retrained Model (loss={wm_training_loss:.6f}) --------------------------------------\n"
                    )
                else:
                    lf.write(
                        "-------------------------------------- Retrained Model --------------------------------------\n"
                    )


if __name__ == "__main__":
    rtpt = RTPT(name_initials="FH", experiment_name="OCActorCritic", max_iterations=3)

    rtpt.start()
    main()
