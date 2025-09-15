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

            # reward = improved_pong_reward(next_obs, action, frame_stack_size=4)
            

            old_score =  obs[-5]-obs[-1]
            new_score =  next_obs[-5]-next_obs[-1]

            reward = new_score - old_score
            reward = jnp.where(jnp.abs(reward) > 1, 0.0, reward)


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
        print(
            f"Action 3 (LEFT): {action_counts[3]}, Action 4 (RIGHTFIRE): {action_counts[4]}"
        )

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
    targets_mean = targets.mean()
    targets_std = targets.std()
    targets_normalized = (targets - targets_mean) / (targets_std + 1e-8)

    print(
        f"Normalized targets - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    )

    print(
        f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    )
    print(
        f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}"
    )
    print(
        f"Lambda returns stats - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    )
    print(
        f"Lambda returns stats - Mean: {targets_normalized.mean():.4f}, Std: {targets_normalized.std():.4f}"
    )

    observations_flat = observations[:-1].reshape((T - 1) * B, -1)
    actions_flat = actions[:-1].reshape((T - 1) * B)
    targets_flat = targets_normalized.reshape((T - 1) * B)
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
    ):
        pi = actor_network.apply(actor_params, obs)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()

        advantages = targets - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        reinforce_obj = log_prob * jax.lax.stop_gradient(advantages)

        objective =  -reinforce_obj 

        entropy_bonus = entropy_scale * entropy
        total_objective = objective + entropy_bonus

        actor_loss = jnp.mean(total_objective)

        return actor_loss, {
            "actor_loss": actor_loss,
            "objective": jnp.mean(objective),
            "entropy": jnp.mean(entropy),
            "advantages_mean": jnp.mean(advantages),
            "mix_ratio": mix_ratio,
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

        # print(
        #     f"Epoch {epoch}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
        #     f"Entropy: {actor_metrics['entropy']:.4f}"
        # )

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
        "rollout_length": 60,
        "num_rollouts": 1600,
        "policy_epochs": 50,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "lambda_": 0.95,
        "entropy_scale": 1e-1,
        "discount": 0.95,
        "max_grad_norm": 10.0,
        "target_kl": 1,
        "early_stopping_patience": 25,
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

        shuffled_obs = jax.random.permutation(jax.random.PRNGKey(SEED), obs)

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
            key=jax.random.PRNGKey(SEED),
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
    rtpt = RTPT(name_initials="FH", experiment_name="DreamerV2Agent", max_iterations=3)
    rtpt.start()
    main()
