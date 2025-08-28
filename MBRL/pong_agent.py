import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Must be set before importing JAX
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


from worldmodelPong import compare_real_vs_model
from model_architectures import PongLSTM
from worldmodelPong import get_reward_from_ball_position


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
    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]
    world_model = PongLSTM(4)

    def single_trajectory_rollout(cur_obs, subkey):
        """Generate a single trajectory starting from cur_obs."""

        print(cur_obs.shape)

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

            # Denormalize and ensure consistent dtype
            next_obs = jnp.round(normalized_next_obs * state_std + state_mean)
            next_obs = next_obs.squeeze()
            # Ensure next_obs has the same dtype as input obs
            next_obs = next_obs.astype(obs.dtype)

            # Get reward
            reward = get_reward_from_ball_position(next_obs)

            step_data = (next_obs, reward, action, value, log_prob)
            new_carry = (key, next_obs, new_lstm_state)

            return new_carry, step_data

        # Initialize LSTM state properly to maintain consistent pytree structure
        initial_reward = 0.0  # Make sure this is float

        # Get initial LSTM state structure by doing a dummy forward pass
        dummy_normalized_obs = (cur_obs - state_mean) / state_std
        dummy_action = jnp.zeros(1, dtype=jnp.int32)  # Ensure correct dtype
        _, initial_lstm_state = world_model.apply(
            dynamics_params,
            None,
            dummy_normalized_obs,
            dummy_action,
            None,
        )

        # Ensure cur_obs has the right dtype (float32)
        cur_obs = cur_obs.astype(jnp.float32)

        # Use the properly structured LSTM state
        init_carry = (subkey, cur_obs, initial_lstm_state)

        # Run rollout
        _, trajectory_data = lax.scan(
            rollout_step, init_carry, None, length=rollout_length
        )

        # Unpack trajectory data
        next_obs_seq, rewards_seq, actions_seq, values_seq, log_probs_seq = (
            trajectory_data
        )

        print(rewards_seq)

        # Build complete sequences including initial state
        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([initial_reward]), rewards_seq])

        # Initial action/value/log_prob are placeholders with correct dtypes
        init_action = jnp.zeros_like(actions_seq[0])
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([jnp.array([0.0]), values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])
        print(rewards)
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
    learning_rate: float = 1e-4,  # Reduced learning rate
    key: jax.random.PRNGKey = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    num_minibatches: int = 4,
) -> Tuple[Any, Dict]:
    """Train the actor-critic network on imagined rollouts."""
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
        """Calculate Generalized Advantage Estimation correctly."""
        # rewards and values shape: [rollout_length, num_rollouts]
        rollout_length, num_rollouts = rewards.shape

        # Initialize advantages
        advantages = jnp.zeros_like(rewards)

        # Start from the last timestep and work backwards
        last_gae = jnp.zeros(num_rollouts)

        def gae_step(i, carry):
            advantages, last_gae = carry

            # Current timestep values
            reward = rewards[i]
            value = values[i]

            # Next value (0 for last timestep, assuming episode ends)
            next_value = jnp.where(
                i == rollout_length - 1, jnp.zeros(num_rollouts), values[i + 1]
            )

            # TD error
            delta = reward + gamma * next_value - value

            # GAE
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages = advantages.at[i].set(last_gae)

            return (advantages, last_gae)

        # Iterate backwards through time
        advantages, _ = jax.lax.fori_loop(
            0,
            rollout_length,
            lambda i, carry: gae_step(rollout_length - 1 - i, carry),
            (advantages, last_gae),
        )

        return advantages

    advantages = calculate_gae(rewards, values, gamma, gae_lambda)
    targets = advantages + values

    # Add debugging information
    print(
        f"Rewards stats - Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}, Min: {rewards.min():.4f}, Max: {rewards.max():.4f}"
    )
    print(
        f"Values stats - Mean: {values.mean():.4f}, Std: {values.std():.4f}, Min: {values.min():.4f}, Max: {values.max():.4f}"
    )
    print(
        f"Targets stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}, Min: {targets.min():.4f}, Max: {targets.max():.4f}"
    )
    print(
        f"Advantages stats - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}"
    )

    # Flatten batch dimensions FIRST, then normalize
    batch_size = observations.shape[0] * observations.shape[1]
    observations_flat = observations.reshape(batch_size, -1)
    actions_flat = actions.reshape(batch_size)
    advantages_flat = advantages.reshape(batch_size)
    targets_flat = targets.reshape(batch_size)  # Flatten before normalization
    old_log_probs_flat = log_probs.reshape(batch_size)
    old_values_flat = values.reshape(batch_size)

    # Normalize advantages
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (
        advantages_flat.std() + 1e-8
    )

    # Clip targets to prevent extreme values
    targets_clipped = jnp.clip(targets_flat, -10.0, 10.0)

    def loss_fn(params, obs, actions, advantages, targets, old_log_probs, old_values):
        """Compute PPO loss."""
        # Forward pass
        pi, value = network.apply(params, obs)
        log_prob = pi.log_prob(actions)

        # Ensure shapes match for value loss calculation

        # Value loss with clipped predictions
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
            "policy_loss": loss_actor,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
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
        targets_shuffled = targets_clipped[perm]  # Use clipped targets
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
                batch_old_values,
            )

            train_state = train_state.apply_gradients(grads=grads)
            epoch_metrics.append(metrics)

        # Average metrics for this epoch
        epoch_avg_metrics = jax.tree.map(
            lambda *x: jnp.mean(jnp.array(x)), *epoch_metrics
        )
        metrics_history.append(epoch_avg_metrics)
        print(epoch_avg_metrics)

    # Average final metrics
    final_metrics = jax.tree.map(lambda *x: jnp.mean(jnp.array(x)), *metrics_history)

    return train_state.params, final_metrics


def main():

    iterations = 1
    frame_stack_size = 1

    # Initialize policy parameters and network
    policy_params = None
    network = None
    action_dim = 18

    # Hyperparameters for policy training
    rollout_length = 30  # Length of imagined rollouts
    num_rollouts = 20  # Number of rollouts per iteration
    policy_epochs = 20  # Number of policy training epochs
    learning_rate = 3e-4

    if os.path.exists("world_model_PongLSTM_pong.pkl"):
        with open("world_model_PongLSTM_pong.pkl", "rb") as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data["dynamics_params"]
            normalization_stats = saved_data.get("normalization_stats", None)

        print(f"Loading existing model from experience_data_LSTM_pong_0.pkl...")
        with open("experience_data_LSTM_pong_0.pkl", "rb") as f:
            saved_data = pickle.load(f)
            obs = saved_data["obs"]
            actions = saved_data["actions"]
            next_obs = saved_data["next_obs"]
            rewards = saved_data["rewards"]
    else:
        print("Train Model first")
        exit()

    # Train policy using imagined rollouts exclusively from the world model
    print("Training policy with imagined rollouts...")

    # Create or update actor-critic network
    if network is None:
        obs_shape = obs.shape[1:]  # Get observation shape
        network = create_actor_critic_network(obs_shape, action_dim)

        # Initialize policy parameters if first iteration
        key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros((1,) + obs_shape)
        policy_params = network.init(key, dummy_obs)

    # Generate imagined rollouts using the world model
    print("Generating imagined rollouts...")
    (
        imagined_obs,
        imagined_actions,
        imagined_rewards,
        imagined_values,
        imagined_log_probs,
    ) = generate_imagined_rollouts(
        dynamics_params=dynamics_params,
        policy_params=policy_params,
        network=network,
        initial_observations=obs[
            :num_rollouts
        ],  # Use real observations as starting points
        rollout_length=rollout_length,
        normalization_stats=normalization_stats,
        key=jax.random.PRNGKey(1000),
    )
    print("imagined ops shape")
    print(imagined_obs.shape)

    # only take one rollout for comparison

    # env = JaxPong()
    # for i in range(10):
    #     single_imagined_obs = imagined_obs[0 + i : 1 + i]
    #     single_imagined_actions = imagined_actions[0 + i : 1 + i]

    #     compare_real_vs_model(
    #         num_steps=1000,
    #         render_scale=6,
    #         obs=single_imagined_obs,
    #         actions=single_imagined_actions,
    #         normalization_stats=normalization_stats,
    #         boundaries=None,
    #         env=env,
    #         starting_step=0,
    #         steps_into_future=0,
    #         render_debugging=True,
    #         frame_stack_size=1,
    #         model_path="world_model_PongLSTM_pong.pkl",
    #     )

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
        key=jax.random.PRNGKey(2000),
    )

    # Print training progress
    if training_metrics:
        print(f"Policy loss: {training_metrics.get('policy_loss', 'N/A'):.4f}")
        print(f"Value loss: {training_metrics.get('value_loss', 'N/A'):.4f}")
        print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")


if __name__ == "__main__":
    # Create RTPT object
    rtpt = RTPT(
        name_initials="FH", experiment_name="TestingIterateAgent", max_iterations=3
    )

    # Start the RTPT tracking
    rtpt.start()
    main()
