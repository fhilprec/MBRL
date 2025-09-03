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

from worldmodelPong import compare_real_vs_model
from model_architectures import PongLSTM
from worldmodelPong import get_reward_from_ball_position
from jaxatari.wrappers import AtariWrapper

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
            # DreamerV2 uses ~1M parameters with ELU activations
            hidden_size = 512
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            # Output layer for categorical distribution (as per DreamerV2 paper)
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
            
            return distrax.Categorical(logits=logits)
    
    return DreamerV2Actor(action_dim=action_dim)


def create_dreamerv2_critic():
    """Create DreamerV2 Critic network with ~1M parameters and ELU activations."""
    
    class DreamerV2Critic(nn.Module):
        
        @nn.compact
        def __call__(self, x):
            hidden_size = 512
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            # Single deterministic output (not a distribution)
            # Initialize with smaller values to prevent explosion
            value = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(x)
            
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
    """
    Compute λ-returns exactly as in DreamerV2 paper.
    
    Args:
        rewards: [T, ...] rewards
        values: [T, ...] state values (same length as rewards)
        discounts: [T, ...] discount factors
        bootstrap: [...] bootstrap value for last timestep
        lambda_: λ parameter for λ-returns
        axis: time axis
    
    Returns:
        λ-returns with same shape as rewards
    """
    # Append bootstrap value
    next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=axis)
    
    # Compute targets recursively from the end
    def compute_target(carry, inputs):
        next_lambda_return = carry
        reward, discount, value, next_value = inputs
        
        # V^λ_t = r_t + γ_t * ((1-λ) * V(s_{t+1}) + λ * V^λ_{t+1})
        target = reward + discount * ((1 - lambda_) * next_value + lambda_ * next_lambda_return)
        
        return target, target
    
    # Reverse along time axis and scan
    reversed_inputs = jax.tree.map(
        lambda x: jnp.flip(x, axis=axis),
        (rewards, discounts, values, next_values)
    )
    
    # Initialize with bootstrap value
    _, reversed_returns = lax.scan(
        compute_target,
        bootstrap,
        reversed_inputs
    )
    
    # Flip back to get correct temporal order
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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

            # Sample action from actor
            key, action_key = jax.random.split(key)
            pi = actor_network.apply(actor_params, obs)
            action = pi.sample(seed=action_key)
            log_prob = pi.log_prob(action)
            
            # Get value from critic (deterministic output)
            value = critic_network.apply(critic_params, obs)

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
            next_obs = normalized_next_obs * state_std + state_mean
            next_obs = next_obs.squeeze().astype(obs.dtype)

            reward = get_reward_from_ball_position(next_obs)
            # Apply tanh reward clipping as in DreamerV2
            reward = jnp.tanh(reward * 0.1)

            # Discount factor (constant in imagination)
            discount_factor = jnp.array(discount)

            step_data = (next_obs, reward, discount_factor, action, value, log_prob)
            new_carry = (key, next_obs, new_lstm_state)

            return new_carry, step_data

        # Initialize LSTM state
        dummy_normalized_obs = (cur_obs - state_mean) / state_std
        dummy_action = jnp.zeros(1, dtype=jnp.int32)
        _, initial_lstm_state = world_model.apply(
            dynamics_params, None, dummy_normalized_obs, dummy_action, None
        )

        cur_obs = cur_obs.astype(jnp.float32)
        init_carry = (subkey, cur_obs, initial_lstm_state)

        # Run rollout
        _, trajectory_data = lax.scan(
            rollout_step, init_carry, None, length=rollout_length
        )

        # Unpack trajectory data
        next_obs_seq, rewards_seq, discounts_seq, actions_seq, values_seq, log_probs_seq = trajectory_data

        # Build complete sequences including initial state
        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([0.0]), rewards_seq])  # First reward is 0
        discounts = jnp.concatenate([jnp.array([discount]), discounts_seq])
        
        # Initial action/value/log_prob
        init_action = jnp.zeros_like(actions_seq[0])
        initial_value = critic_network.apply(critic_params, cur_obs)
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([initial_value[None, ...], values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])

        return observations, rewards, discounts, actions, values, log_probs

    # Split keys for each trajectory
    num_trajectories = initial_observations.shape[0]
    keys = jax.random.split(key, num_trajectories)

    # Vectorize over all initial observations
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
    use_reinforce: bool = True,  # For Atari, DreamerV2 uses Reinforce (ρ=1)
    target_update_freq: int = 100,
    max_grad_norm: float = 100.0,
) -> Tuple[Any, Any, Any, Dict]:
    """Train DreamerV2 actor and critic networks following the paper exactly."""
    
    if key is None:
        key = jax.random.PRNGKey(42)

    # Create optimizers with DreamerV2 hyperparameters
    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    
    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Create train states
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

    # Target critic for stability (updated every target_update_freq steps)
    target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)
    update_counter = 0

    # Prepare data: flatten batch dimensions for training
    batch_size = observations.shape[0] * observations.shape[1]
    T, B = rewards.shape[:2]  # Time, Batch
    
    # For λ-return computation, we need to handle the temporal structure
    # We'll compute λ-returns for each trajectory separately, then flatten
    def compute_trajectory_targets(traj_rewards, traj_values, traj_discounts):
        """Compute λ-returns for a single trajectory."""
        # Bootstrap value is the last value prediction
        bootstrap = traj_values[-1]
        # Compute λ-returns for all timesteps except the last
        targets = lambda_return_dreamerv2(
            traj_rewards[:-1], traj_values[:-1], traj_discounts[:-1], 
            bootstrap, lambda_=lambda_, axis=0
        )
        return targets

    # Compute λ-returns for all trajectories
    targets = jax.vmap(compute_trajectory_targets, in_axes=(1, 1, 1), out_axes=1)(
        rewards, values, discounts
    )

    print(f"Lambda returns stats - Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")

    # Flatten for training (exclude last timestep as it's used for bootstrapping)
    observations_flat = observations[:-1].reshape((T-1)*B, -1)
    actions_flat = actions[:-1].reshape((T-1)*B)
    targets_flat = targets.reshape((T-1)*B)
    values_flat = values[:-1].reshape((T-1)*B)
    old_log_probs_flat = log_probs[:-1].reshape((T-1)*B)

    def critic_loss_fn(critic_params, obs, targets):
        """DreamerV2 critic loss with squared error."""
        predicted_values = critic_network.apply(critic_params, obs)
        # Squared loss as in DreamerV2 paper
        loss = jnp.mean((predicted_values - targets) ** 2)
        return loss, {"critic_loss": loss, "critic_mean": jnp.mean(predicted_values)}

    def actor_loss_fn(actor_params, obs, actions, targets, values, old_log_probs):
        """DreamerV2 actor loss with Reinforce and entropy regularization."""
        pi = actor_network.apply(actor_params, obs)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()

        if use_reinforce:
            # Reinforce gradients with baseline (for Atari)
            advantages = targets - values  # Already computed λ-returns
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Reinforce loss: -log π(a|s) * (target - baseline)
            reinforce_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(advantages))
            policy_loss = reinforce_loss
        else:
            # Dynamics backpropagation (for continuous control)
            # Direct optimization of λ-returns
            policy_loss = -jnp.mean(targets)

        # Entropy regularization
        entropy_loss = -entropy_scale * jnp.mean(entropy)

        total_loss = policy_loss + entropy_loss

        return total_loss, {
            "actor_loss": total_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "entropy": jnp.mean(entropy),
            "advantages_mean": jnp.mean(targets - values) if use_reinforce else 0.0,
        }

    # Training loop
    metrics_history = []

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # Shuffle data
        perm = jax.random.permutation(subkey, (T-1)*B)
        obs_shuffled = observations_flat[perm]
        actions_shuffled = actions_flat[perm]
        targets_shuffled = targets_flat[perm]
        values_shuffled = values_flat[perm]
        old_log_probs_shuffled = old_log_probs_flat[perm]

        # Train critic
        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            critic_state.params, obs_shuffled, targets_shuffled
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        # Train actor
        (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            actor_state.params, obs_shuffled, actions_shuffled, targets_shuffled, 
            values_shuffled, old_log_probs_shuffled
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        # Update target critic periodically
        update_counter += 1
        if update_counter % target_update_freq == 0:
            target_critic_params = jax.tree.map(lambda x: x.copy(), critic_state.params)
            print(f"Updated target critic at step {update_counter}")

        # Combine metrics
        epoch_metrics = {**critic_metrics, **actor_metrics}
        metrics_history.append(epoch_metrics)
        
        print(f"Epoch {epoch}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
              f"Entropy: {actor_metrics['entropy']:.4f}")

    # Average final metrics
    final_metrics = jax.tree.map(lambda *x: jnp.mean(jnp.array(x)), *metrics_history)

    return actor_state.params, critic_state.params, target_critic_params, final_metrics






def evaluate_real_performance(actor_network, actor_params, obs_shape, num_episodes=5):
    """Evaluate the trained policy in the real Pong environment."""
    from jaxatari.games.jax_pong import JaxPong
    
    env = JaxPong()
    env = AtariWrapper(
        env,
        sticky_actions=False, 
        episodic_life=False,
        frame_stack_size=4
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
        
        while not done and step_count < 1000:  # Max steps to prevent infinite episodes
            # Convert observation to the right format
            # print(obs)
            obs_tensor, _ = flatten_obs(obs, single_state=True)
            
            
            # Get action from trained actor (deterministic)
            pi = actor_network.apply(actor_params, obs_tensor)
            action = pi.mode()  # Use deterministic action for evaluation
            print(action)
        
            # Step environment
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
    # DreamerV2 hyperparameters
    action_dim = 6
    rollout_length = 50  # DreamerV2 uses 50 for Atari
    num_rollouts = 1600  # Batch size
    policy_epochs = 20    # Few epochs per iteration
    actor_lr = 8e-5      # DreamerV2 learning rates
    critic_lr = 8e-5
    lambda_ = 0.95       # λ-return parameter
    entropy_scale = 1e-3 # Entropy regularization for Atari
    discount = 0.99      # Discount factor

    for i in range(training_runs):

        parser = argparse.ArgumentParser(description="DreamerV2 Pong agent")
        
        
        
        # Create networks
        actor_network = create_dreamerv2_actor(action_dim)
        critic_network = create_dreamerv2_critic()
        

        

        # Load world model
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

        # Initialize or load parameters
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

        # Count parameters
        actor_param_count = sum(x.size for x in jax.tree.leaves(actor_params))
        critic_param_count = sum(x.size for x in jax.tree.leaves(critic_params))
        print(f"Actor parameters: {actor_param_count:,}")
        print(f"Critic parameters: {critic_param_count:,}")

        # Shuffle observations for diverse starting points
        shuffled_obs = jax.random.permutation(jax.random.PRNGKey(SEED), obs)


        evaluate_real_performance(actor_network, actor_params, obs_shape, num_episodes=5)
        exit()


        # Generate imagined rollouts
        print("Generating imagined rollouts...")
        (
            imagined_obs,
            imagined_rewards,
            imagined_discounts,
            imagined_actions, 
            imagined_values,
            imagined_log_probs,
        ) = generate_imagined_rollouts(
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
                    obs=imagined_obs[i+visualization_offset], 
                    actions=imagined_actions[i+visualization_offset], 
                    frame_stack_size=4
                )

        print(f"Reward stats: min={jnp.min(imagined_rewards):.4f}, max={jnp.max(imagined_rewards):.4f}")
        print(f"Non-zero rewards: {jnp.sum(imagined_rewards != 0.0)} / {imagined_rewards.size}")
        print(f"Imagined rollouts shape: {imagined_obs.shape}")

        # Train DreamerV2 actor-critic
        print("Training DreamerV2 actor-critic...")
        actor_params, critic_params, target_critic_params, training_metrics = train_dreamerv2_actor_critic(
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
            use_reinforce=True,  # For Atari
        )

        # Print training progress
        if training_metrics:
            print(f"Final training metrics:")
            for key, value in training_metrics.items():
                print(f"  {key}: {value:.4f}")
            print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")

        # Save model checkpoints
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