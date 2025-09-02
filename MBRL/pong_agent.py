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

SEED = 42


def create_dreamerv2_actor(action_dim: int):
    """Create DreamerV2 Actor network with ~1M parameters and ELU activations."""
    
    class DreamerV2Actor(nn.Module):
        action_dim: int
        
        @nn.compact
        def __call__(self, x):
            # Calculate hidden size to get ~1M parameters
            # For typical latent state size, we use 512 hidden units
            hidden_size = 512
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)  # ELU activation as specified in DreamerV2
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            # Output layer for categorical distribution
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
            
            return distrax.Categorical(logits=logits)
    
    return DreamerV2Actor(action_dim=action_dim)


def create_dreamerv2_critic():
    """Create DreamerV2 Critic network with ~1M parameters and ELU activations."""
    
    class DreamerV2Critic(nn.Module):
        
        @nn.compact
        def __call__(self, x):
            # Calculate hidden size to get ~1M parameters
            hidden_size = 512
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)  # ELU activation as specified in DreamerV2
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            x = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.elu(x)
            
            # Output single value (deterministic)
            value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
            
            return jnp.squeeze(value, axis=-1)
    
    return DreamerV2Critic()


def compute_lambda_returns(rewards: jnp.ndarray, values: jnp.ndarray, gamma: float = 0.99, lambda_: float = 0.95):
    """
    Compute λ-returns as defined in DreamerV2 paper.
    
    V^λ_t = r_t + γ_t * [(1-λ)v(s_{t+1}) + λV^λ_{t+1}] if t < H
            v(s_H) if t = H
    """
    rollout_length, num_rollouts = rewards.shape
    
    lambda_returns = jnp.zeros_like(rewards)
    
    # Initialize with final values
    lambda_returns = lambda_returns.at[-1].set(values[-1])
    
    def compute_return_step(i, lambda_returns):
        t = rollout_length - 2 - i  # Work backwards from second-to-last
        
        # r_t + γ_t * [(1-λ)v(s_{t+1}) + λV^λ_{t+1}]
        # For simplicity, we assume γ_t = γ (constant discount)
        next_value = values[t + 1]
        next_lambda_return = lambda_returns[t + 1]
        
        lambda_return = rewards[t] + gamma * ((1 - lambda_) * next_value + lambda_ * next_lambda_return)
        lambda_returns = lambda_returns.at[t].set(lambda_return)
        
        return lambda_returns
    
    # Compute returns backwards
    lambda_returns = jax.lax.fori_loop(0, rollout_length - 1, compute_return_step, lambda_returns)
    
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
    key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if key is None:
        key = jax.random.PRNGKey(42)

    # Extract normalization stats
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
            # action = 4
            log_prob = pi.log_prob(action)
            
            # Get value from critic
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
            next_obs = jnp.round(normalized_next_obs * state_std + state_mean)
            next_obs = next_obs.squeeze()
            next_obs = next_obs.astype(obs.dtype)

            reward = get_reward_from_ball_position(next_obs)


            step_data = (next_obs, reward, action, value, log_prob)
            new_carry = (key, next_obs, new_lstm_state)

            return new_carry, step_data

        # Initialize LSTM state
        initial_reward = 0.0
        dummy_normalized_obs = (cur_obs - state_mean) / state_std
        dummy_action = jnp.zeros(1, dtype=jnp.int32)
        _, initial_lstm_state = world_model.apply(
            dynamics_params,
            None,
            dummy_normalized_obs,
            dummy_action,
            None,
        )

        cur_obs = cur_obs.astype(jnp.float32)
        init_carry = (subkey, cur_obs, initial_lstm_state)

        # Run rollout
        _, trajectory_data = lax.scan(
            rollout_step, init_carry, None, length=rollout_length
        )

        # Unpack trajectory data
        next_obs_seq, rewards_seq, actions_seq, values_seq, log_probs_seq = trajectory_data

        # Build complete sequences including initial state
        observations = jnp.concatenate([cur_obs[None, ...], next_obs_seq])
        rewards = jnp.concatenate([jnp.array([initial_reward]), rewards_seq])

        # Initial action/value/log_prob are placeholders
        init_action = jnp.zeros_like(actions_seq[0])
        actions = jnp.concatenate([init_action[None, ...], actions_seq])
        values = jnp.concatenate([jnp.array([0.0]), values_seq])
        log_probs = jnp.concatenate([jnp.array([0.0]), log_probs_seq])

        return observations, rewards, actions, values, log_probs

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
    values: jnp.ndarray,
    log_probs: jnp.ndarray,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    key: jax.random.PRNGKey = None,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    rho: float = 1.0,  # 1.0 for Atari (Reinforce), 0.0 for continuous control (dynamics backprop)
    eta: float = 1e-3,  # Entropy regularization
    max_grad_norm: float = 0.5,
    use_target_network: bool = True,
    target_update_freq: int = 100,
) -> Tuple[Any, Any, Dict]:
    """Train DreamerV2 actor and critic networks."""
    
    if key is None:
        key = jax.random.PRNGKey(42)

    # Create optimizers
    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )
    
    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
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
    
    # Target network for critic (optional but recommended)
    target_critic_params = critic_params if use_target_network else None

    # Compute λ-returns
    lambda_returns = compute_lambda_returns(rewards, values, gamma, lambda_)
    
    print(f"Lambda returns stats - Mean: {lambda_returns.mean():.4f}, Std: {lambda_returns.std():.4f}")

    # Flatten batch dimensions
    batch_size = observations.shape[0] * observations.shape[1]
    observations_flat = observations.reshape(batch_size, -1)
    actions_flat = actions.reshape(batch_size)
    lambda_returns_flat = lambda_returns.reshape(batch_size)
    values_flat = values.reshape(batch_size)
    old_log_probs_flat = log_probs.reshape(batch_size)

    def critic_loss_fn(critic_params, obs, targets, target_critic_params=None):
        """Critic loss: squared error between predicted and target values."""
        predicted_values = critic_network.apply(critic_params, obs)
        
        # Use target network if available, otherwise use current network
        if target_critic_params is not None:
            # Target values are computed using target network (not implemented in this step)
            pass
        
        loss = 0.5 * jnp.mean(jnp.square(predicted_values - targets))
        return loss, {"critic_loss": loss}

    def actor_loss_fn(actor_params, obs, actions, targets, values, old_log_probs):
        """
        DreamerV2 Actor loss combining Reinforce and dynamics backpropagation.
        
        L(ψ) = Σ[-ρ ln pψ(at|zt) sg(V^λ_t - v(zt))  (Reinforce)
                -(1-ρ)V^λ_t                          (dynamics backprop)  
                -η H[at|zt]]                         (entropy regularization)
        """
        pi = actor_network.apply(actor_params, obs)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()
        
        # Baseline-subtracted returns for Reinforce
        advantages = targets - values  # V^λ_t - v(z_t)
        
        # Reinforce loss (stop gradients on advantages)
        reinforce_loss = -rho * log_prob * jax.lax.stop_gradient(advantages)
        
        # Dynamics backpropagation loss (straight-through gradients)
        dynamics_loss = -(1 - rho) * targets  # No stop_gradient here for backprop through dynamics
        
        # Entropy regularization
        entropy_loss = -eta * entropy
        
        # Combined loss
        total_loss = jnp.mean(reinforce_loss + dynamics_loss + entropy_loss)
        
        return total_loss, {
            "actor_loss": total_loss,
            "reinforce_loss": jnp.mean(reinforce_loss),
            "dynamics_loss": jnp.mean(dynamics_loss),
            "entropy_loss": jnp.mean(entropy_loss),
            "entropy": jnp.mean(entropy),
        }

    # Training loop
    metrics_history = []
    step_count = 0

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # Shuffle data
        perm = jax.random.permutation(subkey, batch_size)
        obs_shuffled = observations_flat[perm]
        actions_shuffled = actions_flat[perm]
        returns_shuffled = lambda_returns_flat[perm]
        values_shuffled = values_flat[perm]
        old_log_probs_shuffled = old_log_probs_flat[perm]

        # Train critic
        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            critic_state.params, obs_shuffled, returns_shuffled, target_critic_params
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        # Train actor
        (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            actor_state.params, obs_shuffled, actions_shuffled, returns_shuffled, values_shuffled, old_log_probs_shuffled
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        
        # Update target network periodically
        if use_target_network and step_count % target_update_freq == 0:
            target_critic_params = critic_state.params
        
        step_count += 1

        # Combine metrics
        epoch_metrics = {**critic_metrics, **actor_metrics}
        metrics_history.append(epoch_metrics)
        
        print(f"Epoch {epoch}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
              f"Entropy: {actor_metrics['entropy']:.4f}")

    # Average final metrics
    final_metrics = jax.tree.map(lambda *x: jnp.mean(jnp.array(x)), *metrics_history)

    return actor_state.params, critic_state.params, final_metrics





def main():
    iterations = 1
    frame_stack_size = 4

    parser = argparse.ArgumentParser(description="Render a trained DreamerV2 Pong agent")

    # Initialize networks and parameters
    action_dim = 6
    
    # Create DreamerV2 networks
    actor_network = create_dreamerv2_actor(action_dim)
    critic_network = create_dreamerv2_critic()
    
    actor_params = None
    critic_params = None

    # Hyperparameters for policy training
    rollout_length = 15  
    num_rollouts = 20
    policy_epochs = 100
    learning_rate = 1e-4
    
    # DreamerV2 specific hyperparameters
    rho = 1.0  # Use Reinforce for Atari (set to 0.0 for continuous control)
    eta = 1e-3  # Entropy regularization
    lambda_ = 0.95  # λ-return parameter

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


     #if one of the models does not exist
    obs_shape = obs.shape[1:]
    key = jax.random.PRNGKey(42)


    # Create dummy observation with correct dtype
    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)

   

    if os.path.exists("actor_params.pkl"):
        try:
            with open("actor_params.pkl", "rb") as f:
                saved_data = pickle.load(f)
                # Check if it's the old format (direct params) or new format (with "params" key)
                if isinstance(saved_data, dict) and "params" in saved_data:
                    actor_params = saved_data["params"]
                else:
                    actor_params = saved_data  # Old format
                print("Loaded existing actor parameters")
        except Exception as e:
            print(f"Error loading actor params: {e}. Reinitializing...")
            key, subkey = jax.random.split(key)
            actor_params = actor_network.init(subkey, dummy_obs)
    else:
        key, subkey = jax.random.split(key)
        actor_params = actor_network.init(subkey, dummy_obs)
        print("Initialized new actor parameters")

    # Initialize or load critic parameters
    if os.path.exists("critic_params.pkl"):
        try:
            with open("critic_params.pkl", "rb") as f:
                saved_data = pickle.load(f)
                # Check if it's the old format (direct params) or new format (with "params" key)
                if isinstance(saved_data, dict) and "params" in saved_data:
                    critic_params = saved_data["params"]
                else:
                    critic_params = saved_data  # Old format
                print("Loaded existing critic parameters")
        except Exception as e:
            print(f"Error loading critic params: {e}. Reinitializing...")
            key, subkey = jax.random.split(key)
            critic_params = critic_network.init(subkey, dummy_obs)
    else:
        key, subkey = jax.random.split(key)
        critic_params = critic_network.init(subkey, dummy_obs)
        print("Initialized new critic parameters")

    # Verify parameters are properly initialized
    if actor_params is None:
        raise ValueError("Actor params failed to initialize")
    if critic_params is None:
        raise ValueError("Critic params failed to initialize")

 
    
    # Count parameters
    actor_param_count = sum(x.size for x in jax.tree.leaves(actor_params))
    critic_param_count = sum(x.size for x in jax.tree.leaves(critic_params))
    print(f"Actor parameters: {actor_param_count:,}")
    print(f"Critic parameters: {critic_param_count:,}")


    #shuffle obs for diverse starting points
    shuffled_obs = jax.random.permutation(jax.random.PRNGKey(SEED), obs)


    # Generate imagined rollouts
    print("Generating imagined rollouts...")
    (
        imagined_obs,
        imagined_rewards,
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
        key=jax.random.PRNGKey(SEED),
    )

    parser.add_argument(
        "--render",
        type=int,
        help="Output path for the recorded video",
    )
    args = parser.parse_args()
    if args.render:

        visualization_offset = 50
        for i in range(int(args.render)):
            compare_real_vs_model(steps_into_future=0, obs=imagined_obs[i+visualization_offset], actions = imagined_actions, frame_stack_size=4)
        




    print(f"Reward stats: min={jnp.min(imagined_rewards):.4f}, max={jnp.max(imagined_rewards):.4f}")
    print(f"Non-zero rewards: {jnp.sum(imagined_rewards != 0.0)} / {imagined_rewards.size}")
    print(f"Imagined rollouts shape: {imagined_obs.shape}")
 

    # Train DreamerV2 actor-critic
    print("Training DreamerV2 actor-critic...")
    actor_params, critic_params, training_metrics = train_dreamerv2_actor_critic(
        actor_params=actor_params,
        critic_params=critic_params,
        actor_network=actor_network,
        critic_network=critic_network,
        observations=imagined_obs,
        actions=imagined_actions,
        rewards=imagined_rewards,
        values=imagined_values,
        log_probs=imagined_log_probs,
        num_epochs=policy_epochs,
        learning_rate=learning_rate,
        key=jax.random.PRNGKey(2000),
        lambda_=lambda_,
        rho=rho,
        eta=eta,
    )

    # Print training progress
    if training_metrics:
        print(f"Final training metrics:")
        for key, value in training_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")



    # At the end of main(), replace the save section with:
    def save_model_checkpoints(actor_params, critic_params):
        """Save parameters with consistent structure"""
        with open("actor_params.pkl", "wb") as f:
            pickle.dump({"params": actor_params}, f)
        with open("critic_params.pkl", "wb") as f:
            pickle.dump({"params": critic_params}, f)
        print("Saved actor and critic parameters")

    # Call the save function
    save_model_checkpoints(actor_params, critic_params)


if __name__ == "__main__":
    # Create RTPT object
    rtpt = RTPT(
        name_initials="FH", experiment_name="DreamerV2Agent", max_iterations=3
    )

    # Start the RTPT tracking
    rtpt.start()
    main()