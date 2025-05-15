import jax
import jax.numpy as jnp
import optax
import haiku as hk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# Import your world model
from worldmodel import build_world_model, build_reward_model

# Import environment
import sys
sys.path.append('/home/florian/Dropbox/Masterarbeit/JAXAtari')
from src.jaxatari.games.jax_seaquest import JaxSeaquest

def build_actor_critic():
    """Build actor (policy) and critic (value) networks."""
    def actor_fn(obs):
        x = hk.Linear(128)(obs)
        x = jax.nn.relu(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        logits = hk.Linear(18)(x)  # 18 actions in Seaquest
        return logits
    
    def critic_fn(obs):
        x = hk.Linear(128)(obs)
        x = jax.nn.relu(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        value = hk.Linear(1)(x)
        return value
    
    actor = hk.transform(actor_fn)
    critic = hk.transform(critic_fn)
    
    return actor, critic

def load_world_model(path):
    """Load a trained world model and reward model from file."""
    
    with open(path, 'rb') as f:
        saved_data = pickle.load(f)
    
    dynamics_model = build_world_model()
    reward_model = build_reward_model()
    
    # Check which format the saved data is in
    if 'dynamics_params' in saved_data and 'reward_params' in saved_data:
        # New format with both models
        return (dynamics_model, saved_data['dynamics_params']), (reward_model, saved_data['reward_params'])
    elif 'params' in saved_data:
        # Old format with just world model
        print("Found old model format with only dynamics model parameters.")
        print("Initializing reward model from scratch.")
        
        # Initialize reward model with random parameters
        dummy_obs, _ = JaxSeaquest().reset()
        dummy_flat_obs = JaxSeaquest().obs_to_flat_array(dummy_obs)
        dummy_action = jnp.array(0)
        
        reward_rng = jax.random.PRNGKey(42)
        reward_params = reward_model.init(reward_rng, dummy_flat_obs, dummy_action)
        
        return (dynamics_model, saved_data['params']), (reward_model, reward_params)
    else:
        raise KeyError("Saved model file has unexpected format")


def train_actor_critic_with_world_model(game, dynamics_model, dynamics_params, reward_model, reward_params, 
                                        num_epochs=100, batch_size=64, 
                                        rollout_length=10, gamma=0.99,
                                        entropy_coef=0.01):
    """Train actor-critic agent using world model rollouts with learned rewards."""
    # Initialize actor-critic networks
    actor, critic = build_actor_critic()
    rng = jax.random.PRNGKey(42)
    
    # Initialize parameters
    dummy_obs, _ = game.reset()
    dummy_flat_obs = game.obs_to_flat_array(dummy_obs)
    
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    actor_params = actor.init(actor_rng, dummy_flat_obs)
    critic_params = critic.init(critic_rng, dummy_flat_obs)
    
    # Create optimizers
    actor_opt = optax.adam(learning_rate=3e-4)
    critic_opt = optax.adam(learning_rate=1e-3)
    
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    
    # World model rollout function
    def world_model_rollout(rng_key, obs, horizon=rollout_length):
        """Generate a trajectory using the world model with learned rewards."""
        states = [obs]
        rewards = []
        actions_taken = []
        current_obs = obs
        
        for step in range(horizon):
            # Get action probabilities from current policy
            logits = actor.apply(actor_params, None, current_obs)
            
            # Sample action with unique random key for proper exploration
            rng_key, action_key = jax.random.split(rng_key)
            action = jax.random.categorical(action_key, logits)
            actions_taken.append(action)
            
            # Predict next state using world dynamics model
            next_obs = dynamics_model.apply(dynamics_params, None, current_obs, action)
            
            # Predict reward using reward model
            reward = reward_model.apply(reward_params, None, current_obs, action)
            
            states.append(next_obs)
            rewards.append(reward)
            current_obs = next_obs
        
        return states, rewards, actions_taken, rng_key
    
    # Advantage calculation
    def compute_advantages(values, rewards, gamma):
        """Compute advantages from value estimates and rewards."""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = jnp.array(returns)
        advantages = returns - values[:-1]  # values includes next state
        
        return advantages, returns
    
    # Training step functions
    def actor_loss(actor_params, observations, actions, advantages):
        """Actor loss function (policy gradient with entropy regularization)."""
        logits = jax.vmap(lambda o: actor.apply(actor_params, None, o))(observations)
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[:, None], axis=1
        ).squeeze()
        
        # Policy gradient loss
        pg_loss = -jnp.mean(action_log_probs * advantages)
        
        # Entropy regularization
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs, axis=1)
        entropy_loss = -jnp.mean(entropy)  # Negative because we want to maximize entropy
        
        return pg_loss + entropy_coef * entropy_loss
    
    def critic_loss(critic_params, observations, returns):
        """Critic loss function (value estimation)."""
        values = jax.vmap(lambda o: critic.apply(critic_params, None, o))(observations)
        return jnp.mean(jnp.square(values.squeeze() - returns))
    
    # JIT-compile training steps
    @jax.jit
    def train_actor_step(actor_params, actor_opt_state, observations, actions, advantages):
        """Update actor network."""
        loss, grads = jax.value_and_grad(actor_loss)(
            actor_params, observations, actions, advantages
        )
        updates, new_opt_state = actor_opt.update(grads, actor_opt_state)
        new_params = optax.apply_updates(actor_params, updates)
        return new_params, new_opt_state, loss
    
    @jax.jit
    def train_critic_step(critic_params, critic_opt_state, observations, returns):
        """Update critic network."""
        loss, grads = jax.value_and_grad(critic_loss)(
            critic_params, observations, returns
        )
        updates, new_opt_state = critic_opt.update(grads, critic_opt_state)
        new_params = optax.apply_updates(critic_params, updates)
        return new_params, new_opt_state, loss
    
    # Training loop
    actor_losses = []
    critic_losses = []
    rng_key = jax.random.PRNGKey(0)  # Different from model initialization key
    
    for epoch in range(num_epochs):
        # Sample initial state from environment
        obs, state = game.reset()
        flat_obs = game.obs_to_flat_array(obs)
        
        # Generate rollout using world model with proper random key propagation
        rng_key, rollout_key = jax.random.split(rng_key)
        states, rewards, actions, rng_key = world_model_rollout(rollout_key, flat_obs)
        
        # Get value estimates for all states in rollout
        values = jax.vmap(lambda o: critic.apply(critic_params, None, o))(jnp.array(states))
        
        # Compute advantages and returns
        advantages, returns = compute_advantages(values, rewards, gamma)
        
        # Prepare data for training (excluding final state)
        train_obs = jnp.array(states[:-1])
        actions = jnp.array(actions)
        
        # Update actor and critic
        actor_params, actor_opt_state, actor_loss_val = train_actor_step(
            actor_params, actor_opt_state, train_obs, actions, advantages
        )
        
        critic_params, critic_opt_state, critic_loss_val = train_critic_step(
            critic_params, critic_opt_state, train_obs, returns
        )
        
        actor_losses.append(actor_loss_val)
        critic_losses.append(critic_loss_val)
        
        # Log progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}")
            
            # For debugging: print distribution of actions in latest rollout
            action_counts = np.bincount(np.array(actions), minlength=18)
            print(f"Action distribution: {action_counts}")
    
    # Plot training losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(actor_losses)
    plt.title('Actor Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(critic_losses)
    plt.title('Critic Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('actor_critic_training_losses.png')
    plt.show()
    
    return actor_params, critic_params

if __name__ == "__main__":
    # Initialize game
    game = JaxSeaquest()
    
    # Load trained world model and reward model
    (dynamics_model, dynamics_params), (reward_model, reward_params) = load_world_model('world_model.pkl')
    
    # Train actor-critic using world model rollouts
    actor_params, critic_params = train_actor_critic_with_world_model(
        game, 
        dynamics_model, dynamics_params, 
        reward_model, reward_params,
        num_epochs=500,           # More epochs
        rollout_length=40,        # Longer rollouts
        entropy_coef=0.1,        # Higher entropy bonus to promote exploration
        gamma=0.99
    )
    
    # Save trained policy
    with open('actor_critic_model.pkl', 'wb') as f:
        pickle.dump({
            'actor_params': actor_params,
            'critic_params': critic_params
        }, f)
    
    print("Actor-critic training complete!")