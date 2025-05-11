import jax
import jax.numpy as jnp
import optax
import haiku as hk
import pickle
from typing import Tuple, Dict, List

# Import your world model
from worldmodel import build_world_model

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
    """Load a trained world model from file."""
    with open(path, 'rb') as f:
        saved_data = pickle.load(f)
    
    model = build_world_model()
    return model, saved_data['params']

def train_actor_critic_with_world_model(game, world_model, world_model_params, 
                                        num_epochs=100, batch_size=64, 
                                        rollout_length=5, gamma=0.99):
    """Train actor-critic agent using world model rollouts."""
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
    def world_model_rollout(params, obs, horizon=rollout_length):
        """Generate a trajectory using the world model."""
        states = [obs]
        rewards = []
        current_obs = obs
        
        for _ in range(horizon):
            # Get action probabilities from current policy
            logits = actor.apply(actor_params, None, current_obs)
            probs = jax.nn.softmax(logits)
            
            # Sample action
            action = jax.random.categorical(jax.random.PRNGKey(42), logits)
            
            # Predict next state using world model
            next_obs = world_model.apply(world_model_params, None, current_obs, action)
            
            # Simple reward function (can be learned separately if needed)
            reward = jnp.sum(jnp.abs(next_obs - current_obs))  # Proxy reward
            
            states.append(next_obs)
            rewards.append(reward)
            current_obs = next_obs
        
        return states, rewards
    
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
        """Actor loss function (policy gradient)."""
        logits = jax.vmap(lambda o: actor.apply(actor_params, None, o))(observations)
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[:, None], axis=1
        ).squeeze()
        
        return -jnp.mean(action_log_probs * advantages)
    
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
    for epoch in range(num_epochs):
        # Sample initial state from environment
        obs, state = game.reset()
        flat_obs = game.obs_to_flat_array(obs)
        
        # Generate rollout using world model
        states, rewards = world_model_rollout(world_model_params, flat_obs)
        
        # Get value estimates for all states in rollout
        values = jax.vmap(lambda o: critic.apply(critic_params, None, o))(jnp.array(states))
        
        # Compute advantages and returns
        advantages, returns = compute_advantages(values, rewards, gamma)
        
        # Prepare data for training (excluding final state)
        train_obs = jnp.array(states[:-1])
        
        # Sample actions for training data
        logits = jax.vmap(lambda o: actor.apply(actor_params, None, o))(train_obs)
        actions = jax.random.categorical(jax.random.PRNGKey(epoch), logits)
        
        # Update actor and critic
        actor_params, actor_opt_state, actor_loss_val = train_actor_step(
            actor_params, actor_opt_state, train_obs, actions, advantages
        )
        
        critic_params, critic_opt_state, critic_loss_val = train_critic_step(
            critic_params, critic_opt_state, train_obs, returns
        )
        
        # Log progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Actor Loss: {actor_loss_val}, Critic Loss: {critic_loss_val}")
    
    return actor_params, critic_params

# Main execution
if __name__ == "__main__":
    # Initialize game
    game = JaxSeaquest()
    
    # Load trained world model
    world_model, world_model_params = load_world_model('world_model.pkl')
    
    # Train actor-critic using world model rollouts
    actor_params, critic_params = train_actor_critic_with_world_model(
        game, world_model, world_model_params
    )
    
    # Save trained policy
    with open('actor_critic_model.pkl', 'wb') as f:
        pickle.dump({
            'actor_params': actor_params,
            'critic_params': critic_params
        }, f)
    
    print("Actor-critic training complete!")