import os
# Add CUDA paths to environment
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
# Remove CPU restriction to allow GPU usage
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Comment this out

import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from jaxatari.games.jax_seaquest import SeaquestRenderer, JaxSeaquest
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper


VERBOSE = True



# Global variable to hold model for evaluate_model function
model = None

def build_world_model():
    def forward(state, action):
        # Flatten the state tree structure to a 1D vector
        flat_state = hk.Flatten()(jax.flatten_util.ravel_pytree(state)[0])
        
        # Convert action to one-hot
        action_one_hot = jax.nn.one_hot(action, num_classes=18)
        
        # Determine if we have a batch dimension
        is_batched = len(action_one_hot.shape) > 1
        
        # Concatenate along the appropriate axis
        if is_batched:
            inputs = jnp.concatenate([flat_state, action_one_hot], axis=1)
        else:
            inputs = jnp.concatenate([flat_state, action_one_hot], axis=0)
        
        # Feed through MLP
        x = hk.Linear(512)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        
        # Final output layer
        # For simplicity, predict the flattened state vector directly
        # The state size will be determined by the flattened input
        output_size = flat_state.shape[-1]
        flat_next_state = hk.Linear(output_size)(x)
        
        return flat_next_state
    
    return hk.transform(forward)






#this function does not differentiate between done and not done environments
#maybe this is not a problem since the model should learn to predict the next state regardless of whether the environment is done or not
def collect_experience(game: JaxSeaquest, num_episodes: int = 100, 
                       max_steps_per_episode: int = 1000, num_envs: int = 512) -> Tuple[List, List, List]:

    print(f"Collecting experience data from {num_envs} parallel environments...")
    
    
    
    # Create vectorized reset and step functions stolen from ppo
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
        jax.random.split(rng, n_envs)
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step
    )(jax.random.split(rng, n_envs), env_state, action)
    
    # Initialize storage for collected data
    states = []
    next_states = []
    actions = []
    rewards = []
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)  # Use a fixed seed for reproducibility
    
    # JIT compile the reset and step functions
    jitted_reset = jax.jit(vmap_reset(num_envs))
    jitted_step = jax.jit(vmap_step(num_envs))
    
    # Reset all environments
    rng, reset_rng = jax.random.split(rng) #returns two random keys
    # Reset all environments in parallel
    _, state = jitted_reset(reset_rng)
    

    
    total_steps = 0
    total_episodes = 0
    

    while total_episodes < num_episodes * num_envs:
        # Store the current state
        current_state_repr = jax.tree.map(lambda x: x, state.env_state.env_state)
        
        # Generate random actions for all environments
        rng, action_rng = jax.random.split(rng)
        action_batch = jax.random.randint(action_rng, (num_envs,), 0, 18)
        
        # Step all environments
        rng, step_rng = jax.random.split(rng)
        _, next_state, reward_batch, done_batch, _ = jitted_step(step_rng, state, action_batch)
        

        
        if jnp.any(done_batch):
            # Reset environments that are done
            rng, reset_rng = jax.random.split(rng)
            _, reset_states = jitted_reset(reset_rng)
            
            # Create a function to update only the done states
            def update_where_done(old_state, new_state, done_mask):
                """Update states only where done_mask is True."""
                def where_with_correct_broadcasting(x, y, mask):
                    # Handle broadcasting for different array dimensions
                    if hasattr(x, 'shape') and hasattr(y, 'shape'):
                        if x.ndim > 1:
                            # Create mask with right shape for broadcasting
                            new_shape = (mask.shape[0],) + (1,) * (x.ndim - 1)
                            reshaped_mask = mask.reshape(new_shape)
                            return jnp.where(reshaped_mask, y, x)
                        else:
                            return jnp.where(mask, y, x)
                    else:
                        # For non-array elements
                        return x  # Keep original for simplicity
                
                return jax.tree.map(
                    lambda x, y: where_with_correct_broadcasting(x, y, done_mask),
                    old_state, new_state
                )
            
            # Update only the states that are done
            next_state = update_where_done(next_state, reset_states, done_batch)
        




        # Extract state representation from the new state
        next_state_repr = jax.tree.map(lambda x: x, next_state.env_state.env_state) #not sure here whether there are multiple states or not
        

        # Store experience
        states.append(current_state_repr)
        actions.append(action_batch)
        next_states.append(next_state_repr)
        rewards.append(reward_batch)
        
        # Count completed episodes from done signals
        newly_completed = jnp.sum(done_batch)
        total_episodes += newly_completed
        total_steps += num_envs
        
        
        # Just update state for next iteration, ignoring done environments (inefficient but simple)
        state = next_state

        # Break if we've collected enough episodes
        if total_episodes >= num_episodes * num_envs:
            break
    

    if VERBOSE:
        print(f"Experience collection completed:")
        print(f"- Total steps: {total_steps}")
        print(f"- Total episodes: {total_episodes}")
        print(f"- Total transitions: {len(states)}")
    

    # Convert lists to arrays
    states = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *states)
    actions = jnp.concatenate(actions, axis=0)
    next_states = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *next_states)
    rewards = jnp.concatenate(rewards, axis=0)

    if VERBOSE:
        print(f"Final flattened shape: states: {jax.tree.map(lambda x: x.shape, states)}")
        print(f"Actions shape: {actions.shape}, Rewards shape: {rewards.shape}")

    return states, actions, next_states, rewards



def train_world_model(
    states, 
    actions, 
    next_states, 
    rewards,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    num_epochs: int = 100,
    validation_split: float = 0.1,
    grad_clip_norm: float = 1.0,
    verbose: bool = True
) -> Tuple[hk.Params, Dict[str, Any]]:
    pass

    print(states.shape)
    print()



    return 1,2
    
    






if __name__ == "__main__":
    # Initialize the game environment
    game = JaxSeaquest()
    env = AtariWrapper(game, sticky_actions=False, episodic_life=False)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    
    # Train the world model
    save_path = "world_model.pkl"

    # Set the global model variable
    model = build_world_model()




    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data['dynamics_params']
    else:
        print("No existing model found. Training a new model...")

        # Define a file path for the experience data
        experience_data_path = "experience_data.pkl"

        # Check if experience data file exists
        if os.path.exists(experience_data_path):
            print(f"Loading existing experience data from {experience_data_path}...")
            with open(experience_data_path, 'rb') as f:
                saved_data = pickle.load(f)
                states = saved_data['states']
                actions = saved_data['actions']
                next_states = saved_data['next_states']
                rewards = saved_data['rewards']
        else:
            print("No existing experience data found. Collecting new experience data...")
            # Collect experience data
            states, actions, next_states, rewards = collect_experience(
                game,
                num_episodes=1,
                max_steps_per_episode=10000,
                num_envs=512
            )
            
            # Save the collected experience data
            with open(experience_data_path, 'wb') as f:
                pickle.dump({
                    'states': states,
                    'actions': actions,
                    'next_states': next_states,
                    'rewards': rewards
                }, f)
            print(f"Experience data saved to {experience_data_path}")

        #train world model
        dynamics_params, training_info = train_world_model(
            states,
            actions,
            next_states,
            rewards,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=100,
            validation_split=0.1,
            grad_clip_norm=1.0,
            verbose=True
        )

    # # Evaluate the model
    # eval_mse = evaluate_model(dynamics_params, env, model)
    # print(f"Final evaluation MSE: {eval_mse:.6f}")


    #  # Then visualize the predictions
    # print("Visualizing model predictions vs. actual gameplay...")
    # stats = visualize_predictions(dynamics_params, env, model, num_steps=500, delay=0.05)
    # print(f"Visualization stats: {stats}")