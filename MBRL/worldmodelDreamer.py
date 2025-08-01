import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Must be set before importing JAX

import pygame
import time
import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle
from jaxatari.games.jax_seaquest import SeaquestRenderer, JaxSeaquest
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper
from jax import lax
import gc

from obs_state_converter import flat_observation_to_state, OBSERVATION_INDEX_MAP

from model_architectures import *



def get_reward_from_observation(obs):
    if len(obs) != 180:
        raise ValueError(f"Observation must have 180 elements, got {len(obs)}")
    return obs[177]



import jax
import jax.numpy as jnp
import haiku as hk
from typing import Any, Dict, Tuple, NamedTuple

class RSSMState(NamedTuple):
    """State of the RSSM."""
    stoch: jnp.ndarray  # Stochastic state component
    deter: jnp.ndarray  # Deterministic state component
    mean: jnp.ndarray   # Mean of stochastic state prior/posterior
    std: jnp.ndarray    # Std deviation of stochastic state prior/posterior

class RSSM(hk.Module):
    """Recurrent State Space Model from Dreamer."""
    
    def __init__(self, 
                 stoch_size=30, 
                 deter_size=200, 
                 hidden_size=200,
                 min_std=0.1,
                 name="rssm"):
        super().__init__(name=name)
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.min_std = min_std
        
    def initial_state(self, batch_size=1):
        """Return initial RSSM state."""
        return RSSMState(
            stoch=jnp.zeros((batch_size, self.stoch_size)),
            deter=jnp.zeros((batch_size, self.deter_size)),
            mean=jnp.zeros((batch_size, self.stoch_size)),
            std=jnp.ones((batch_size, self.stoch_size)),
        )
    
    def __call__(self, obs, action, prev_state=None, training=True):
        """Forward pass through the RSSM."""
        # Ensure inputs have correct batch dimensions
        if obs.ndim == 1:
            obs = obs[None, :]  # Add batch dimension
        
        if action.ndim == 1:
            action = action[None, :]  # Add batch dimension
            
        batch_size = obs.shape[0]
        
        if prev_state is None:
            prev_state = self.initial_state(batch_size)
        
        # Ensure action has the same batch size as prev_state.stoch
        if action.shape[0] != prev_state.stoch.shape[0]:
            action = jnp.repeat(action, prev_state.stoch.shape[0], axis=0)

        # Concatenate stochastic state and action
        x = jnp.concatenate([prev_state.stoch, action], axis=-1)
        print(f"Step 1 - After concatenation: x type = {type(x)}, x shape = {x.shape}")
        
        # Apply first linear transformation
        x = hk.Linear(self.hidden_size)(x)
        print(f"Step 2 - After first hk.Linear: x type = {type(x)}, x shape = {x.shape}")
        
        # Apply activation
        x = jax.nn.elu(x)
        print(f"Step 3 - After jax.nn.elu: x type = {type(x)}, x shape = {x.shape}")
        
        # Apply second linear transformation
        x = hk.Linear(self.hidden_size)(x)
        print(f"Step 4 - After second hk.Linear: x type = {type(x)}, x shape = {x.shape}")
        
        # Apply activation
        x = jax.nn.elu(x)
        print(f"Step 5 - After second jax.nn.elu: x type = {type(x)}, x shape = {x.shape}")
        
        # GRU update
        gru = hk.GRU(self.deter_size)
        deter, new_state = gru(x, prev_state.deter)  # Unpack the GRU output
        print(f"Step 6 - After GRU update: deter type = {type(deter)}, deter shape = {deter.shape}")
        
        # Get prior distribution parameters
        x = deter
        x = hk.Linear(self.hidden_size)(x)
        print(f"Step 7 - After third hk.Linear: x type = {type(x)}, x shape = {x.shape}")
        
        x = jax.nn.elu(x)
        print(f"Step 8 - After third jax.nn.elu: x type = {type(x)}, x shape = {x.shape}")
        
        mean = hk.Linear(self.stoch_size)(x)
        print(f"Step 9 - After mean hk.Linear: mean type = {type(mean)}, mean shape = {mean.shape}")
        
        std = hk.Linear(self.stoch_size)(x)
        print(f"Step 10 - After std hk.Linear: std type = {type(std)}, std shape = {std.shape}")
        
        std = jax.nn.softplus(std) + self.min_std
        print(f"Step 11 - After jax.nn.softplus: std type = {type(std)}, std shape = {std.shape}")
        
        # Sample from prior during training, or use mean during eval
        if training:
            key = hk.next_rng_key()
            stoch = mean + std * jax.random.normal(key, mean.shape)
        else:
            stoch = mean
        print(f"Step 12 - After sampling: stoch type = {type(stoch)}, stoch shape = {stoch.shape}")
        
        prior = RSSMState(stoch=stoch, deter=deter, mean=mean, std=std)
        
        # Posterior model: q(z_t | h_t, o_t)
        # Ensure obs has the same batch size as deter
        if obs.shape[0] != deter.shape[0]:
            obs = jnp.repeat(obs, deter.shape[0], axis=0)

        # Concatenate deterministic state and observation
        x = jnp.concatenate([deter, obs], axis=-1)
        print(f"Step 13 - After concatenating deter and obs: x type = {type(x)}, x shape = {x.shape}")
        x = hk.Linear(self.hidden_size)(x)
        print(f"Step 14 - After fourth hk.Linear: x type = {type(x)}, x shape = {x.shape}")
        
        x = jax.nn.elu(x)
        print(f"Step 15 - After fourth jax.nn.elu: x type = {type(x)}, x shape = {x.shape}")
        
        x = hk.Linear(self.hidden_size)(x)
        print(f"Step 16 - After fifth hk.Linear: x type = {type(x)}, x shape = {x.shape}")
        
        x = jax.nn.elu(x)
        print(f"Step 17 - After fifth jax.nn.elu: x type = {type(x)}, x shape = {x.shape}")
        
        mean = hk.Linear(self.stoch_size)(x)
        print(f"Step 18 - After posterior mean hk.Linear: mean type = {type(mean)}, mean shape = {mean.shape}")
        
        std = hk.Linear(self.stoch_size)(x)
        print(f"Step 19 - After posterior std hk.Linear: std type = {type(std)}, std shape = {std.shape}")
        
        std = jax.nn.softplus(std) + self.min_std
        print(f"Step 20 - After posterior jax.nn.softplus: std type = {type(std)}, std shape = {std.shape}")
        
        if training:
            key = hk.next_rng_key()
            stoch = mean + std * jax.random.normal(key, mean.shape)
        else:
            stoch = mean
        print(f"Step 21 - After posterior sampling: stoch type = {type(stoch)}, stoch shape = {stoch.shape}")
        
        posterior = RSSMState(stoch=stoch, deter=deter, mean=mean, std=std)
        
        return prior, posterior


# Define the DreamerWorldModel as a Haiku module
class DreamerWorldModel(hk.Module):
    """World Model using RSSM architecture from Dreamer."""
    
    def __init__(self, 
                obs_size=160,
                action_size=18,
                stoch_size=30, 
                deter_size=200, 
                hidden_size=200,
                min_std=0.1,
                name="dreamer_world_model"):
        super().__init__(name=name)
        self.obs_size = obs_size
        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.min_std = min_std
        
    def __call__(self, obs, action, rssm_state=None, training=True):
        """Forward pass through the world model."""
        # Print shapes for debugging
        print(f"DreamerWorldModel input - obs.shape: {obs.shape}, action.shape: {action.shape}")
        print(f"DreamerWorldModel rssm_state: {rssm_state}")
        
        # Ensure inputs have proper batch dimensions
        if obs.ndim == 1:
            obs = obs[None, :]  # Add batch dimension
        if action.ndim == 1:
            action = action[None, :]  # Add batch dimension
            
        # Create RSSM component
        rssm = RSSM(self.stoch_size, self.deter_size, self.hidden_size, self.min_std)
        
        # Process through RSSM
        prior, posterior = rssm(obs, action, rssm_state, training)
        
        # Feature representation combines stochastic and deterministic parts
        features = jnp.concatenate([posterior.stoch, posterior.deter], axis=-1)
        
        # Observation model: p(o_t | z_t, h_t)
        x = features
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.elu(x)
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.elu(x)
        next_obs = hk.Linear(self.obs_size)(x)
        
        # Reward prediction model
        x = features
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.elu(x)
        reward_pred = hk.Linear(1)(x)
        
        return next_obs, posterior, prior, reward_pred


VERBOSE = True
model = None

action_map = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}





def flatten_obs(
    state, single_state: bool = False, is_list=False
) -> Tuple[jnp.ndarray, Any]:
    """
    Flatten the state PyTree into a single array.
    This is useful for debugging and visualization.
    """
    # check whether it is a single state or a batch of states

    if type(state) == list:
        flat_states = []

        for s in state:
            flat_state, _ = jax.flatten_util.ravel_pytree(s)
            flat_states.append(flat_state)
        flat_states = jnp.stack(flat_states, axis=0)  # Shape: (1626, 160)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener
    batch_shape = state.player_x.shape[0]

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)
    return flat_state, unflattener


def collect_experience_sequential(
    env,
    num_episodes: int = 1,
    max_steps_per_episode: int = 1000,
    episodic_life: bool = False,
    seed: int = 42,
    policy_params=None,
    network=None,
):
    """Collect experience data sequentially to ensure proper transitions."""
    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    boundaries = []

    dead = False
    total_steps = 0
    rng = jax.random.PRNGKey(seed)

    # policies--------------------
    # OPTION 1: Biased Movement Policy (encourages more movement)
    def biased_movement_policy(rng):
        """Bias towards movement actions to explore more of the screen"""
        rng, action_key = jax.random.split(rng)

        # 30% chance for movement actions (2-9: UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT)
        # 20% chance for fire actions (10-17)
        # 10% chance for NOOP/FIRE (0,1)

        action_prob = jax.random.uniform(action_key)

        if action_prob < 0.3:  # Movement actions
            action = jax.random.randint(action_key, (), 2, 10)  # UP through DOWNLEFT
        elif action_prob < 0.5:  # Fire actions
            action = jax.random.randint(
                action_key, (), 10, 18
            )  # UPFIRE through DOWNLEFTFIRE
        else:  # NOOP or basic FIRE
            action = jax.random.randint(action_key, (), 0, 2)

        return action

    # OPTION 2: Directional Sweep Policy (systematic exploration)
    def directional_sweep_policy(rng, step_count):
        """Sweep left and right periodically to explore horizontally"""
        rng, action_key = jax.random.split(rng)

        # Every 50 steps, do a directional sweep
        sweep_cycle = step_count % 100

        if sweep_cycle < 25:  # Move left for 25 steps
            if jax.random.uniform(action_key) < 0.6:
                action = 4  # LEFT
            else:
                action = jax.random.randint(action_key, (), 0, 18)  # Random
        elif sweep_cycle < 50:  # Move right for 25 steps
            if jax.random.uniform(action_key) < 0.6:
                action = 3  # RIGHT
            else:
                action = jax.random.randint(action_key, (), 0, 18)  # Random
        else:  # Random for remaining 50 steps
            action = jax.random.randint(action_key, (), 0, 18)

        return action

    # OPTION 3: Vertical + Horizontal Bias (my recommendation)
    def exploration_bias_policy(rng):
        """Simple policy that encourages both horizontal and vertical movement"""
        rng, action_key = jax.random.split(rng)

        action_type = jax.random.uniform(action_key)

        if action_type < 0.25:  # 25% - Horizontal movement
            rng, move_key = jax.random.split(rng)
            action = jax.random.choice(
                move_key, jnp.array([3, 4, 6, 7, 8, 9])
            )  # RIGHT, LEFT, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT
        elif action_type < 0.4:  # 15% - Vertical movement
            rng, move_key = jax.random.split(rng)
            action = jax.random.choice(move_key, jnp.array([2, 5]))  # UP, DOWN
        elif action_type < 0.65:  # 25% - Fire while moving
            rng, fire_key = jax.random.split(rng)
            action = jax.random.randint(fire_key, (), 10, 18)  # All fire actions
        else:  # 35% - Completely random
            rng, rand_key = jax.random.split(rng)
            action = jax.random.randint(rand_key, (), 0, 18)

        return action

    # OPTION 4: Simple Edge Explorer (forces ship to edges)
    def edge_explorer_policy(rng, step_count):
        """Periodically forces ship to screen edges"""
        rng, action_key = jax.random.split(rng)

        # Every 80 steps, spend 20 steps going to an edge
        cycle = step_count % 80

        if cycle < 20:  # Go to left edge
            if jax.random.uniform(action_key) < 0.7:
                action = jax.random.choice(
                    action_key, jnp.array([4, 7, 9])
                )  # LEFT, UPLEFT, DOWNLEFT
            else:
                action = jax.random.randint(action_key, (), 0, 18)
        elif cycle < 40:  # Go to right edge
            if jax.random.uniform(action_key) < 0.7:
                action = jax.random.choice(
                    action_key, jnp.array([3, 6, 8])
                )  # RIGHT, UPRIGHT, DOWNRIGHT
            else:
                action = jax.random.randint(action_key, (), 0, 18)
        else:  # Random exploration
            action = jax.random.randint(action_key, (), 0, 18)

        return action

    # OPTION 5: down_biased_policy
    def down_biased_policy(rng):
        """Simple policy that encourages both horizontal and vertical movement"""
        rng, action_key = jax.random.split(rng)
        action = (
            5
            if jax.random.uniform(action_key) < 0.2
            else jax.random.randint(action_key, (), 0, 18)
        )
        return action

    for episode in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, state = env.reset(reset_key)

        for step in range(max_steps_per_episode):
            current_state = state
            current_obs = obs

            # rng action key
            rng, action_key = jax.random.split(rng)

            # Choose a random action
            if network and policy_params:
                # Use policy to select action
                flat_obs, _ = flatten_obs(obs, single_state=True)
                pi, _ = network.apply(policy_params, flat_obs)
                action = pi.sample(seed=action_key)
            else:
                action = down_biased_policy(rng)

            # Take a step in the environment
            rng, step_key = jax.random.split(rng)
            next_obs, next_state, reward, done, _ = env.step(state, action)

            # Store the transition
            observations.append(current_obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            dones.append(done)

            # If episode is done, reset the environment

            if not episodic_life:
                if current_state.env_state.death_counter > 0 and not dead:
                    dead = True
                if not current_state.env_state.death_counter > 0 and dead:
                    dead = False
                    boundaries.append(total_steps)

            if done:
                print(f"Episode {episode+1} done after {step+1} steps")

                if episodic_life:
                    if len(boundaries) == 0:
                        boundaries.append(step)
                    else:
                        boundaries.append(boundaries[-1] + step + 1)
                break

            # Update state for the next step
            state = next_state
            obs = next_obs
            total_steps += 1

    # Convert to JAX arrays (but don't flatten the structure yet)
    # Use tree_map to maintain structure with jnp arrays

    # Stack states correctly to form batch
    # Step 1: Stack states across time
    # batched_observations = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *observations)

    actions_array = jnp.array(actions)
    rewards_array = jnp.array(rewards)
    dones_array = jnp.array(dones)

    # print("Boundaries:")
    # print(boundaries)

    return (
        flatten_obs(observations, is_list=True),
        actions_array,
        rewards_array,
        dones_array,
        boundaries,
    )


def train_world_model(
    obs,
    actions,
    next_obs,
    rewards,
    learning_rate=2e-4,
    batch_size=4,
    num_epochs=100,
    sequence_length=32,
    episode_boundaries=None,
    frame_stack_size=4,
):
    """Train a JAX-based Dreamer-style world model on structured observations."""
    
    gpu_batch_size = 125
    gpu_batch_size = gpu_batch_size // frame_stack_size
    
    # Calculate normalization statistics from the flattened obs
    state_mean = jnp.mean(obs, axis=0)
    state_std = jnp.std(obs, axis=0) + 1e-8

    # Store normalization stats for later use
    normalization_stats = {"mean": state_mean, "std": state_std}

    # Normalize obs and next_obs
    normalized_obs = (obs - state_mean) / state_std
    normalized_next_obs = (next_obs - state_mean) / state_std
    
    # Create sequential batches that respect episode boundaries
    def create_sequential_batches(batch_size=32):
        """
        Create batches of sequential data for training
        Args:
            batch_size: Number of sequences per batch
        Returns:
            List of batches, each containing (state_batch, action_batch, next_state_batch)
            where each has shape (batch_size, seq_len, feature_dim)
        """
        sequences = []

        # First, collect all sequences
        for i in range(len(episode_boundaries) - 1):
            if i == 0:
                start_idx = 0
                end_idx = episode_boundaries[0]
            else:
                start_idx = episode_boundaries[i - 1]
                end_idx = episode_boundaries[i]

            # Create sequences within this episode with stride for better coverage
            for j in range(
                0, end_idx - start_idx - sequence_length + 1, sequence_length // 4
            ):
                if start_idx + j + sequence_length > end_idx:
                    # Padding strategy for sequences that exceed episode boundary
                    padding_length = start_idx + j + sequence_length - end_idx
                    padded_obs = jnp.concatenate(
                        [
                            normalized_obs[start_idx + j : end_idx],
                            jnp.tile(normalized_obs[end_idx - 1], (padding_length, 1)),
                        ],
                        axis=0,
                    )
                    padded_actions = jnp.concatenate(
                        [
                            actions[start_idx + j : end_idx],
                            jnp.tile(actions[end_idx - 1], (padding_length,)),
                        ],
                        axis=0,
                    )
                    padded_next_obs = jnp.concatenate(
                        [
                            normalized_next_obs[start_idx + j : end_idx],
                            jnp.tile(
                                normalized_next_obs[end_idx - 1], (padding_length, 1)
                            ),
                        ],
                        axis=0,
                    )

                    sequences.append((padded_obs, padded_actions, padded_next_obs))
                    continue

                sequences.append(
                    (
                        normalized_obs[start_idx + j : start_idx + j + sequence_length],
                        actions[start_idx + j : start_idx + j + sequence_length],
                        normalized_next_obs[
                            start_idx + j : start_idx + j + sequence_length
                        ],
                    )
                )

        return sequences

    # Create sequential batches
    batches = create_sequential_batches()
    print(f"Created {len(batches)} sequential batches of size {sequence_length}")

    # Split data into training (80%) and validation (20%)
    total_batches = len(batches)
    train_size = int(0.8 * total_batches)

    # Shuffle batches before splitting to ensure random distribution
    rng_split = jax.random.PRNGKey(42)
    indices = jax.random.permutation(rng_split, total_batches)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_batches = [batches[i] for i in train_indices]
    val_batches = [batches[i] for i in val_indices]

    print(
        f"Training batches: {len(train_batches)}, Validation batches: {len(val_batches)}"
    )
    
    
    
    # Define the forward function for the world model
    def forward_fn(obs, action, state=None, training=True):
        """Forward function with proper handling of action dimensions."""
        
        print(f"forward_fn input - obs type: {type(obs)}, action type: {type(action)}")
        print(f"forward_fn input - obs.shape: {obs.shape}, action.shape: {action.shape}")
        
        # Handle batch dimensions properly
        if obs.ndim == 2:  # Already has batch dimension
            batch_obs = obs
        else:
            batch_obs = obs[None, :]  # Add batch dimension
            
        if action.ndim == 2:  # Already has batch dimension
            batch_action = action
        elif action.ndim == 0:  # Single scalar action
            # Convert scalar action to one-hot encoding with batch dimension
            batch_action = jax.nn.one_hot(jnp.array([action]), num_classes=18)
        elif action.ndim == 1:
            if action.shape[0] == 18:  # Already one-hot encoded
                batch_action = action[None, :]  # Add batch dimension
            else:  # Single action index
                batch_action = jax.nn.one_hot(action, num_classes=18)[None, :]  # One-hot and add batch dim
        else:
            batch_action = action
        
        # Ensure batch dimensions match
        if batch_action.shape[0] != batch_obs.shape[0]:
            batch_action = jnp.repeat(batch_action, batch_obs.shape[0], axis=0)
        
        print(f"forward_fn processed - batch_obs.shape: {batch_obs.shape}, batch_action.shape: {batch_action.shape}")
        
        model = DreamerWorldModel(
            obs_size=batch_obs.shape[-1],
            action_size=18  # Hardcoded since we're using one-hot actions
        )
        
        return model(batch_obs, batch_action, state, training)
    
    # Transform the forward function
    forward = hk.transform(forward_fn)
    
    # Initialize parameters with a dummy batch
    rng = jax.random.PRNGKey(42)

    # Create properly shaped dummy inputs
    dummy_obs = normalized_obs[0:1]  # Shape: (1, obs_size)
    dummy_action = jnp.zeros((1, 18))  # Create a dummy one-hot action

    print(f"Dummy obs shape: {dummy_obs.shape}, Dummy action shape: {dummy_action.shape}")

    # Initialize parameters with simplified forward pass to avoid complex tensor operations
    try:
        params = forward.init(rng, dummy_obs, dummy_action)
    except Exception as e:
        print(f"Error during initialization: {e}")
        # Fallback initialization with different inputs
        rng, new_key = jax.random.split(rng)
        dummy_obs_simple = jnp.zeros((1, normalized_obs.shape[1]))  # Simple zeros array
        dummy_action_simple = jnp.zeros((1, 18))  # Simple zeros array
        params = forward.init(new_key, dummy_obs_simple, dummy_action_simple)
    
    # Define optimizer with learning rate scheduling
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=num_epochs,
        alpha=0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8),
    )
    opt_state = optimizer.init(params)
    
    # Define KL divergence between two Gaussian distributions
    def kl_divergence(prior, posterior):
        """KL divergence between Gaussian distributions."""
        return jnp.sum(
            jnp.log(prior.std) - jnp.log(posterior.std) + 
            (posterior.std**2 + (posterior.mean - prior.mean)**2) / 
            (2 * prior.std**2) - 0.5
        )
    
    # Define single-sequence loss function
    @jax.jit
    def sequence_loss(params, seq_obs, seq_actions, seq_next_obs, seq_rewards, key, beta=0.1):
        """Compute loss for a sequence of transitions."""
        seq_len = seq_obs.shape[0]
        
        def step_fn(carry, inputs):
            rssm_state, key = carry
            obs, action, target_obs, target_reward = inputs
            
            key, subkey = jax.random.split(key)
            
            # Process action to ensure it has the right shape
            if action.ndim == 0:  # Scalar action
                action = jax.nn.one_hot(jnp.array([action]), num_classes=18)[0]
            
            # Ensure obs has correct shape
            if obs.ndim == 0:
                obs = obs.reshape(1)  # Reshape to 1D array
            
            # Forward pass through model - properly handle state
            pred_obs, posterior, prior, pred_reward = forward.apply(
                params, subkey, jnp.expand_dims(obs, 0), jnp.expand_dims(action, 0), rssm_state, True
            )
            
            # Remove batch dimension
            if pred_reward.shape[-1] == 1:
                pred_reward = pred_reward.squeeze(-1)
            
            # Compute losses
            reconstruction_loss = jnp.mean((pred_obs - target_obs) ** 2)
            reward_loss = jnp.mean((pred_reward - target_reward) ** 2) if target_reward is not None else 0.0
            kl_loss = kl_divergence(prior, posterior)
            
            # Total loss with weighting
            total_loss = reconstruction_loss + beta * kl_loss + 0.1 * reward_loss
            
            return (posterior, key), (total_loss, reconstruction_loss, kl_loss, reward_loss)
        
        # Initialize the carry with an initial RSSM state and the key
        batch_size = seq_obs.shape[1]  # Assuming seq_obs has shape (seq_len, batch_size, feature_dim)
        initial_rssm_state = RSSMState(
            stoch=jnp.zeros((batch_size, 30)),  # Replace 30 with your `stoch_size`
            deter=jnp.zeros((batch_size, 200)),  # Replace 200 with your `deter_size`
            mean=jnp.zeros((batch_size, 30)),  # Replace 30 with your `stoch_size`
            std=jnp.ones((batch_size, 30))  # Replace 30 with your `stoch_size`
        )
        init_carry = (initial_rssm_state, key)
        
        # Set up scan inputs
        inputs = (seq_obs, seq_actions, seq_next_obs, seq_rewards)
        
        # Run scan through sequence
        (_, _), (losses, rec_losses, kl_losses, reward_losses) = jax.lax.scan(
            step_fn, init_carry, inputs
        )
        
        # Return mean losses
        mean_loss = jnp.mean(losses)
        mean_rec_loss = jnp.mean(rec_losses)
        mean_kl_loss = jnp.mean(kl_losses)
        mean_reward_loss = jnp.mean(reward_losses)
        
        return mean_loss, (mean_rec_loss, mean_kl_loss, mean_reward_loss)
    
    # Vectorize loss function for batch processing
    batched_sequence_loss = jax.vmap(
        sequence_loss, in_axes=(None, 0, 0, 0, 0, None, None)
    )
    
    # Define update step function
    @jax.jit
    def update_step(params, opt_state, batch_obs, batch_actions, batch_next_obs, batch_rewards, key, beta):
        """Update model parameters using a batch of sequences."""
        
        def loss_fn(p):
            total_losses, component_losses = batched_sequence_loss(
                p, batch_obs, batch_actions, batch_next_obs, batch_rewards, key, beta
            )
            return jnp.mean(total_losses), component_losses
        
        (loss, component_losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, component_losses
    
    @jax.jit
    def compute_validation_loss(params, batch_obs, batch_actions, batch_next_obs, batch_rewards, key, beta):
        """Compute loss on validation data."""
        total_losses, component_losses = batched_sequence_loss(
            params, batch_obs, batch_actions, batch_next_obs, batch_rewards, key, beta
        )
        return jnp.mean(total_losses), component_losses
    
    # Convert training and validation batches to arrays
    train_batch_states = jnp.stack([batch[0] for batch in train_batches])
    train_batch_actions = jnp.stack([batch[1] for batch in train_batches])
    train_batch_next_states = jnp.stack([batch[2] for batch in train_batches])
    # Create reward batches (zeros if rewards are not used)
    train_batch_rewards = jnp.zeros((train_batch_states.shape[0], train_batch_states.shape[1], 1))
    
    val_batch_states = jnp.stack([batch[0] for batch in val_batches])
    val_batch_actions = jnp.stack([batch[1] for batch in val_batches])
    val_batch_next_states = jnp.stack([batch[2] for batch in val_batches])
    # Create reward batches (zeros if rewards are not used)
    val_batch_rewards = jnp.zeros((val_batch_states.shape[0], val_batch_states.shape[1], 1))
    
    # Training loop with validation tracking
    best_loss = float("inf")
    patience = 50
    no_improve_count = 0
    rng_train = jax.random.PRNGKey(123)
    
    # KL divergence weight annealing schedule
    kl_beta = lambda epoch: min(1.0, 0.1 + epoch / (num_epochs * 0.5))
    
    for epoch in range(num_epochs):
        # Shuffle data each epoch
        rng_train, shuffle_key = jax.random.split(rng_train)
        indices = jax.random.permutation(shuffle_key, len(train_batches))

        shuffled_train_states = train_batch_states[indices]
        shuffled_train_actions = train_batch_actions[indices]
        shuffled_train_next_states = train_batch_next_states[indices]

        # Extract rewards from observations using the provided function
        shuffled_train_rewards = jnp.array([
            get_reward_from_observation(obs) for obs in shuffled_train_states.reshape(-1, shuffled_train_states.shape[-1])
        ]).reshape(shuffled_train_states.shape[0], shuffled_train_states.shape[1], 1)

        # Only use gpu_batch_size elements for training to avoid OOM
        if shuffled_train_states.shape[0] > gpu_batch_size:
            shuffled_train_states = shuffled_train_states[:gpu_batch_size]
            shuffled_train_actions = shuffled_train_actions[:gpu_batch_size]
            shuffled_train_next_states = shuffled_train_next_states[:gpu_batch_size]
            shuffled_train_rewards = shuffled_train_rewards[:gpu_batch_size]
        
        # Current beta for KL divergence weighting
        current_beta = kl_beta(epoch)
        
        # Update model parameters
        rng_train, update_key = jax.random.split(rng_train)
        params, opt_state, train_loss, (rec_loss, kl_loss, reward_loss) = update_step(
            params,
            opt_state,
            shuffled_train_states,
            shuffled_train_actions,
            shuffled_train_next_states,
            shuffled_train_rewards,
            update_key,
            current_beta
        )
        
        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Compute validation loss occasionally
        if VERBOSE and (epoch + 1) % 10 == 0:
            rng_train, val_key = jax.random.split(rng_train)
            val_loss, (val_rec_loss, val_kl_loss, val_reward_loss) = compute_validation_loss(
                params,
                val_batch_states,
                val_batch_actions,
                val_batch_next_states,
                val_batch_rewards,
                val_key,
                current_beta
            )
            
            current_lr = lr_schedule(epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {jnp.mean(train_loss).item():.4f} (Rec: {jnp.mean(rec_loss).item():.4f}, KL: {jnp.mean(kl_loss).item():.4f}), "
                f"Val Loss: {jnp.mean(val_loss).item():.4f} (Rec: {jnp.mean(val_rec_loss).item():.4f}, KL: {jnp.mean(val_kl_loss).item():.4f}), "
                f"LR: {current_lr:.2e}, Beta: {current_beta:.2f}"
            )
        elif VERBOSE and (epoch + 1) % 1 == 0:
            current_lr = lr_schedule(epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {jnp.mean(train_loss).item():.4f} (Rec: {jnp.mean(rec_loss).item():.4f}, KL: {jnp.mean(kl_loss).item():.4f}), "
                f"LR: {current_lr:.2e}, Beta: {current_beta:.2f}"
            )
    
    print("Training completed")
    return params, {
        "final_loss": train_loss,
        "normalization_stats": normalization_stats,
        "best_loss": best_loss,
        "forward_fn": forward_fn,  # Store the function for later use
    }








def compare_real_vs_model(
    num_steps: int = 150,
    render_scale: int = 2,
    obs=None,
    actions=None,
    normalization_stats=None,
    steps_into_future: int = 20,
    clock_speed=10,
    boundaries=None,
    env=None,
    starting_step: int = 0,
    render_debugging: bool = False,
    frame_stack_size: int = 4,
):
    """Compare real environment with Dreamer world model predictions."""
    if len(obs) == 1:
        obs = obs.squeeze(0)

    # Load the saved model
    with open("world_model.pkl", "rb") as f:
        saved_data = pickle.load(f)
        dynamics_params = saved_data["dynamics_params"]
        normalization_stats = saved_data.get("normalization_stats", None)
    
    # Define or load forward function
    def forward_fn(obs, action, state=None, training=True):
        # Handle observation dimensions
        if obs.ndim == 1:
            obs = obs[None, :]  # Add batch dimension
        
        # Handle action dimensions
        if isinstance(action, int) or (action.ndim == 0):  # Scalar action
            action = jax.nn.one_hot(jnp.array([action]), num_classes=18)
        elif action.ndim == 1 and action.shape[0] == 1:  # Single action in a batch
            action = jax.nn.one_hot(action, num_classes=18)
        elif action.ndim == 1 and action.shape[0] != 18:  # Multiple actions without batch dim
            action = action[None, :]
        
        # Ensure batch dimensions match
        if action.shape[0] != obs.shape[0]:
            action = jnp.repeat(action, obs.shape[0], axis=0)
        
        model = DreamerWorldModel(
            obs_size=obs.shape[-1],
            action_size=18  # Hardcoded since we're using one-hot actions
        )
        
        return model(obs, action, state, training)
    
    # Transform the forward function
    forward = hk.transform(forward_fn)

    # Setup rendering
    renderer = SeaquestRenderer()
    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs Dreamer World Model")

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    # Get unflattener for rendering
    game = JaxSeaquest()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    dummy_obs, _ = env.reset(jax.random.PRNGKey(int(time.time())))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    # Initialize observations and state
    real_obs = obs[0]
    model_obs = obs[0]
    rssm_state = None  # Initial RSSM state
    step_count = 0 + starting_step
    clock = pygame.time.Clock()
    
    # Main loop
    while step_count < min(num_steps, len(obs) - 1):
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Get action from saved actions
        action = actions[step_count]
        
        # Get next real observation from data
        next_real_obs = obs[step_count + 1]

        # Check if we need to reset the model state
        if steps_into_future > 0 and (step_count % steps_into_future == 0 or step_count in boundaries):
            model_obs = obs[step_count]  # Reset to current real observation
            rssm_state = None  # Reset state
            print(f"State reset at step {step_count}")

        # Use model to predict next observation
        model_prediction, rssm_state = dreamer_predict_next_state(
            dynamics_params, 
            forward,
            model_obs, 
            action, 
            rssm_state, 
            normalization_stats
        )
        
        # Update model observation
        model_obs = model_prediction
        
        # Convert observations to state representations for rendering
        real_base_state = flat_observation_to_state(
            real_obs, unflattener, frame_stack_size=frame_stack_size
        )
        model_base_state = flat_observation_to_state(
            model_obs, unflattener, frame_stack_size=frame_stack_size
        )
        
        # Render both observations
        real_raster = renderer.render(real_base_state)
        real_img = np.array(real_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(real_surface, real_img)
        
        model_raster = renderer.render(model_base_state)
        model_img = np.array(model_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(model_surface, model_img)
        
        # Display on screen
        screen.fill((0, 0, 0))
        scaled_real = pygame.transform.scale(
            real_surface, (WIDTH * render_scale, HEIGHT * render_scale)
        )
        screen.blit(scaled_real, (0, 0))
        scaled_model = pygame.transform.scale(
            model_surface, (WIDTH * render_scale, HEIGHT * render_scale)
        )
        screen.blit(scaled_model, (WIDTH * render_scale + 20, 0))
        
        # Add labels
        font = pygame.font.SysFont(None, 24)
        real_text = font.render("Real Environment", True, (255, 255, 255))
        model_text = font.render("Dreamer World Model", True, (255, 255, 255))
        step_text = font.render(f"Step: {step_count} Action: {action_map[action]}", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        screen.blit(step_text, (20, HEIGHT * render_scale - 30))
        
        pygame.display.flip()
        
        # Update for next iteration
        real_obs = next_real_obs
        step_count += 1
        clock.tick(clock_speed)

    pygame.quit()
    print("Comparison completed")


def dreamer_predict_next_state(params, forward, current_obs, action, state=None, normalization_stats=None):
    """Predict next state using the Dreamer world model."""
    # Normalize observation
    if normalization_stats:
        state_mean = normalization_stats["mean"]
        state_std = normalization_stats["std"]
        normalized_obs = (current_obs - state_mean) / state_std
    else:
        normalized_obs = current_obs
    
    # Process action to one-hot if it's a scalar
    if isinstance(action, (int, np.int32, np.int64)) or (action.ndim == 0):
        action = jax.nn.one_hot(jnp.array([action]), num_classes=18)[0]
    
    # Forward pass through the model
    next_obs, posterior, _, _ = forward.apply(
        params, 
        jax.random.PRNGKey(0), 
        normalized_obs[None] if normalized_obs.ndim == 1 else normalized_obs, 
        action[None] if action.ndim == 1 else action,
        state,
        training=False
    )
    
    # Denormalize prediction
    if normalization_stats:
        next_obs = next_obs * state_std + state_mean
    
    return next_obs.squeeze(0), posterior





def main():

    frame_stack_size = 1

    game = JaxSeaquest()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    env = FlattenObservationWrapper(env)

    save_path = f"world_model.pkl"
    experience_data_path = "experience_data_LSTM.pkl"
    normalization_stats = None

    # print(next_states[300][:-2])
    # pred = model.apply(dynamics_params, None, states[300], actions[300])
    # print(pred)

    # print(((next_states[300][:-2] - pred) ** 2))

    experience_its = 2

    if not os.path.exists("experience_data_LSTM_0.pkl"):
        print("No existing experience data found. Collecting new experience data...")
        # Collect experience data (AtariWrapper handles frame stacking automatically)

        for i in range(0, experience_its):
            print(f"Collecting experience data (iteration {i+1}/{experience_its})...")
            obs, actions, rewards, _, boundaries = collect_experience_sequential(
                env, num_episodes=2, max_steps_per_episode=10000, seed=i
            )
            next_obs = obs[1:]
            obs = obs[:-1]

            experience_path = "experience_data_LSTM" + "_" + str(i) + ".pkl"

            with open(experience_path, "wb") as f:
                pickle.dump(
                    {
                        "obs": obs,
                        "actions": actions,
                        "next_obs": next_obs,
                        "rewards": rewards,
                        "boundaries": boundaries,
                    },
                    f,
                )
            print(f"Experience data saved to {experience_path}")

            # Explicitly delete large variables to free memory
            del obs, actions, rewards, boundaries, next_obs
            gc.collect()  # Force garbage collection

    # load all experience data into memory
    obs = []
    actions = []
    next_obs = []
    rewards = []
    boundaries = []

    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, "rb") as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data["dynamics_params"]
            normalization_stats = saved_data.get("normalization_stats", None)
    else:
        print("No existing model found. Training a new model...")

        # Define a file path for the experience data

        # Check if experience data file exists
        for i in range(0, experience_its - 1):  # reserve last for training
            experience_path = "experience_data_LSTM" + "_" + str(i) + ".pkl"
            with open(experience_path, "rb") as f:
                saved_data = pickle.load(f)
                obs.extend(saved_data["obs"])
                actions.extend(saved_data["actions"])
                next_obs.extend(saved_data["next_obs"])
                rewards.extend(saved_data["rewards"])
                # Calculate the offset from previous data
                offset = boundaries[-1] if boundaries else 0
                # Add offset to each boundary before extending
                adjusted_boundaries = [b + offset for b in saved_data["boundaries"]]
                boundaries.extend(adjusted_boundaries)

        obs_array = jnp.array(obs)
        actions_array = jnp.array(actions)
        next_obs_array = jnp.array(next_obs)
        rewards_array = jnp.array(rewards)

        # Train world model with improved hyperparameters
        dynamics_params, training_info = train_world_model(
            obs_array,
            actions_array,
            next_obs_array,
            rewards_array,
            episode_boundaries=boundaries,
            frame_stack_size=frame_stack_size,
        )
        normalization_stats = training_info.get("normalization_stats", None)

        # Save the model and scaling factor
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "dynamics_params": dynamics_params,
                    "normalization_stats": training_info.get(
                        "normalization_stats", None
                    ),
                },
                f,
            )
        print(f"Model saved to {save_path}")

    gc.collect()

    with open(f"experience_data_LSTM_{0}.pkl", "rb") as f:
        saved_data = pickle.load(f)
        obs = saved_data["obs"]
        actions = saved_data["actions"]
        next_obs = saved_data["next_obs"]
        rewards = saved_data["rewards"]
        boundaries = saved_data["boundaries"]

    if len(args := sys.argv) > 2 and args[2] == "render":
        compare_real_vs_model(
            num_steps=1000,
            render_scale=6,
            obs=obs,
            actions=actions,
            normalization_stats=normalization_stats,
            boundaries=boundaries,
            env=env,
            starting_step=0,
            steps_into_future=10,
            render_debugging=(args[3] == "verbose" if len(args) > 3 else False),
            frame_stack_size=frame_stack_size,
        )




if __name__ == "__main__":
    from rtpt import RTPT
    rtpt = RTPT(
        name_initials="FH", experiment_name="World model Dreamer", max_iterations=1000
    )
    rtpt.start()
    main()
