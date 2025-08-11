import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
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
from rtpt import RTPT
from obs_state_converter import flat_observation_to_state, OBSERVATION_INDEX_MAP

from model_architectures import *


def get_reward_from_observation(obs):
    if len(obs) != 180:
        raise ValueError(f"Observation must have 180 elements, got {len(obs)}")
    return obs[177]


# get the model architecture from the command line argument
if len(sys.argv) > 1:
    model_architecture_name = sys.argv[1]
    if model_architecture_name == "V2_NO_SEP":
        MODEL_ARCHITECTURE = V2_NO_SEP
    elif model_architecture_name == "V2":
        MODEL_ARCHITECTURE = V2_LSTM
    elif model_architecture_name == "MLP":
        MODEL_ARCHITECTURE = MLP
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture_name}")
else:
    # Default model if no argument provided
    MODEL_ARCHITECTURE = V2_LSTM


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


def render_trajectory(
    states, num_frames: int = 100, render_scale: int = 3, delay: int = 50
):
    """
    Render a trajectory of states in a single window.
    Args:
        states: PyTree containing the collected states to visualize
        num_frames: Maximum number of frames to show
        render_scale: Scaling factor for rendering
        delay: Milliseconds to delay between frames
    """
    import pygame
    import time

    pygame.init()
    renderer = SeaquestRenderer()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("State Trajectory Visualization")
    surface = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont(None, 24)
    if isinstance(states, dict) or hasattr(states, "env_state"):
        total_frames = 1
    else:
        first_field = jax.tree_util.tree_leaves(states)[0]
        total_frames = first_field.shape[0] if hasattr(first_field, "shape") else 1
    frames_to_show = min(total_frames, num_frames)
    print(f"Rendering trajectory with {frames_to_show} frames...")
    running = True
    frame_idx = 0
    while running and frame_idx < frames_to_show:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if total_frames > 1:
            current_state = jax.tree.map(
                lambda x: (
                    x[frame_idx]
                    if hasattr(x, "shape") and x.shape[0] > frame_idx
                    else x
                ),
                states,
            )
        else:
            current_state = states
        try:
            raster = renderer.render(current_state)
            img = np.array(raster * 255, dtype=np.uint8)
            pygame.surfarray.blit_array(surface, img)
            screen.fill((0, 0, 0))
            scaled_surface = pygame.transform.scale(
                surface, (WIDTH * render_scale, HEIGHT * render_scale)
            )
            screen.blit(scaled_surface, (0, 0))
            frame_text = font.render(
                f"Frame: {frame_idx + 1}/{frames_to_show}", True, (255, 255, 255)
            )
            screen.blit(frame_text, (10, 10))
            pygame.display.flip()
        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            frame_idx += 1
            continue
        pygame.time.wait(delay)
        frame_idx += 1
    if running:
        pygame.time.wait(1000)
    pygame.quit()
    print(f"Rendered {frame_idx} frames from trajectory")


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
        print(flat_states.shape)
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

    gpu_batch_size = 250

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

                # print("Creating sequence from index:", start_idx + j)
                # print("Creating sequence to index:", start_idx + j + sequence_length)
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

    model = MODEL_ARCHITECTURE()

    # Improved optimizer with learning rate scheduling
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=num_epochs,
        alpha=0.1,  # Final learning rate will be 0.1 * initial
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8),
    )

    rng = jax.random.PRNGKey(42)
    dummy_state = normalized_obs[:1]
    dummy_action = actions[:1]
    params = model.init(rng, dummy_state, dummy_action, None)
    opt_state = optimizer.init(params)

    # Pre-initialize LSTM state template
    _, lstm_state_template = model.apply(params, None, dummy_state, dummy_action, None)

    # Improved loss function with multiple components
    @jax.jit
    def single_sequence_loss(
        params,
        state_batch,
        action_batch,
        next_state_batch,
        lstm_template,
        epoch,
        max_epochs,
    ):
        """Enhanced loss with feature-specific weighting"""
        seq_len, state_dim = state_batch.shape

        def scan_fn(lstm_state, inputs):
            current_state, current_action, target_next_state = inputs

            pred_next_state, new_lstm_state = model.apply(
                params, None, current_state[None, :], current_action[None], lstm_state
            )

            pred_next_state = pred_next_state.squeeze()

            # # Simpler, more balanced weighting
            # static_weight = 1.0
            # dynamic_weight = 1.0  # Equal weighting instead of 2.0

            # # Create weight mask
            # weights = jnp.concatenate([
            #     jnp.full((170,), static_weight),
            #     jnp.full((9,), dynamic_weight),
            #     jnp.full((state_dim - 179,), static_weight)
            # ])

            # Standard weighted MSE loss
            mse_loss = jnp.mean((target_next_state - pred_next_state) ** 2)
            # mse_loss = jnp.mean(weights * (target_next_state - pred_next_state) ** 2)

            # Remove the additional stability loss for now
            total_loss = mse_loss

            return new_lstm_state, total_loss

        scan_inputs = (state_batch, action_batch, next_state_batch)
        _, step_losses = lax.scan(scan_fn, lstm_template, scan_inputs)

        return jnp.mean(step_losses)

    @jax.jit
    def multi_step_sequence_loss(
        params,
        state_batch,
        action_batch,
        next_state_batch,
        lstm_template,
        epoch,
        max_epochs,
        num_steps=30,
    ):
        """Loss that includes multi-step predictions"""
        seq_len, state_dim = state_batch.shape

        def scan_fn(lstm_state, inputs):
            current_state, current_action, target_next_state = inputs

            # Single step prediction
            pred_next_state, new_lstm_state = model.apply(
                params, None, current_state[None, :], current_action[None], lstm_state
            )
            pred_next_state = pred_next_state.squeeze()

            # Multi-step predictions using the model iteratively
            multi_step_losses = []
            temp_state = pred_next_state
            temp_lstm_state = new_lstm_state

            # Predict 2, 3, ... steps ahead
            for step in range(1, min(num_steps, seq_len - inputs[0].shape[0] + 1)):
                if inputs[0].shape[0] + step < seq_len:  # Make sure we have target
                    next_action = action_batch[inputs[0].shape[0] + step]
                    target_future = next_state_batch[inputs[0].shape[0] + step]

                    pred_future, temp_lstm_state = model.apply(
                        params,
                        None,
                        temp_state[None, :],
                        next_action[None],
                        temp_lstm_state,
                    )
                    pred_future = pred_future.squeeze()

                    # Weight decreases with prediction horizon
                    weight = 0.8**step
                    multi_step_loss = weight * jnp.mean(
                        (target_future - pred_future) ** 2
                    )
                    multi_step_losses.append(multi_step_loss)

                    temp_state = pred_future

            # Combine single-step and multi-step losses
            single_step_loss = jnp.mean((target_next_state - pred_next_state) ** 2)
            total_multi_step_loss = (
                jnp.sum(jnp.array(multi_step_losses)) if multi_step_losses else 0.0
            )

            total_loss = (
                single_step_loss + 0.5 * total_multi_step_loss
            )  # Weight multi-step contribution

            return new_lstm_state, total_loss

        scan_inputs = (state_batch, action_batch, next_state_batch)
        _, step_losses = lax.scan(scan_fn, lstm_template, scan_inputs)

        return jnp.mean(step_losses)

    @jax.jit
    def scheduled_sampling_loss(
        params,
        state_batch,
        action_batch,
        next_state_batch,
        lstm_template,
        epoch,
        max_epochs,
    ):
        """Loss with scheduled sampling - gradually use model predictions instead of ground truth"""
        seq_len, state_dim = state_batch.shape

        # Probability of using model prediction increases with training progress
        use_prediction_prob = jnp.minimum(0.2, epoch / (max_epochs * 10))

        def scan_fn(carry, inputs):
            lstm_state, previous_state = carry
            current_action, target_next_state, step_idx = inputs

            # Decide whether to use ground truth or model prediction as input
            key = jax.random.PRNGKey(step_idx.astype(int))
            use_prediction = jax.random.uniform(key) < use_prediction_prob

            # Use either ground truth or previous model prediction
            input_state = jnp.where(use_prediction, previous_state, target_next_state)

            pred_next_state, new_lstm_state = model.apply(
                params, None, input_state[None, :], current_action[None], lstm_state
            )
            pred_next_state = pred_next_state.squeeze()

            loss = jnp.mean((target_next_state - pred_next_state) ** 2)

            return (new_lstm_state, pred_next_state), loss

        # Prepare inputs for scan
        step_indices = jnp.arange(seq_len)
        scan_inputs = (action_batch, next_state_batch, step_indices)

        # Start with first ground truth state
        initial_carry = (lstm_template, state_batch[0])
        _, step_losses = lax.scan(scan_fn, initial_carry, scan_inputs)

        return jnp.mean(step_losses)

    # Vectorize loss over batch dimension
    batched_loss_fn = jax.vmap(
        single_sequence_loss, in_axes=(None, 0, 0, 0, None, None, None)
    )

    @jax.jit
    def update_step_batched(
        params,
        opt_state,
        batch_states,
        batch_actions,
        batch_next_states,
        lstm_template,
        epoch=0,
        num_epochs=20000,
    ):
        # Compute loss for all sequences in parallel
        def loss_fn(p):
            losses = batched_loss_fn(
                p,
                batch_states,
                batch_actions,
                batch_next_states,
                lstm_template,
                epoch,
                num_epochs,
            )
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def compute_validation_loss(
        params,
        batch_states,
        batch_actions,
        batch_next_states,
        lstm_template,
        epoch,
        num_epochs,
    ):
        """Compute validation loss without updating parameters"""
        losses = batched_loss_fn(
            params,
            batch_states,
            batch_actions,
            batch_next_states,
            lstm_template,
            epoch,
            num_epochs,
        )
        return jnp.mean(losses)

    # Convert training and validation batches to arrays
    train_batch_states = jnp.stack([batch[0] for batch in train_batches])
    train_batch_actions = jnp.stack([batch[1] for batch in train_batches])
    train_batch_next_states = jnp.stack([batch[2] for batch in train_batches])

    val_batch_states = jnp.stack([batch[0] for batch in val_batches])
    val_batch_actions = jnp.stack([batch[1] for batch in val_batches])
    val_batch_next_states = jnp.stack([batch[2] for batch in val_batches])

    # Shuffle indices for each epoch
    rng_shuffle = jax.random.PRNGKey(123)

    # Training loop with validation tracking
    best_loss = float("inf")
    patience = 50
    no_improve_count = 0

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        rng_shuffle, shuffle_key = jax.random.split(rng_shuffle)
        indices = jax.random.permutation(shuffle_key, len(train_batches))

        shuffled_train_states = train_batch_states[train_indices]
        shuffled_train_actions = train_batch_actions[train_indices]
        shuffled_train_next_states = train_batch_next_states[train_indices]

        # only use gpu_batch_size elements for training

        # will increase the amount of epochs a lot
        if shuffled_train_states.shape[0] > gpu_batch_size:
            shuffled_train_states = shuffled_train_states[:gpu_batch_size]
            shuffled_train_actions = shuffled_train_actions[:gpu_batch_size]
            shuffled_train_next_states = shuffled_train_next_states[:gpu_batch_size]

        params, opt_state, train_loss = update_step_batched(
            params,
            opt_state,
            shuffled_train_states,
            shuffled_train_actions,
            shuffled_train_next_states,
            lstm_state_template,
            epoch,
            num_epochs,
        )

        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if False:
            # if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if VERBOSE and (epoch + 1) % 1 == 0:
            # if VERBOSE and ((epoch + 1) % max(1, num_epochs // 10) or epoch == 0) == 0:
            val_loss = compute_validation_loss(
                params,
                val_batch_states,
                val_batch_actions,
                val_batch_next_states,
                lstm_state_template,
                epoch,
                num_epochs,
            )

            current_lr = lr_schedule(epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            )

    print("Training completed")
    return params, {
        "final_loss": train_loss,
        "normalization_stats": normalization_stats,
        "best_loss": best_loss,
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

    if len(obs) == 1:
        obs = obs.squeeze(0)

    def debug_obs(
        step,
        real_obs,
        pred_obs,
        action,
    ):
        error = jnp.mean((real_obs - pred_obs[0]) ** 2)
        # print(pred_obs)
        print(
            f"Step {step}, Unnormalized Error: {error:.2f} | Action: {action_map[int(action)]} Predicted Score : {get_reward_from_observation(pred_obs[0])} | Real Score: {get_reward_from_observation(real_obs)}"
        )
        print(real_obs)

        if error > 20 and render_debugging:
            print(
                "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
            )
            print("Indexes where difference > 1:")
            for j in range(len(pred_obs[0])):
                if jnp.abs(pred_obs[0][j] - real_obs[j]) > 10:
                    print(
                        f"Prediction Index ({OBSERVATION_INDEX_MAP[j]}) {j}: {pred_obs[0][j]} vs Real Index {real_obs[j]}"
                    )
            # print(f"Difference: {pred_obs[0] - real_obs}")
            print(
                "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
            )

    def check_lstm_state_health(lstm_state, step):
        if lstm_state is not None:
            # V2_LSTM returns (lstm1_state, lstm2_state) tuple
            lstm1_state, lstm2_state = lstm_state

            # Check first LSTM layer
            hidden1_norm = jnp.linalg.norm(lstm1_state.hidden)
            cell1_norm = jnp.linalg.norm(lstm1_state.cell)

            # Check second LSTM layer
            hidden2_norm = jnp.linalg.norm(lstm2_state.hidden)
            cell2_norm = jnp.linalg.norm(lstm2_state.cell)

            # Check for problems in either layer
            max_norm = max(hidden1_norm, cell1_norm, hidden2_norm, cell2_norm)
            min_norm = min(hidden1_norm, cell1_norm, hidden2_norm, cell2_norm)

            if max_norm > 5.0:
                print(
                    f"Step {step}: LSTM state explosion - Layer1 h:{hidden1_norm:.2f} c:{cell1_norm:.2f}, Layer2 h:{hidden2_norm:.2f} c:{cell2_norm:.2f}"
                )
            elif min_norm < 0.01:
                print(
                    f"Step {step}: LSTM state vanishing - Layer1 h:{hidden1_norm:.2f} c:{cell1_norm:.2f}, Layer2 h:{hidden2_norm:.2f} c:{cell2_norm:.2f}"
                )
            else:
                # Only print occasionally when healthy to avoid spam
                if step % 50 == 0:  # Print every 50 steps when healthy
                    print(
                        f"Step {step}: LSTM states healthy - Layer1 h:{hidden1_norm:.2f} c:{cell1_norm:.2f}, Layer2 h:{hidden2_norm:.2f} c:{cell2_norm:.2f}"
                    )

    def detect_game_events(real_obs, prev_obs, step):
        # Detect significant state changes that might confuse the model
        if prev_obs is not None:
            total_change = jnp.sum(jnp.abs(real_obs - prev_obs))

            # Detect potential death/reset events
            if total_change > 50:  # Threshold to tune
                print(
                    f"Step {step}: Major state change detected (change: {total_change:.2f})"
                )

            # Check for specific entity spawn/despawn
            dynamic_change = jnp.sum(jnp.abs(real_obs[170:179] - prev_obs[170:179]))
            if dynamic_change > 20:
                print(
                    f"Step {step}: Entity spawn/despawn (dynamic change: {dynamic_change:.2f})"
                )

    def analyze_prediction_errors(real_obs, pred_obs, step):
        error_agg = jnp.mean((real_obs - pred_obs[0]) ** 2)
        error = jnp.abs(real_obs - pred_obs[0])
        # Check which parts are wrong
        static_error = jnp.mean(error[:170])
        dynamic_error = jnp.mean(error[170:179])
        other_error = jnp.mean(error[179:])

        if error_agg > 10:  # Large error threshold
            print(f"  LARGE ERROR BREAKDOWN:")
            print(f"    Static (background): {static_error:.1f}")
            print(f"    Dynamic (entities): {dynamic_error:.1f}")
            print(f"    Other: {other_error:.1f}")

            # Find the worst predictions
            worst_indices = jnp.where(error > 20)[0]
            if len(worst_indices) > 0:
                print(f"    Worst predictions at indices: {worst_indices[:5]}")

    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]

    renderer = SeaquestRenderer()
    if len(sys.argv) > 4 and sys.argv[4].startswith("check"):
        model_path = sys.argv[4]
    else:
        if os.path.exists(f"world_model_{MODEL_ARCHITECTURE.__name__}.pkl"):
            model_path = f"world_model_{MODEL_ARCHITECTURE.__name__}.pkl"
        else:
            model_path = "model.pkl"

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        dynamics_params = model_data["dynamics_params"]
        normalization_stats = model_data.get("normalization_stats", None)
    world_model = MODEL_ARCHITECTURE()

    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(
        "Real Environment vs World Model (AtariWrapper Frame Stack)"
    )

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    step_count = 0 + starting_step
    clock = pygame.time.Clock()

    # code to get the unflattener
    game = JaxSeaquest()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    dummy_obs, _ = env.reset(jax.random.PRNGKey(int(time.time())))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    # init the first observation and model observation
    real_obs = obs[0]
    model_obs = obs[0]  # Start identical

    # Initialize LSTM state for model predictions
    lstm_state = None
    lstm_real_state = None
    print(obs.shape)
    print(len(obs))

    while step_count < min(num_steps, len(obs) - 1):
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False

        # Use the saved action
        action = actions[step_count]
        # action = 5

        # Use the saved next state directly instead of environment stepping
        next_real_obs = obs[step_count + 1]

        # Check if we need to reset the model state before making prediction
        if steps_into_future > 0 and (
            step_count % steps_into_future == 0 or step_count in boundaries
        ):
            print("State reset")
            model_obs = obs[step_count]  # Reset to current real observation
            # We'll reset lstm_state to lsmt_real_state after computing it below

        # Apply model prediction with normalization and LSTM state
        normalized_flattened_model_obs = (model_obs - state_mean) / state_std

        if steps_into_future > 0:
            # Use the stateful model (returns both prediction and new LSTM state)
            normalized_model_prediction, lstm_state = world_model.apply(
                dynamics_params,
                None,
                normalized_flattened_model_obs,
                jnp.array([action]),
                lstm_state,
            )
        else:
            normalized_model_prediction = normalized_flattened_model_obs

        # unnormalized_model_prediction = (
        #     normalized_model_prediction * state_std + state_mean
        # )

        # Convert model predictions to integers
        unnormalized_model_prediction = jnp.round(
            normalized_model_prediction * state_std + state_mean
        )

        model_obs = unnormalized_model_prediction

        if steps_into_future > 0:
            debug_obs(step_count, next_real_obs, unnormalized_model_prediction, action)
        # check_lstm_state_health(lstm_state, step_count)
        # analyze_prediction_errors(next_real_obs, unnormalized_model_prediction, step_count)
        # detect_game_events(next_real_obs, real_obs, step_count)

        # Rendering stuff start -------------------------------------------------------

        real_base_state = flat_observation_to_state(
            real_obs, unflattener, frame_stack_size=frame_stack_size
        )  # Get the last state for rendering
        model_base_state = flat_observation_to_state(
            model_obs.squeeze(), unflattener, frame_stack_size=frame_stack_size
        )  # Get the last state for renderi

        # print(real_base_state)

        real_raster = renderer.render(real_base_state)
        real_img = np.array(real_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(real_surface, real_img)
        model_raster = renderer.render(model_base_state)
        model_img = np.array(model_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(model_surface, model_img)
        screen.fill((0, 0, 0))
        scaled_real = pygame.transform.scale(
            real_surface, (WIDTH * render_scale, HEIGHT * render_scale)
        )
        screen.blit(scaled_real, (0, 0))
        scaled_model = pygame.transform.scale(
            model_surface, (WIDTH * render_scale, HEIGHT * render_scale)
        )
        screen.blit(scaled_model, (WIDTH * render_scale + 20, 0))
        font = pygame.font.SysFont(None, 24)
        real_text = font.render("Real Environment", True, (255, 255, 255))
        model_text = font.render("World Model (4 Frames)", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        pygame.display.flip()

        # Rendering stuff end -------------------------------------------------------

        # Separate prediction just to have the lstm state for the current real trajectory at all times
        # This tracks the "ground truth" LSTM state
        real_obs = obs[step_count]  # Current real observation
        if steps_into_future > 0:
            normalized_real_obs = (real_obs - state_mean) / state_std
            _, lstm_real_state = world_model.apply(
                dynamics_params,
                None,
                normalized_real_obs,
                jnp.array([action]),
                lstm_real_state,
            )

        # Reset LSTM state if we're at a reset point
        if steps_into_future > 0 and (
            step_count % steps_into_future == 0 or step_count in boundaries
        ):
            # lstm_state = None
            lstm_state = lstm_real_state

        step_count += 1
        # print(obs[step_count][:-2])
        clock.tick(clock_speed)
        # Rendering stuff end -------------------------------------------------------

    pygame.quit()
    print("Comparison completed")


def add_training_noise(obs, actions, next_obs, rewards, noise_config=None):
    """Add various types of noise to training data for improved robustness"""

    if noise_config is None:
        noise_config = {
            "observation_noise": 0.01,  # Gaussian noise on observations
            "action_dropout": 0.02,  # Randomly change some actions
            "state_dropout": 0.005,  # Randomly zero out some state features
            "temporal_noise": 0.01,  # Small time-shift noise
            "entity_noise": 0.02,  # Extra noise on entity positions
        }

    key = jax.random.PRNGKey(42)
    noisy_obs = obs.copy()
    noisy_actions = actions.copy()
    noisy_next_obs = next_obs.copy()

    # 1. Observation noise - general Gaussian noise
    if noise_config["observation_noise"] > 0:
        key, subkey = jax.random.split(key)
        obs_noise = (
            jax.random.normal(subkey, obs.shape) * noise_config["observation_noise"]
        )
        noisy_obs = noisy_obs + obs_noise

        key, subkey = jax.random.split(key)
        next_obs_noise = (
            jax.random.normal(subkey, next_obs.shape)
            * noise_config["observation_noise"]
        )
        noisy_next_obs = noisy_next_obs + next_obs_noise

    # 2. Action dropout - randomly change some actions
    if noise_config["action_dropout"] > 0:
        key, subkey = jax.random.split(key)
        action_mask = (
            jax.random.uniform(subkey, (len(actions),)) < noise_config["action_dropout"]
        )
        key, subkey = jax.random.split(key)
        random_actions = jax.random.randint(subkey, (len(actions),), 0, 18)
        noisy_actions = jnp.where(action_mask, random_actions, actions)

    # 3. State feature dropout - randomly zero some features
    if noise_config["state_dropout"] > 0:
        key, subkey = jax.random.split(key)
        dropout_mask = (
            jax.random.uniform(subkey, obs.shape) < noise_config["state_dropout"]
        )
        noisy_obs = jnp.where(dropout_mask, 0, noisy_obs)

        key, subkey = jax.random.split(key)
        dropout_mask = (
            jax.random.uniform(subkey, next_obs.shape) < noise_config["state_dropout"]
        )
        noisy_next_obs = jnp.where(dropout_mask, 0, noisy_next_obs)

    # 4. Entity-specific noise (higher noise on dynamic entities)
    if noise_config["entity_noise"] > 0:
        # Add extra noise to missile positions (indices 170-174) and entity positions
        key, subkey = jax.random.split(key)
        entity_noise = (
            jax.random.normal(subkey, noisy_obs[..., 5:175].shape)
            * noise_config["entity_noise"]
        )
        noisy_obs = noisy_obs.at[..., 5:175].add(entity_noise)

        key, subkey = jax.random.split(key)
        entity_noise = (
            jax.random.normal(subkey, noisy_next_obs[..., 5:175].shape)
            * noise_config["entity_noise"]
        )
        noisy_next_obs = noisy_next_obs.at[..., 5:175].add(entity_noise)

    # 5. Ensure observations stay in reasonable bounds
    noisy_obs = jnp.clip(noisy_obs, -100, 300)
    noisy_next_obs = jnp.clip(noisy_next_obs, -100, 300)

    return noisy_obs, noisy_actions, noisy_next_obs, rewards


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

    save_path = f"world_model_{MODEL_ARCHITECTURE.__name__}.pkl"
    experience_data_path = "experience_data_LSTM.pkl"
    model = MODEL_ARCHITECTURE()
    normalization_stats = None

    # print(next_states[300][:-2])
    # pred = model.apply(dynamics_params, None, states[300], actions[300])
    # print(pred)

    # print(((next_states[300][:-2] - pred) ** 2))

    experience_its = 5

    if not os.path.exists("experience_data_LSTM_0.pkl"):
        print("No existing experience data found. Collecting new experience data...")
        # Collect experience data (AtariWrapper handles frame stacking automatically)

        for i in range(0, experience_its):
            print(f"Collecting experience data (iteration {i+1}/{experience_its})...")
            obs, actions, rewards, _, boundaries = collect_experience_sequential(
                env, num_episodes=50, max_steps_per_episode=10000, seed=i
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
    rtpt = RTPT(
        name_initials="FH", experiment_name="TestingIterateAgent", max_iterations=3
    )

    # Start the RTPT tracking
    rtpt.start()
    main()
