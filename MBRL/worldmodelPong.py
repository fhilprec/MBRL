import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from jaxatari.games.jax_pong import PongRenderer, JaxPong
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper
from jax import lax
import gc
from rtpt import RTPT
from obs_state_converter import pong_flat_observation_to_state

from model_architectures import *

MODEL_ARCHITECTURE = PongLSTM
model_scale_factor = 5


def get_reward_from_observation_score(obs):
    """Extract reward from Pong observation - adjust index as needed for Pong"""
    if len(obs) < 100:
        raise ValueError(f"Observation must have sufficient elements, got {len(obs)}")

    return obs[-3] if len(obs) > 3 else 0


def calculate_score_based_reward(flat_obs, next_flat_obs):
    """
    Calculate reward based on score difference between observations.
    Returns +1 if player scored, -1 if enemy scored, 0 otherwise.

    Args:
        flat_obs: Current observation (flattened)
        next_flat_obs: Next observation (flattened)

    Returns:
        Score-based reward: 1, -1, or 0
    """
    # Extract scores from observations
    # flat_obs[-5] is player score, flat_obs[-1] is enemy score
    player_score_old = flat_obs[..., -5]
    enemy_score_old = flat_obs[..., -1]
    player_score_new = next_flat_obs[..., -5]
    enemy_score_new = next_flat_obs[..., -1]

    # Calculate score changes
    player_scored = player_score_new - player_score_old  # Should be 0 or 1
    enemy_scored = enemy_score_new - enemy_score_old    # Should be 0 or 1

    # Reward is +1 if player scored, -1 if enemy scored, 0 otherwise
    score_reward = player_scored - enemy_scored

    # Filter out rewards where abs > 1 (safety check for corrupted data)
    score_reward = jnp.where(jnp.abs(score_reward) > 1, 0.0, score_reward)

    return score_reward





VERBOSE = True
model = None

def print_full_array(arr):
    with jnp.printoptions(threshold=jnp.inf, linewidth=200):
        print("Full observation array:")
        print(arr)


action_map = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
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
    renderer = PongRenderer()
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


def collect_experience_sequential(
    env,
    num_episodes: int = 1,
    max_steps_per_episode: int = 1000,
    episodic_life: bool = True,
    seed: int = 42,
    policy_params=None,
    network=None,
):
    """Collect experience data using JAX vmap and scan for efficiency."""

    rng = jax.random.PRNGKey(seed)

    def perfect_policy(obs, rng):
        """Perfect policy that tracks ball position"""
        rng, action_key = jax.random.split(rng)
        # 50% random exploration
        do_random = jax.random.uniform(action_key) < 0.5
        random_action = jax.random.randint(action_key, (), 0, 6)

        # Ball tracking logic
        perfect_action = jax.lax.cond(
            obs.player.y[3] > obs.ball.y[3],
            lambda _: jnp.array(4),
            lambda _: jax.lax.cond(
                obs.player.y[3] < obs.ball.y[3],
                lambda _: jnp.array(3),
                lambda _: jnp.array(0),
                None
            ),
            None
        )

        return jax.lax.select(do_random, random_action, perfect_action)

    def run_single_episode(episode_key):
        """Run one complete episode using JAX scan."""
        reset_key, step_key = jax.random.split(episode_key)
        obs, state = env.reset(reset_key)

        def step_fn(carry, _):
            rng, obs, state, done = carry

            # Skip if already done
            def continue_step(_):
                # Get action
                rng_new, action_key = jax.random.split(rng)

                if network and policy_params:
                    flat_obs, _ = flatten_obs(obs, single_state=True)
                    pi, _ = network.apply(policy_params, flat_obs)
                    action = pi.sample(seed=action_key)
                else:
                    action = perfect_policy(obs, rng_new)

                # Step environment
                next_obs, next_state, reward, next_done, _ = env.step(state, action)

                # Store transition with valid mask (valid = not done BEFORE this step)
                transition = (obs, state, action, jnp.float32(reward), next_done, ~done)

                return (rng_new, next_obs, next_state, next_done), transition

            def skip_step(_):
                # Return dummy data when done - ENSURE EXACT TYPE MATCHING
                dummy_action = jnp.array(0, dtype=jnp.int32)
                dummy_reward = jnp.array(0.0, dtype=jnp.float32)
                dummy_done = jnp.array(False, dtype=jnp.bool_)
                dummy_valid = jnp.array(False, dtype=jnp.bool_)

                dummy_transition = (obs, state, dummy_action, dummy_reward, dummy_done, dummy_valid)
                return (rng, obs, state, done), dummy_transition

            return jax.lax.cond(done, skip_step, continue_step, None)

        initial_carry = (step_key, obs, state, jnp.array(False))
        _, transitions = jax.lax.scan(step_fn, initial_carry, None, length=max_steps_per_episode)

        observations, states, actions, rewards, dones, valid_mask = transitions

        # Calculate episode length
        episode_length = jnp.sum(valid_mask)

        return observations, states, actions, rewards, dones, valid_mask, episode_length

    # Generate episode keys
    episode_keys = jax.random.split(rng, num_episodes)

    print(f"Collecting {num_episodes} episodes with vmap...")

    # Run episodes in parallel with vmap
    vmapped_episode_fn = jax.vmap(run_single_episode, in_axes=0)
    observations, states, actions, rewards, dones, valid_mask, episode_lengths = vmapped_episode_fn(episode_keys)

    # Process each episode separately to extract only valid steps
    all_valid_obs = []
    all_valid_states = []
    all_valid_actions = []
    all_valid_rewards = []
    all_valid_dones = []
    boundaries = []
    cumulative_steps = 0

    for ep_idx in range(num_episodes):
        ep_length = int(episode_lengths[ep_idx])

        if ep_length > 0:
            # Extract valid steps for this episode
            # Use tree.map for PyTree observations
            valid_obs = jax.tree.map(lambda x: x[ep_idx, :ep_length], observations)
            valid_states = jax.tree.map(lambda x: x[ep_idx, :ep_length], states)
            valid_actions = actions[ep_idx, :ep_length]
            valid_rewards = rewards[ep_idx, :ep_length]
            valid_dones = dones[ep_idx, :ep_length]

            all_valid_obs.append(valid_obs)
            all_valid_states.append(valid_states)
            all_valid_actions.append(valid_actions)
            all_valid_rewards.append(valid_rewards)
            all_valid_dones.append(valid_dones)

            cumulative_steps += ep_length
            boundaries.append(cumulative_steps)

            print(f"Episode {ep_idx+1} done after {ep_length} steps")

    # Concatenate all valid episodes
    if all_valid_obs:
        all_obs = jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *all_valid_obs)
        all_states = jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *all_valid_states)
        all_actions = jnp.concatenate(all_valid_actions, axis=0)
        all_rewards = jnp.concatenate(all_valid_rewards, axis=0)
        all_dones = jnp.concatenate(all_valid_dones, axis=0)

        # Flatten observations and states efficiently
        # Simply concatenate all leaf arrays along the feature dimension
        obs_leaves = jax.tree_util.tree_leaves(all_obs)
        state_leaves = jax.tree_util.tree_leaves(all_states)

        # Ensure all leaves are 2D (batch_size, features) before concatenating
        obs_leaves_2d = [leaf if leaf.ndim == 2 else leaf[:, None] for leaf in obs_leaves]
        state_leaves_2d = [leaf if leaf.ndim == 2 else leaf[:, None] for leaf in state_leaves]

        # Concatenate all leaves to create flat arrays
        flat_obs = jnp.concatenate(obs_leaves_2d, axis=1)
        flat_states = jnp.concatenate(state_leaves_2d, axis=1)

        print(f"Total valid steps: {cumulative_steps}")

        return (
            flat_obs,
            all_actions,
            all_rewards,
            all_dones,
            flat_states,
            boundaries,
        )
    else:
        # Return empty arrays if no valid data
        return (
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            [],
        )


def train_world_model(
    obs,
    actions,
    next_obs,
    rewards,
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=1000,
    sequence_length=32,
    episode_boundaries=None,
    frame_stack_size=4,
    model_scale_factor=1,
    checkpoint_path=None,
    save_every=10,
):

    # Increased batch size for better GPU utilization and faster training
    gpu_batch_size = 500

    gpu_batch_size = gpu_batch_size // frame_stack_size

    state_mean = jnp.mean(obs, axis=0)
    state_std = jnp.std(obs, axis=0) + 1e-8

    # IMPORTANT: Normalization is now ENABLED
    # state_mean = 0
    # state_std = 1

    normalization_stats = {"mean": state_mean, "std": state_std}

    normalized_obs = (obs - state_mean) / state_std
    normalized_next_obs = (next_obs - state_mean) / state_std

    def create_sequential_batches(batch_size=32):
        """Optimized batch creation using list comprehension and minimizing operations"""
        sequences = []
        stride = sequence_length // 4

        # Add 0 at the beginning for easier indexing
        boundaries_with_zero = [0] + episode_boundaries

        for i in range(len(episode_boundaries)):
            start_idx = boundaries_with_zero[i]
            end_idx = boundaries_with_zero[i + 1]
            episode_length = end_idx - start_idx

            # Skip episodes that are too short
            if episode_length < sequence_length:
                continue

            # Calculate valid starting positions
            num_sequences = (episode_length - sequence_length) // stride + 1

            for seq_idx in range(num_sequences):
                seq_start = start_idx + seq_idx * stride
                seq_end = seq_start + sequence_length

                # Extract sequences directly - no padding needed since we validated length
                sequences.append((
                    normalized_obs[seq_start:seq_end],
                    actions[seq_start:seq_end],
                    normalized_next_obs[seq_start:seq_end]
                ))

        return sequences

    batches = create_sequential_batches()
    print(f"Created {len(batches)} sequential batches of size {sequence_length}")

    total_batches = len(batches)
    train_size = int(0.8 * total_batches)

    rng_split = jax.random.PRNGKey(42)
    indices = jax.random.permutation(rng_split, total_batches)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_batches = [batches[i] for i in train_indices]
    val_batches = [batches[i] for i in val_indices]

    print(
        f"Training batches: {len(train_batches)}, Validation batches: {len(val_batches)}"
    )

    model = MODEL_ARCHITECTURE(model_scale_factor)
    reward_model = RewardPredictorMLP(model_scale_factor)

    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=num_epochs,
        alpha=0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8),
    )

    # Separate optimizer for reward predictor
    reward_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8),
    )

    rng = jax.random.PRNGKey(42)
    reward_rng = jax.random.PRNGKey(43)
    dummy_state = normalized_obs[:1]
    dummy_action = actions[:1]


    start_epoch = 0
    if checkpoint_path:
        checkpoint_file = checkpoint_path.replace(".pkl", "_checkpoint.pkl")
    else:
        checkpoint_file = None

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
            params = checkpoint_data["params"]
            opt_state = checkpoint_data["opt_state"]
            reward_params = checkpoint_data["reward_params"]
            reward_opt_state = checkpoint_data["reward_opt_state"]
            start_epoch = checkpoint_data["epoch"] + 1
            best_loss = checkpoint_data.get("best_loss", float("inf"))
            normalization_stats = checkpoint_data.get("normalization_stats", normalization_stats)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        params = model.init(rng, dummy_state, dummy_action, None)
        opt_state = optimizer.init(params)

        # Initialize reward predictor
        rng, reward_rng = jax.random.split(rng)
        reward_params = reward_model.init(reward_rng, dummy_state)
        reward_opt_state = reward_optimizer.init(reward_params)
        best_loss = float("inf")

    _, lstm_state_template = model.apply(params, rng, dummy_state, dummy_action, None)

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
        # Teacher forcing probability (start at 1.0, decay to 0.3 over training)
        teacher_forcing_prob = jnp.maximum(
            0.3,
            1.0 - 0.7 * (epoch / max_epochs)
        )

        # Decide ONCE per sequence whether to use teacher forcing (per-sequence, not per-step)
        use_teacher_forcing = jax.random.uniform(jax.random.fold_in(rng, 0)) < teacher_forcing_prob

        def scan_fn(carry, inputs):
            rssm_state, prev_prediction = carry
            current_state, current_action, target_next_state, step_idx = inputs

            # Use teacher forcing decision for whole sequence
            # Only first step gets ground truth when NOT using teacher forcing
            input_state = jnp.where(
                use_teacher_forcing | (step_idx == 0),
                current_state,
                prev_prediction
            )

            # Single-step prediction
            pred_next_state, new_rssm_state = model.apply(
                params, rng, input_state[None, :], current_action[None], rssm_state
            )
            pred_next_state = pred_next_state.squeeze()

            # Compute prediction error
            step_error = (target_next_state - pred_next_state) ** 2
            error_magnitude = jnp.mean(step_error)

            # Adaptive loss weighting: automatically upweight hard predictions
            # Hard predictions (large errors) get higher weight → model focuses on them
            # This is GAME-AGNOSTIC: works for any sudden state transition
            # Weight scales from 1.0 (easy) to 3.0 (hard) using tanh function
            adaptive_weight = 1.0 + 2.0 * jnp.tanh(error_magnitude * 10.0)

            # Apply adaptive weight to single-step loss
            single_step_loss = jnp.mean(step_error) * adaptive_weight

            # LSTM state regularization - prevent explosion/vanishing
            # Use tree_map to compute norms for all arrays in the LSTM state
            def compute_state_penalty(arr):
                """Compute penalty for a single state array"""
                norm = jnp.linalg.norm(arr)
                # Penalize norms outside healthy range [0.5, 3.0]
                penalty = jnp.maximum(0.0, norm - 3.0) ** 2 + jnp.maximum(0.0, 0.5 - norm) ** 2
                return penalty

            # Apply to all arrays in LSTM state and sum
            penalties = jax.tree.map(compute_state_penalty, new_rssm_state)
            # Increased from 0.001 to 0.01 for stronger regularization
            lstm_penalty = 0.01 * jnp.sum(jnp.array(jax.tree_util.tree_leaves(penalties)))

            return (new_rssm_state, pred_next_state), (single_step_loss, lstm_penalty)

        # Add step indices for teacher forcing decision
        step_indices = jnp.arange(state_batch.shape[0])
        scan_inputs = (state_batch, action_batch, next_state_batch, step_indices)

        # Initialize with zero prediction (will use teacher forcing for first step)
        initial_carry = (lstm_template, jnp.zeros_like(state_batch[0]))
        (final_rssm_state, _), (step_losses, lstm_penalties) = lax.scan(scan_fn, initial_carry, scan_inputs)

        # Single-step loss component
        single_step_loss = jnp.mean(step_losses)
        lstm_regularization = jnp.mean(lstm_penalties)

        # ========== MULTI-STEP ROLLOUT LOSS ==========
        # Curriculum learning: gradually increase horizon from 1 to 10 over training
        max_horizon = 10
        # Faster curriculum: reach horizon 10 by epoch 500 (was 889)
        # This gives model more time to learn long rollouts
        current_horizon = 1.0 + jnp.minimum(9.0, jnp.floor(9.0 * epoch / (max_epochs * 0.5)))

        # Compute multi-step losses for different horizons
        def compute_multistep_loss(horizon):
            """Rollout for 'horizon' steps and compute CUMULATIVE loss at every step"""

            def rollout_fn(carry, inputs):
                state, rssm, step = carry
                action, target = inputs

                # Only apply model if step < horizon
                def do_step(_):
                    pred, new_rssm = model.apply(params, rng, state[None, :], action[None], rssm)
                    pred = pred.squeeze()
                    # Compute loss at THIS step, not just at the end
                    step_loss = jnp.mean((pred - target) ** 2)
                    return pred, new_rssm, step_loss

                def skip_step(_):
                    return state, rssm, 0.0

                new_state, new_rssm, step_loss = lax.cond(
                    step < horizon,
                    do_step,
                    skip_step,
                    None
                )

                return (new_state, new_rssm, step + 1), step_loss

            # Check if we have enough sequence length
            seq_length = state_batch.shape[0]
            has_enough_length = horizon <= seq_length

            # Initial state and LSTM state
            init_state = state_batch[0]
            init_rssm = lstm_template

            # Use static slicing to horizon 10 (max)
            actions_slice = action_batch[:10]
            targets_slice = next_state_batch[:10]

            # Rollout and collect loss at EACH step
            _, step_losses = lax.scan(
                rollout_fn,
                (init_state, init_rssm, 0),
                (actions_slice, targets_slice)
            )

            # Average over all steps in this rollout (not just final step!)
            # This is the KEY change: model now sees error at every intermediate step
            avg_loss = jnp.mean(step_losses)

            # Only include this loss if:
            # 1. We have enough sequence length
            # 2. This horizon is within the current curriculum horizon
            horizon_active = (horizon <= current_horizon) & has_enough_length

            return jnp.where(horizon_active, avg_loss, 0.0)

        # Compute losses for horizons 2-10 (horizon 1 is already covered by single-step loss)
        # Use static list to avoid JAX tracer issues
        horizon_losses = jnp.array([
            compute_multistep_loss(2),
            compute_multistep_loss(3),
            compute_multistep_loss(4),
            compute_multistep_loss(5),
            compute_multistep_loss(6),
            compute_multistep_loss(7),
            compute_multistep_loss(8),
            compute_multistep_loss(9),
            compute_multistep_loss(10),
        ])

        # Average over active horizons (zeros don't contribute)
        num_active = jnp.sum(horizon_losses > 0.0)
        multistep_loss = jnp.where(
            num_active > 0,
            jnp.sum(horizon_losses) / num_active,
            0.0
        )

        # ========== LSTM STATE CONTINUITY LOSS ==========
        # This addresses the key problem: validation uses fresh LSTM states,
        # but real rollouts use persistent states that accumulate errors.
        # We need to train the model to handle LSTM state drift.

        # Do a full-sequence rollout with persistent LSTM state
        # (simulating what happens during actual rendering)
        def full_sequence_rollout_loss():
            """Rollout through entire sequence with persistent LSTM state"""
            def persistent_rollout_fn(carry, inputs):
                state, rssm = carry
                action, target = inputs

                pred, new_rssm = model.apply(params, rng, state[None, :], action[None], rssm)
                pred = pred.squeeze()

                # Loss at this step
                step_loss = jnp.mean((pred - target) ** 2)

                return (pred, new_rssm), step_loss

            # Start from first state with fresh LSTM
            init_state = state_batch[0]
            init_rssm = lstm_template

            # Roll through entire sequence, feeding predictions back
            inputs = (action_batch, next_state_batch)
            _, step_losses = lax.scan(persistent_rollout_fn, (init_state, init_rssm), inputs)

            # Weight later steps more heavily (they're harder and more important)
            num_steps = step_losses.shape[0]
            step_weights = jnp.linspace(1.0, 2.0, num_steps)  # 1.0 → 2.0
            weighted_losses = step_losses * step_weights

            return jnp.mean(weighted_losses)

        # Compute this loss (it's expensive but crucial)
        continuity_loss = full_sequence_rollout_loss()

        # Combined loss: single-step + multi-step + continuity + LSTM regularization
        # Weights:
        # - 1.0 for single-step (basic accuracy)
        # - 1.0 for multi-step (short-horizon stability with per-step errors)
        # - 2.0 for continuity (long-horizon with persistent LSTM - CRITICAL for model-based RL!)
        # - lstm_regularization (already scaled)
        total_loss = single_step_loss + 1.0 * multistep_loss + 2.0 * continuity_loss + lstm_regularization

        return total_loss

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

    def compute_multistep_validation_metric(
        params,
        batch_states,
        batch_actions,
        batch_next_states,
        lstm_template,
        horizon=10,
    ):
        """Compute multi-step prediction error for validation

        Note: Not JIT compiled to allow dynamic horizon parameter.
        Horizon must be static (not a traced value).
        """
        def single_sequence_multistep(state_seq, action_seq, target_seq):
            def rollout_fn(carry, action):
                state, rssm_state = carry
                pred, new_rssm = model.apply(params, rng, state[None, :], action[None], rssm_state)
                return (pred.squeeze(), new_rssm), pred.squeeze()

            # Take first state as initial condition
            init_state = state_seq[0]
            # Rollout for horizon steps (horizon is static, so this is fine)
            actions_to_use = action_seq[:horizon]
            _, predictions = lax.scan(rollout_fn, (init_state, lstm_template), actions_to_use)

            # Compare to ground truth
            targets = target_seq[:horizon]
            errors = jnp.mean((predictions - targets) ** 2, axis=-1)  # MSE per step
            return errors

        # Vmap over batch - this is JIT compiled internally
        vmapped_fn = jax.jit(jax.vmap(single_sequence_multistep, in_axes=(0, 0, 0)))
        all_errors = vmapped_fn(batch_states, batch_actions, batch_next_states)

        # Return mean error per step (shape: [horizon])
        return jnp.mean(all_errors, axis=0)

    @jax.jit
    def reward_update_step(reward_params, reward_opt_state, batch_obs, batch_next_obs):
        """Update reward predictor parameters"""
        def reward_loss_fn(r_params):
            # Flatten batch dimensions for processing
            flat_obs = batch_obs.reshape(-1, batch_obs.shape[-1])
            flat_next_obs = batch_next_obs.reshape(-1, batch_next_obs.shape[-1])

            # Denormalize observations to get raw scores
            unnorm_obs = flat_obs * state_std + state_mean
            unnorm_next_obs = flat_next_obs * state_std + state_mean

            # Calculate target rewards based on score difference
            target_rewards = calculate_score_based_reward(unnorm_obs, unnorm_next_obs)
            # Use jax.debug.print with formatting to avoid trace-time stringification
            # 'summarize' controls how many elements are printed
            # jax.debug.print("target_rewards (shape {s}): {r}", s=target_rewards.shape, r=target_rewards)
            # Predict rewards - use UNNORMALIZED observations so inference matches training
            predicted_rewards = reward_model.apply(r_params, reward_rng, unnorm_next_obs)

            # MSE loss
            loss = jnp.mean((predicted_rewards - target_rewards) ** 2)
            return loss

        loss, grads = jax.value_and_grad(reward_loss_fn)(reward_params)
        updates, new_reward_opt_state = reward_optimizer.update(grads, reward_opt_state, reward_params)
        new_reward_params = optax.apply_updates(reward_params, updates)
        return new_reward_params, new_reward_opt_state, loss

    # Don't stack all batches upfront - keep as list and stack on-demand
    # This saves massive memory and startup time
    rng_shuffle = jax.random.PRNGKey(123)

    patience = 50
    no_improve_count = 0

    # Warm up JIT compilation with a small batch
    print("Warming up JIT compilation...")
    warmup_states = jnp.stack([train_batches[0][0]])
    warmup_actions = jnp.stack([train_batches[0][1]])
    warmup_next = jnp.stack([train_batches[0][2]])
    _, _, _ = update_step_batched(
        params, opt_state, warmup_states, warmup_actions, warmup_next,
        lstm_state_template, 0, num_epochs
    )
    print("JIT warmup complete")

    # Use tqdm for progress tracking
    pbar = tqdm(range(start_epoch, num_epochs), desc="Training", initial=start_epoch, total=num_epochs)

    for epoch in pbar:

        epoch_shuffle_key = jax.random.fold_in(rng_shuffle, epoch)
        epoch_indices = jax.random.permutation(epoch_shuffle_key, len(train_batches))

        # Shuffle batch indices, not the data itself (more efficient)
        shuffled_train_batches = [train_batches[i] for i in epoch_indices]

        # Process all training data in chunks to improve stability
        num_train_batches = len(shuffled_train_batches)
        num_chunks = (num_train_batches + gpu_batch_size - 1) // gpu_batch_size

        epoch_train_losses = []
        epoch_reward_losses = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * gpu_batch_size
            chunk_end = min((chunk_idx + 1) * gpu_batch_size, num_train_batches)

            # Stack batches on-demand only for this chunk
            chunk_batch_slice = shuffled_train_batches[chunk_start:chunk_end]
            chunk_states = jnp.stack([b[0] for b in chunk_batch_slice])
            chunk_actions = jnp.stack([b[1] for b in chunk_batch_slice])
            chunk_next_states = jnp.stack([b[2] for b in chunk_batch_slice])

            params, opt_state, chunk_train_loss = update_step_batched(
                params,
                opt_state,
                chunk_states,
                chunk_actions,
                chunk_next_states,
                lstm_state_template,
                epoch,
                num_epochs,
            )

            # Train reward predictor
            reward_params, reward_opt_state, chunk_reward_loss = reward_update_step(
                reward_params,
                reward_opt_state,
                chunk_states,
                chunk_next_states,
            )

            epoch_train_losses.append(chunk_train_loss)
            epoch_reward_losses.append(chunk_reward_loss)

        # Average losses over all chunks
        train_loss = jnp.mean(jnp.array(epoch_train_losses))
        reward_loss = jnp.mean(jnp.array(epoch_reward_losses))

        if train_loss < best_loss:
            best_loss = train_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if False and no_improve_count >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs"
            )
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
            break

        if VERBOSE and (epoch + 1) % 10 == 0:
            # Compute validation less frequently to save time (every 50 epochs instead of 10)
            if (epoch + 1) % 50 == 0:
                # Stack validation batches on-demand only when needed
                val_batch_states = jnp.stack([batch[0] for batch in val_batches])
                val_batch_actions = jnp.stack([batch[1] for batch in val_batches])
                val_batch_next_states = jnp.stack([batch[2] for batch in val_batches])

                val_loss = compute_validation_loss(
                    params,
                    val_batch_states,
                    val_batch_actions,
                    val_batch_next_states,
                    lstm_state_template,
                    epoch,
                    num_epochs,
                )

                # Compute 10-step prediction metric
                multistep_errors = compute_multistep_validation_metric(
                    params,
                    val_batch_states,
                    val_batch_actions,
                    val_batch_next_states,
                    lstm_state_template,
                    horizon=10,
                )
                # Report the 10th step error (most important for your use case)
                step_10_error = multistep_errors[9] if len(multistep_errors) >= 10 else multistep_errors[-1]

                current_lr = lr_schedule(epoch)
                log_message = f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Reward Loss: {reward_loss:.6f}, 10-Step MSE: {step_10_error:.6f}, LR: {current_lr:.2e}"
            else:
                # Just print training loss without validation
                current_lr = lr_schedule(epoch)
                log_message = f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Reward Loss: {reward_loss:.6f}, LR: {current_lr:.2e}"

            print(log_message)

            # Write to log file
            with open("world_model_training_log", "a") as log_file:
                log_file.write(log_message + "\n")

            # Update progress bar
            if (epoch + 1) % 50 == 0:
                pbar.set_postfix({"train_loss": f"{train_loss:.6f}", "reward_loss": f"{reward_loss:.6f}", "10-step": f"{step_10_error:.6f}"})
            else:
                pbar.set_postfix({"train_loss": f"{train_loss:.6f}", "reward_loss": f"{reward_loss:.6f}"})

        # Save checkpoint every save_every epochs
        if checkpoint_file and (epoch + 1) % save_every == 0:
            checkpoint_data = {
                "params": params,
                "opt_state": opt_state,
                "reward_params": reward_params,
                "reward_opt_state": reward_opt_state,
                "epoch": epoch,
                "best_loss": best_loss,
                "normalization_stats": normalization_stats,
            }
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)


    pbar.close()
    print("Training completed")
    return params, reward_params, {
        "final_loss": train_loss,
        "final_reward_loss": reward_loss,
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
    clock_speed=20,
    boundaries=None,
    env=None,
    starting_step: int = 0,
    render_debugging: bool = False,
    frame_stack_size: int = 4,
    model_scale_factor=4,
    model_path=None,
    show_only_one_step = False,
    reward_predictor_params=None,
):

    rng = jax.random.PRNGKey(0)
    print(obs.shape)

    if len(obs) == 1:
        obs = obs.squeeze(0)

    def debug_obs(
        step,
        real_obs,
        pred_obs,
        action,
    ):
        # pred_obs is now squeezed, so it's 1D
        error = jnp.mean((real_obs - pred_obs) ** 2)
        # print(
        #     f"Step {step}, MSE Error: {error:.4f} | Action: {action_map.get(int(action), action)} FOR DEBUGGING : Player y position : {pred_obs[7]:.2f} "
        # )

        # print(pred_obs)
        # print the reward model prediction
        if reward_predictor_params is not None:
            reward_model = RewardPredictorMLP(model_scale_factor)
            predicted_reward = reward_model.apply(reward_predictor_params, rng, real_obs)
            # rounded_reward = jnp.round(predicted_reward * 2) / 2
            if abs(predicted_reward) > 0.3:
                print(f"Step {step}, Reward Model Prediction: {predicted_reward}")

        if error > 20 and render_debugging:
            print("-" * 100)
            print("Indexes where difference > 1:")
            for j in range(len(pred_obs)):
                if jnp.abs(pred_obs[j] - real_obs[j]) > 10:
                    print(
                        f"Index {j}: Predicted {pred_obs[j]:.2f} vs Real {real_obs[j]:.2f}"
                    )
            print("-" * 100)

    def check_lstm_state_health(lstm_state, step):
        if lstm_state is not None:

            lstm1_state, lstm2_state = lstm_state

            hidden1_norm = jnp.linalg.norm(lstm1_state.hidden)
            cell1_norm = jnp.linalg.norm(lstm1_state.cell)

            hidden2_norm = jnp.linalg.norm(lstm2_state.hidden)
            cell2_norm = jnp.linalg.norm(lstm2_state.cell)

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

                if step % 50 == 0:
                    print(
                        f"Step {step}: LSTM states healthy - Layer1 h:{hidden1_norm:.2f} c:{cell1_norm:.2f}, Layer2 h:{hidden2_norm:.2f} c:{cell2_norm:.2f}"
                    )

    if normalization_stats:
        state_mean = normalization_stats["mean"]
        state_std = normalization_stats["std"]
    else:
        state_mean = 0
        state_std = 1

    renderer = PongRenderer()
    if not model_path:
        if len(sys.argv) > 4 and sys.argv[4].startswith("check"):
            model_path = sys.argv[4]
        else:
            # Try final model first
            final_model = f"world_model_{MODEL_ARCHITECTURE.__name__}.pkl"
            checkpoint_model = f"world_model_{MODEL_ARCHITECTURE.__name__}_checkpoint.pkl"

            if os.path.exists(final_model):
                model_path = final_model
            elif os.path.exists(checkpoint_model):
                model_path = checkpoint_model
            else:
                model_path = "model_pong.pkl"

    if steps_into_future != 0:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            # Check if it's a checkpoint or final model
            if "dynamics_params" in model_data:
                dynamics_params = model_data["dynamics_params"]
                normalization_stats = model_data.get("normalization_stats", None)
            else:
                # It's a checkpoint
                dynamics_params = model_data["params"]
                normalization_stats = model_data.get("normalization_stats", None)

    # Initialize world_model outside the if block so it's always available
    world_model = MODEL_ARCHITECTURE(model_scale_factor)

    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs World Model (Pong)")

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    step_count = 0 + starting_step
    clock = pygame.time.Clock()

    game = JaxPong()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    dummy_obs, _ = env.reset(jax.random.PRNGKey(int(time.time())))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    real_obs = obs[0]
    model_obs = obs[0]

    # Initialize LSTM state (only if using model predictions)
    if steps_into_future > 0:
        dummy_action = jnp.zeros(1, dtype=jnp.int32)
        normalized_init_obs = (obs[0] - state_mean) / state_std
        _, lstm_state = world_model.apply(
            dynamics_params, rng, normalized_init_obs, dummy_action, None
        )
    else:
        lstm_state = None

    lstm_real_state = None

    model_base_state = None



    while step_count < min(num_steps, len(obs) - 1):

        action = actions[step_count]
        # if int(step_count / 50) % 2 == 0:
        #     action = jnp.array(3) #overwrite for testing
        # else:
        #     action = jnp.array(4) #overwrite for testing
        # print(
        #     f"Reward : {improved_pong_reward(obs[step_count + 1], action, frame_stack_size=frame_stack_size):.2f}"
        # )

        next_real_obs = obs[step_count + 1]

        if steps_into_future > 0 and (
            step_count % steps_into_future == 0 or step_count in boundaries
        ):
            print("State reset")
            model_obs = obs[step_count]

        normalized_flattened_model_obs = (model_obs - state_mean) / state_std

        if steps_into_future > 0:

            normalized_model_prediction, lstm_state = world_model.apply(
                dynamics_params,
                rng,
                normalized_flattened_model_obs,
                jnp.array([action]),
                lstm_state,
            )
        else:
            normalized_model_prediction = normalized_flattened_model_obs

        # Denormalize WITHOUT rounding to avoid error accumulation
        # The model was trained on continuous values, not quantized ones
        unnormalized_model_prediction = normalized_model_prediction * state_std + state_mean

        # Squeeze batch dimension to maintain shape consistency (feature_dim,)
        model_obs = unnormalized_model_prediction.squeeze()

        # if steps_into_future > 0:
        debug_obs(step_count, next_real_obs, model_obs, action)

        real_base_state = pong_flat_observation_to_state(
            real_obs, unflattener, frame_stack_size=frame_stack_size
        )
        model_base_state = pong_flat_observation_to_state(
            model_obs.squeeze(), unflattener, frame_stack_size=frame_stack_size
        )

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
        model_text = font.render("World Model (Pong)", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        pygame.display.flip()

        real_obs = obs[step_count]
        if steps_into_future > 0:
            normalized_real_obs = (real_obs - state_mean) / state_std
            _, lstm_real_state = world_model.apply(
                dynamics_params,
                rng,
                normalized_real_obs,
                jnp.array([action]),
                lstm_real_state,
            )

        if steps_into_future > 0 and (
            step_count % steps_into_future == 0 or step_count in boundaries
        ):
            lstm_state = lstm_real_state

        step_count += 1
        clock.tick(clock_speed)
        #we are doing this just for testing now
        if show_only_one_step:
            print_obs = real_obs[(4 - 1) :: 4]
            print_full_array(print_obs)
            time.sleep(1)
            break
        

    pygame.quit()
    print("Comparison completed")

import time

def main():

    frame_stack_size = 4

    game = JaxPong()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    env = FlattenObservationWrapper(env)

    save_path = f"world_model_{MODEL_ARCHITECTURE.__name__}.pkl"
    experience_data_path = "experience_data_LSTM_pong.pkl"
    model = MODEL_ARCHITECTURE(model_scale_factor)
    normalization_stats = None

    experience_its = 5

    if not os.path.exists("experience_data_LSTM_pong_0.pkl"):
        print("No existing experience data found. Collecting new experience data...")

        for i in range(0, experience_its):
            print(f"Collecting experience data (iteration {i+1}/{experience_its})...")
            obs, actions, rewards, _, states, boundaries = collect_experience_sequential(
                env, num_episodes=40, max_steps_per_episode=10000, seed=i
            )

            


            next_obs = obs[1:]
            obs = obs[:-1]

            experience_path = "experience_data_LSTM_pong" + "_" + str(i) + ".pkl"

            with open(experience_path, "wb") as f:
                pickle.dump(
                    {
                        "obs": obs,
                        "actions": actions,
                        "next_obs": next_obs,
                        "rewards": rewards,
                        "boundaries": boundaries,
                        "states": states,
                    },
                    f,
                )
            print(f"Experience data saved to {experience_path}")

            del obs, actions, rewards, boundaries, next_obs
            gc.collect()

    obs = []
    actions = []
    next_obs = []
    rewards = []
    boundaries = []

    # Check if we should skip training (for rendering only)
    skip_training = len(sys.argv) > 1 and sys.argv[1] in ["render", "renderonce"]

    if skip_training:
        checkpoint_file = save_path.replace(".pkl", "_checkpoint.pkl")

        # Try final model first, then checkpoint
        if os.path.exists(save_path):
            print(f"Loading existing model from {save_path} (training skipped)...")
            with open(save_path, "rb") as f:
                saved_data = pickle.load(f)
                dynamics_params = saved_data["dynamics_params"]
                reward_predictor_params = saved_data.get("reward_predictor_params", None)
                normalization_stats = saved_data.get("normalization_stats", None)
        elif os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file} (training skipped)...")
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
                dynamics_params = checkpoint_data["params"]
                reward_predictor_params = checkpoint_data["reward_params"]
                normalization_stats = checkpoint_data.get("normalization_stats", None)
        else:
            print(f"Error: No model found. Looked for {save_path} and {checkpoint_file}")
            print("Please train first.")
            sys.exit(1)
    else:
        # Load experience data for training
        for i in range(0, experience_its):
            experience_path = "experience_data_LSTM_pong" + "_" + str(i) + ".pkl"
            with open(experience_path, "rb") as f:
                saved_data = pickle.load(f)
                obs.extend(saved_data["obs"])
                actions.extend(saved_data["actions"])
                next_obs.extend(saved_data["next_obs"])
                rewards.extend(saved_data["rewards"])

                offset = boundaries[-1] if boundaries else 0

                adjusted_boundaries = [b + offset for b in saved_data["boundaries"]]
                boundaries.extend(adjusted_boundaries)

        obs_array = jnp.array(obs)
        actions_array = jnp.array(actions)
        next_obs_array = jnp.array(next_obs)
        rewards_array = jnp.array(rewards)

        # Train or continue training
        if os.path.exists(save_path):
            print(f"Existing model found. Continuing training from checkpoint...")
        else:
            print("No existing model found. Training a new model...")

        dynamics_params, reward_predictor_params, training_info = train_world_model(
            obs_array,
            actions_array,
            next_obs_array,
            rewards_array,
            episode_boundaries=boundaries,
            frame_stack_size=frame_stack_size,
            model_scale_factor=model_scale_factor,
            checkpoint_path=save_path,
            save_every=10,
        )
        normalization_stats = training_info.get("normalization_stats", None)

        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "dynamics_params": dynamics_params,
                    "reward_predictor_params": reward_predictor_params,
                    "normalization_stats": training_info.get(
                        "normalization_stats", None
                    ),
                },
                f,
            )
        print(f"Model saved to {save_path}")

    gc.collect()

    with open(f"experience_data_LSTM_pong_{0}.pkl", "rb") as f:
        saved_data = pickle.load(f)
        obs = saved_data["obs"]
        actions = saved_data["actions"]
        next_obs = saved_data["next_obs"]
        rewards = saved_data["rewards"]
        boundaries = saved_data["boundaries"]

    if len(args := sys.argv) > 1 and args[1] == "renderonce":


        
        #just for now
        for i in range(100):
            shuffled_obs = jax.random.permutation(jax.random.PRNGKey(i), obs)
            compare_real_vs_model(
                num_steps=1000,
                render_scale=2,
                obs=shuffled_obs,
                actions=actions,
                normalization_stats=normalization_stats,
                boundaries=boundaries,
                env=env,
                starting_step=0,
                steps_into_future=10,
                render_debugging=(args[3] == "verbose" if len(args) > 3 else False),
                frame_stack_size=frame_stack_size,
                model_scale_factor=model_scale_factor,
                show_only_one_step=True
            )
    if len(args := sys.argv) > 1 and args[1] == "render":
        compare_real_vs_model(
                num_steps=1000,
                render_scale=2,
                obs=obs,
                actions=actions,
                normalization_stats=normalization_stats,
                boundaries=boundaries,
                env=env,
                starting_step=0,
                steps_into_future=10,
                render_debugging=(args[3] == "verbose" if len(args) > 3 else False),
                frame_stack_size=frame_stack_size,
                model_scale_factor=model_scale_factor,
                reward_predictor_params=reward_predictor_params,
            )


if __name__ == "__main__":
    rtpt = RTPT(name_initials="FH", experiment_name="WorldModelTraining", max_iterations=3)

    rtpt.start()
    main()
