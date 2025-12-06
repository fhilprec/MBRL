"""
Lightweight MLP World Model for Seaquest

A simple, fast world model using:
- Frame stacking (4 frames) for temporal information
- Simple MLP architecture (no recurrent state)
- Life-aware batching (sequences don't cross death events)
- Full JAX JIT compilation for speed
"""

import time
import jax
import jax.numpy as jnp
import optax
import pickle
import os
import sys
import pygame
from tqdm import tqdm
from typing import Tuple, Any
import numpy as np

# Import from jaxatari for environment
from obs_state_converter import flat_observation_to_state, OBSERVATION_INDEX_MAP
from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper

# Import from existing codebase
from model_architectures import SeaquestMLPDeep, RewardPredictorMLPTransition


MODEL_SCALE_FACTOR = 1  # Keep at 1 for speed

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

# ============================================================================
# MLP World Model
# ============================================================================

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
    model_path=None,
    show_only_one_step=False,
    reward_predictor_params=None,
    calc_score_based_reward: bool = True,
    print_error: bool = True,
    rollout_length: int = 0,
):

    rng = jax.random.PRNGKey(0)

    if len(obs) == 1:
        obs = obs.squeeze(0)

    def debug_obs(
        step,
        real_obs,
        pred_obs,
        action,
        previous_model_obs=None,
    ):
        # pred_obs is now squeezed, so it's 1D
        error = jnp.mean((real_obs - pred_obs) ** 2)
        if print_error:
            print(
                f"Step {step}, MSE Error: {error:.4f} | Action: {action_map.get(int(action), action)}"
            )

        if error > 20 and render_debugging:
            print("-" * 100)
            print("Indexes where difference > 1:")
            for j in range(len(pred_obs)):
                if jnp.abs(pred_obs[j] - real_obs[j]) > 10:
                    obs_name = OBSERVATION_INDEX_MAP.get(j, f"unknown_{j}")
                    print(
                        f"Index {j} ({obs_name}): Predicted {pred_obs[j]:.2f} vs Real {real_obs[j]:.2f}"
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

    renderer = SeaquestRenderer()


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
    world_model = SeaquestMLPDeep(MODEL_SCALE_FACTOR)

    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs World Model (Seaquest)")

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    step_count = 0 + starting_step
    clock = pygame.time.Clock()

    game = JaxSeaquest()
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

    reset = False

    # Pre-create font outside loop for performance
    font = pygame.font.SysFont(None, 24)
    real_text = font.render("Real Environment", True, (255, 255, 255))
    model_text = font.render("World Model (Seaquest)", True, (255, 255, 255))

    while step_count < min(num_steps, len(obs) - 1):

        # Handle pygame events to prevent freezing (check less frequently)
        if step_count % 10 == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        action = actions[step_count]
        next_real_obs = obs[step_count + 1]

        if rollout_length and (step_count % rollout_length) - 1 == 0:
            print("NEXT ROLLOUT")
            time.sleep(0.25)

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
        unnormalized_model_prediction = jnp.round(
            normalized_model_prediction * state_std + state_mean
        )

        # Squeeze batch dimension to maintain shape consistency (feature_dim,)
        model_obs = unnormalized_model_prediction.squeeze()

        if print_error:
            if steps_into_future == 0 or reset:
                debug_obs(step_count, next_real_obs, model_obs, action)
            if steps_into_future > 0 and not reset:
                previous_model_obs = normalized_flattened_model_obs * state_std + state_mean
                debug_obs(step_count, next_real_obs, model_obs, action, previous_model_obs)

        # Convert observations to Seaquest states for rendering
        real_base_state = flat_observation_to_state(
            real_obs, unflattener, frame_stack_size=frame_stack_size
        )
        model_base_state = flat_observation_to_state(
            model_obs, unflattener, frame_stack_size=frame_stack_size
        )

        # Render both states
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
            lstm_state = None
            reset = True

        step_count += 1
        clock.tick(clock_speed)

        reset = False

    pygame.quit()
    print("Comparison completed")



def create_world_model(model_scale_factor=MODEL_SCALE_FACTOR, use_deep=True):
    """
    Create the world model.
    use_deep=True uses SeaquestMLPDeep (4 layers with LayerNorm and residual connections)
    use_deep=False uses SeaquestMLPLight (2 layers, faster but less expressive)
    """

    return SeaquestMLPDeep(MODEL_SCALE_FACTOR)



# ============================================================================
# Helper Functions
# ============================================================================


def flatten_obs(state, single_state: bool = False) -> Tuple[jnp.ndarray, Any]:
    """Flatten the state PyTree into a single array.

    For Seaquest, this maintains the same approach as Pong but handles
    Seaquest-specific state structure.
    """
    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener

    # Get all leaves - each has shape (..., frame_stack_size) where frame_stack_size is typically 4
    # Concatenate along last axis to get full observation
    leaves = jax.tree_util.tree_leaves(state)
    flat_state = jnp.concatenate(leaves, axis=-1)

    return flat_state, None


def detect_life_boundaries_vectorized(obs, next_obs, frame_stack_size=4):
    """
    Detect death events in Seaquest using oxygen level or lives remaining.

    Args:
        obs: (N, feature_dim) current observations
        next_obs: (N, feature_dim) next observations

    Returns:
        Boolean array of shape (N,) indicating death events
    """
    # For Seaquest, we can detect deaths by checking if oxygen resets to max
    # or if the submarine position changes dramatically (respawn)
    # This needs to be adapted based on Seaquest's observation structure

    # Placeholder: detect large state changes indicating death/reset
    total_change = jnp.sum(jnp.abs(next_obs - obs), axis=-1)
    death_threshold = 50.0  # Tune this based on Seaquest dynamics

    death_events = total_change > death_threshold

    return death_events


# ============================================================================
# Experience Collection (JAX-accelerated with vmap + scan)
# ============================================================================


def collect_experience(
    env,
    num_episodes=100,
    max_steps_per_episode=1000,
    frame_stack_size=4,
    exploration_rate=0.5,
    seed=42,
    actor_params=None,
    actor_network=None,
):
    """
    Collect experience data using JAX vmap and scan for speed.

    Uses a trained actor policy if provided, otherwise uses random policy with exploration.
    """
    rng = jax.random.PRNGKey(seed)

    def random_policy(obs, rng):
        """Random exploration policy for Seaquest."""
        rng, action_key = jax.random.split(rng)

        # If actor is provided, use it instead of random
        if actor_params is not None and actor_network is not None:
            # Flatten observation for actor
            flat_obs, _ = flatten_obs(obs, single_state=True)
            pi = actor_network.apply(actor_params, flat_obs)

            # Sample action with exploration
            do_random = jax.random.uniform(action_key) < exploration_rate
            random_action = jax.random.randint(action_key, (), 0, 18)
            actor_action = pi.sample(seed=action_key)

            return jax.lax.select(do_random, random_action, actor_action)
        else:
            # Pure random policy for Seaquest (18 actions)
            random_action = jax.random.randint(action_key, (), 0, 18)
            return random_action

    def run_single_episode(episode_key):
        """Run one complete episode using JAX scan."""
        reset_key, step_key = jax.random.split(episode_key)
        obs, state = env.reset(reset_key)

        def step_fn(carry, _):
            rng, obs, state, done = carry

            def continue_step(_):
                rng_new, _ = jax.random.split(rng)
                action = random_policy(obs, rng_new)

                next_obs, next_state, reward, next_done, _ = env.step(state, action)
                transition = (obs, action, jnp.float32(reward), next_done, ~done)

                return (rng_new, next_obs, next_state, next_done), transition

            def skip_step(_):
                dummy_action = jnp.array(0, dtype=jnp.int32)
                dummy_reward = jnp.array(0.0, dtype=jnp.float32)
                dummy_done = jnp.array(False, dtype=jnp.bool_)
                dummy_valid = jnp.array(False, dtype=jnp.bool_)
                dummy_transition = (
                    obs,
                    dummy_action,
                    dummy_reward,
                    dummy_done,
                    dummy_valid,
                )
                return (rng, obs, state, done), dummy_transition

            return jax.lax.cond(done, skip_step, continue_step, None)

        initial_carry = (step_key, obs, state, jnp.array(False))
        _, transitions = jax.lax.scan(
            step_fn, initial_carry, None, length=max_steps_per_episode
        )

        observations, actions, rewards, dones, valid_mask = transitions
        episode_length = jnp.sum(valid_mask)

        return observations, actions, rewards, dones, valid_mask, episode_length

    # Generate episode keys and run in parallel
    episode_keys = jax.random.split(rng, num_episodes)

    print(f"Collecting {num_episodes} episodes with vmap + scan...")
    vmapped_episode_fn = jax.vmap(run_single_episode)
    observations, actions, rewards, _, _, episode_lengths = vmapped_episode_fn(
        episode_keys
    )

    print("Processing collected data (fully vectorized)...")

    # Flatten all observations at once (vectorized)
    flat_obs_all, _ = flatten_obs(observations)

    # Create a mask for valid transitions across all episodes
    # Valid transition at (ep, step) if step < episode_length - 1
    max_steps = flat_obs_all.shape[1]
    step_indices = jnp.arange(max_steps)

    # Broadcast: valid_mask[ep, step] = (step < episode_length[ep] - 1)
    valid_mask = step_indices[None, :] < (episode_lengths[:, None] - 1)

    # Extract all transitions at once using the mask
    # Flatten across episodes: (num_episodes, max_steps, features) -> (num_episodes * max_steps, features)
    flat_obs_reshaped = flat_obs_all.reshape(-1, flat_obs_all.shape[-1])
    actions_reshaped = actions.reshape(-1)
    rewards_reshaped = rewards.reshape(-1)
    valid_mask_flat = valid_mask.reshape(-1)

    # Create next_obs by shifting: for each (ep, step), next_obs is (ep, step+1)
    # Shift observations within each episode
    next_obs_all = jnp.roll(flat_obs_all, -1, axis=1)  # Shift left along time axis
    next_obs_reshaped = next_obs_all.reshape(-1, flat_obs_all.shape[-1])

    # Filter to only valid transitions
    all_obs = flat_obs_reshaped[valid_mask_flat]
    all_next_obs = next_obs_reshaped[valid_mask_flat]
    all_actions = actions_reshaped[valid_mask_flat]
    all_rewards = rewards_reshaped[valid_mask_flat]

    # Compute episode boundaries (cumulative sum of valid transitions per episode)
    valid_transitions_per_episode = jnp.sum(valid_mask, axis=1)
    episode_boundaries = jnp.cumsum(valid_transitions_per_episode).tolist()

    # Detect life boundaries (vectorized across all valid transitions)
    print("Detecting life boundaries...")
    boundary_mask = detect_life_boundaries_vectorized(
        all_obs, all_next_obs, frame_stack_size
    )
    life_boundaries = jnp.where(boundary_mask)[0].tolist()

    print(f"Collected {len(all_obs)} total transitions")
    print(f"Found {len(life_boundaries)} death events (life boundaries)")
    print(f"Episode boundaries: {len(episode_boundaries)}")

    return {
        "obs": all_obs,
        "actions": all_actions,
        "next_obs": all_next_obs,
        "rewards": all_rewards,
        "episode_boundaries": episode_boundaries,
        "life_boundaries": life_boundaries,
    }


# ============================================================================
# Training
# ============================================================================


def create_life_aware_batches(data, frame_stack_size=4):
    """
    Create training indices that don't include transitions crossing life boundaries.
    """
    obs = data["obs"]
    episode_boundaries = set(data["episode_boundaries"])
    life_boundaries = set(data.get("life_boundaries", []))

    # Combine all boundaries - we don't want to predict across these
    all_boundaries = episode_boundaries | life_boundaries

    # Create valid indices (exclude the step right before a boundary)
    valid_indices = []
    for i in range(len(obs)):
        # The transition from i to i+1 is invalid if i+1 is a boundary
        # (because that means something unusual happened between i and i+1)
        if (i + 1) not in all_boundaries and i < len(obs):
            valid_indices.append(i)

    valid_indices = jnp.array(valid_indices)
    print(
        f"Valid training indices: {len(valid_indices)} / {len(obs)} "
        f"({100 * len(valid_indices) / len(obs):.1f}%)"
    )

    return valid_indices


def create_sequence_indices(data, sequence_length=4):
    """
    Create indices where we can do sequence_length steps without hitting a boundary.
    Used for multi-step rollout training.
    """
    obs = data["obs"]
    episode_boundaries = set(data["episode_boundaries"])
    life_boundaries = set(data.get("life_boundaries", []))
    all_boundaries = episode_boundaries | life_boundaries

    valid_sequence_starts = []
    for i in range(len(obs) - sequence_length):
        # Check that no boundary exists in [i+1, i+sequence_length]
        valid = True
        for j in range(1, sequence_length + 1):
            if (i + j) in all_boundaries:
                valid = False
                break
        if valid:
            valid_sequence_starts.append(i)

    valid_sequence_starts = jnp.array(valid_sequence_starts)
    print(
        f"Valid sequence starts (len={sequence_length}): {len(valid_sequence_starts)} / {len(obs)} "
        f"({100 * len(valid_sequence_starts) / len(obs):.1f}%)"
    )

    return valid_sequence_starts


def train_world_model(
    data,
    learning_rate=1e-3,
    num_epochs=100,
    batch_size=256,
    model_scale_factor=MODEL_SCALE_FACTOR,
    checkpoint_path="worldmodel_seaquest.pkl",
    save_every=10,
    rollout_steps=4,
    rollout_weight=0.0,  # Set > 0 to enable rollout loss (slower but better physics)
    use_deep=True,
    use_multistep=True,  # Use multi-step consistency loss (1-step + 2-step)
):
    """Train the MLP world model with life-aware batching and multi-step rollout loss."""

    print("Preparing training data...")
    print(f"Using {'deep' if use_deep else 'light'} model")
    print(f"Training mode: {'Multi-step (1+2)' if use_multistep else 'Single-step'}")

    obs = data["obs"]
    actions = data["actions"]
    next_obs = data["next_obs"]

    valid_indices = create_life_aware_batches(data)

    # For multi-step, also create 2-step sequences
    if use_multistep:
        sequence_indices = create_sequence_indices(data, sequence_length=2)

    # Compute normalization stats on valid data only
    valid_obs = obs[valid_indices]
    state_mean = jnp.mean(valid_obs, axis=0)
    state_std = jnp.std(valid_obs, axis=0) + 1e-8

    # Normalize ALL data (we'll index into it)
    obs_normalized = (obs - state_mean) / state_std
    next_obs_normalized = (next_obs - state_mean) / state_std

    # Initialize model
    model = create_world_model(model_scale_factor=MODEL_SCALE_FACTOR, use_deep=use_deep)

    rng = jax.random.PRNGKey(42)
    dummy_obs = obs_normalized[0]
    dummy_action = jnp.array([actions[0]])  # Model expects (batch,) shape

    # Load existing checkpoint if available, otherwise initialize new params
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint from {checkpoint_path}...")
        checkpoint_data = load_checkpoint(checkpoint_path)
        params = checkpoint_data.get("params", checkpoint_data.get("dynamics_params"))
        print(f"Loaded checkpoint from epoch {checkpoint_data.get('epoch', 'unknown')}")
    else:
        print("No existing checkpoint found, initializing new parameters...")
        params = model.init(rng, dummy_obs, dummy_action, None)

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {num_params:,}")

    # Optimizer - CONSTANT learning rate, NO weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(
            learning_rate=learning_rate
        ),  # Pure Adam, no schedule, no weight decay
    )
    opt_state = optimizer.init(params)

    # Feature weighting can be adapted for Seaquest-specific features
    # For now, use uniform weighting
    obs_dim = obs.shape[1]
    feature_weights = jnp.ones(obs_dim)

    # Simple single-step training with weighted loss
    @jax.jit
    def train_step_simple(params, opt_state, obs_batch, action_batch, target_batch):
        """
        Simple single-step prediction training.
        """

        def loss_fn(params):
            def single_forward(obs, action):
                pred, _ = model.apply(params, None, obs, jnp.array([action]), None)
                return pred.squeeze()

            # Vectorized prediction
            predictions = jax.vmap(single_forward)(obs_batch, action_batch)

            # Weighted MSE loss
            squared_errors = (predictions - target_batch) ** 2
            weighted_errors = squared_errors * feature_weights
            loss = jnp.mean(weighted_errors)

            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Multi-step consistency loss - trains on 1-step + 2-step predictions
    @jax.jit
    def train_step_multistep(
        params, opt_state, obs_batch, actions_batch, targets_batch
    ):
        """
        Multi-step consistency training.
        """

        def loss_fn(params):
            def single_forward(obs, action):
                pred, _ = model.apply(params, None, obs, jnp.array([action]), None)
                return pred.squeeze()

            def compute_multistep_loss(start_obs, actions, targets):
                # 1-step prediction (from ground truth)
                pred_1 = single_forward(start_obs, actions[0])
                loss_1step = jnp.mean(((pred_1 - targets[0]) ** 2) * feature_weights)

                # 2-step prediction (from model's 1-step prediction)
                pred_2 = single_forward(pred_1, actions[1])
                loss_2step = jnp.mean(((pred_2 - targets[1]) ** 2) * feature_weights)

                # Combine losses: full weight on 1-step, 0.5 weight on 2-step
                total_loss = loss_1step + 0.5 * loss_2step

                return total_loss, loss_1step

            # Vectorize over batch
            total_losses, step1_losses = jax.vmap(compute_multistep_loss)(
                obs_batch, actions_batch, targets_batch
            )

            avg_total = jnp.mean(total_losses)
            avg_step1 = jnp.mean(step1_losses)

            return avg_total, avg_step1

        (loss, step1_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, step1_loss

    # Training loop setup
    print(f"Starting training for {num_epochs} epochs...")

    if use_multistep:
        # Prepare 2-step sequences
        print("Preparing 2-step sequences for multi-step training...")
        train_obs = []
        train_actions = []
        train_targets = []
        for idx in sequence_indices:
            train_obs.append(obs_normalized[idx])
            train_actions.append(actions[idx : idx + 2])
            train_targets.append(next_obs_normalized[idx : idx + 2])
        train_obs = jnp.array(train_obs)
        train_actions = jnp.array(train_actions)
        train_targets = jnp.array(train_targets)
        print(f"Prepared {len(train_obs)} 2-step sequences")

        # JIT warmup
        print("Warming up JIT (multi-step)...")
        _ = train_step_multistep(
            params,
            opt_state,
            train_obs[:batch_size],
            train_actions[:batch_size],
            train_targets[:batch_size],
        )
        print("JIT warmup complete")
    else:
        # Use valid indices for single-step training
        train_obs = obs_normalized[valid_indices]
        train_actions = actions[valid_indices]
        train_targets = next_obs_normalized[valid_indices]

        # JIT warmup
        print("Warming up JIT (single-step)...")
        _ = train_step_simple(
            params,
            opt_state,
            train_obs[:batch_size],
            train_actions[:batch_size],
            train_targets[:batch_size],
        )
        print("JIT warmup complete")

    # Debug: compute approximate unnormalized MSE scale
    avg_std_sq = float(jnp.mean(state_std**2))
    print(f"DEBUG: avg(std^2) = {avg_std_sq:.2f}")
    print(f"Expected normalized loss if predicting mean: ~1.0")

    best_loss = float("inf")
    num_samples = len(train_obs)
    num_batches = num_samples // batch_size

    print(f"Training on {num_samples} samples, {num_batches} batches per epoch")

    for epoch in tqdm(range(num_epochs), desc="Training"):
        rng, shuffle_key = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_key, num_samples)

        epoch_losses = []
        epoch_step1_losses = []

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_perm = perm[start:end]

            if use_multistep:
                params, opt_state, loss, step1_loss = train_step_multistep(
                    params,
                    opt_state,
                    train_obs[batch_perm],
                    train_actions[batch_perm],
                    train_targets[batch_perm],
                )
                epoch_step1_losses.append(step1_loss)
            else:
                params, opt_state, loss = train_step_simple(
                    params,
                    opt_state,
                    train_obs[batch_perm],
                    train_actions[batch_perm],
                    train_targets[batch_perm],
                )
            epoch_losses.append(loss)

        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            if use_multistep:
                avg_step1 = float(jnp.mean(jnp.array(epoch_step1_losses)))
                print(
                    f"Epoch {epoch + 1}: TotalLoss={avg_loss:.6f}, Step1Loss={avg_step1:.6f}, Best={best_loss:.6f}"
                )
            else:
                print(f"Epoch {epoch + 1}: Loss={avg_loss:.6f}, Best={best_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                checkpoint_path,
                params,
                {"mean": state_mean, "std": state_std},
                epoch + 1,
                avg_loss,
                MODEL_SCALE_FACTOR,
                use_deep,
            )

    # Final save
    save_checkpoint(
        checkpoint_path,
        params,
        {"mean": state_mean, "std": state_std},
        num_epochs,
        best_loss,
        MODEL_SCALE_FACTOR,
        use_deep,
    )

    print(f"Training complete! Best loss: {best_loss:.6f}")
    if use_multistep:
        print(
            f"Multi-step training complete. Model trained on 1-step + 2-step predictions."
        )
    else:
        print(f"Single-step training complete. You can now test rollout performance.")
    return params, {"mean": state_mean, "std": state_std}


def save_checkpoint(
    path,
    params,
    normalization_stats,
    epoch,
    loss,
    model_scale_factor=MODEL_SCALE_FACTOR,
    use_deep=True,
):
    """Save model checkpoint."""
    model_type = "SeaquestMLPDeep" if use_deep else "SeaquestMLPLight"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "params": params,
                "dynamics_params": params,  # Alias for compatibility
                "normalization_stats": normalization_stats,
                "epoch": epoch,
                "loss": float(loss),
                "model_scale_factor": MODEL_SCALE_FACTOR,
                "model_type": model_type,
                "use_deep": use_deep,
            },
            f,
        )
    print(f"Saved checkpoint to {path} (epoch {epoch}, loss {loss:.6f})")


def load_checkpoint(path):
    """Load model checkpoint."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data




def create_env(frame_stack_size=4):
    """Create the Seaquest environment with wrappers."""
    game = JaxSeaquest()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    env = FlattenObservationWrapper(env)
    return env


def main():
    args = sys.argv[1:]

    if not args:
        print("Lightweight MLP World Model for Seaquest")
        print()
        print("Usage:")
        print(
            "  python worldmodel_seaquest.py collect [num_steps] [actor_type]  - Collect experience data"
        )
        print(
            "                                                     actor_type: 'real', 'imagined', or 'none' (default: none)"
        )
        print(
            "  python worldmodel_seaquest.py train [num_epochs]                - Train the world model"
        )
        print(
            "  python worldmodel_seaquest.py render [start_idx]                - Visualize predictions"
        )
        print()
        print("Examples:")
        print("  python worldmodel_seaquest.py collect 10000        # Collect 10k steps")
        print("  python worldmodel_seaquest.py train 100            # Train for 100 epochs")
        print("  python worldmodel_seaquest.py render 0             # Render from beginning")
        print()
        print("Files:")
        print("  experience_seaquest.pkl   - Collected experience data")
        print("  worldmodel_seaquest.pkl   - Trained model checkpoint")
        return

    command = args[0]
    frame_stack_size = 4

    if command == "collect":
        num_steps = int(args[1]) if len(args) > 1 else 100
        actor_type = (
            args[2] if len(args) > 2 else "none"
        )  # 'real', 'imagined', or 'none'

        env = create_env(frame_stack_size)

        # Load trained actor based on actor_type parameter
        actor_params = None
        actor_network = None

        if actor_type in ["real", "imagined"]:
            actor_path = f"{actor_type}_actor_params_seaquest.pkl"

            if os.path.exists(actor_path):
                print(f"Loading {actor_type} actor from {actor_path}...")
                try:
                    # Import actor creation function
                    sys.path.append(os.path.dirname(__file__))
                    from seaquest_agent import create_dreamerv2_actor

                    with open(actor_path, "rb") as f:
                        saved_data = pickle.load(f)
                        actor_params = saved_data.get("params", saved_data)

                    actor_network = create_dreamerv2_actor(action_dim=18)
                    print(
                        f"Successfully loaded {actor_type} actor for experience collection!"
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not load {actor_type} actor ({e}), using random policy"
                    )
                    actor_params = None
                    actor_network = None
            else:
                print(
                    f"No {actor_type} actor found at {actor_path}, using random policy"
                )
        else:
            print("Using random policy (no actor specified)")


        num_episodes = int(num_steps/1000) #simple heuristic to determine number of episodes


        data = collect_experience(
            env,
            num_episodes=num_episodes,
            frame_stack_size=frame_stack_size,
            actor_params=actor_params,
            actor_network=actor_network,
        )

        if len(data["obs"]) < num_steps:
            while True:
                #sample again
                print("Sampling more experience to reach desired number of steps...")
                data_extra = collect_experience(
                    env,
                    num_episodes=num_episodes,
                    frame_stack_size=frame_stack_size,
                    actor_params=actor_params,
                    actor_network=actor_network,
                )
                #concat onto data
                for key in data:
                    if key in ["episode_boundaries", "life_boundaries"]:
                        # These are lists, extend them
                        data[key].extend(data_extra[key])
                    else:
                        # These are arrays, concatenate them
                        data[key] = jnp.concatenate([data[key], data_extra[key]], axis=0)
                if len(data["obs"]) >= num_steps:
                    break

        print(f"Collected {len(data['obs'])} total transitions cutting to {num_steps}...")
        #cut data to exactly num_steps
        for key in data:
            data[key] = data[key][:num_steps]

        save_path = "experience_seaquest.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved experience to {save_path}")

    elif command == "train":
        num_epochs = int(args[1]) if len(args) > 1 else 100

        # Load experience
        experience_path = "experience_seaquest.pkl"
        if not os.path.exists(experience_path):
            print(f"No experience data found at {experience_path}")
            print("Run 'python worldmodel_seaquest.py collect' first")
            return

        with open(experience_path, "rb") as f:
            data = pickle.load(f)

        print(f"Loaded {len(data['obs'])} samples")
        print(f"Observation shape: {data['obs'].shape}")

        params, norm_stats = train_world_model(
            data,
            num_epochs=num_epochs,
            rollout_weight=0.0,
            model_scale_factor=MODEL_SCALE_FACTOR,
            learning_rate=3e-4,
            batch_size=512,
            checkpoint_path="worldmodel_seaquest.pkl",
            use_multistep=False,
        )

    elif command == "render":
        start_idx = int(args[1]) if len(args) > 1 else 0

        # Load model
        checkpoint_path = "worldmodel_seaquest.pkl"
        if not os.path.exists(checkpoint_path):
            print(f"No model found at {checkpoint_path}")
            print("Run 'python worldmodel_seaquest.py train' first")
            return

        checkpoint = load_checkpoint(checkpoint_path)
        norm_stats = checkpoint["normalization_stats"]

        # Generate FRESH test episode to check for overfitting
        print("\n" + "=" * 60)
        print("GENERATING FRESH TEST EPISODE (not from training data)")
        print("This tests if the model generalizes vs. overfits to training data")
        print("=" * 60 + "\n")

        env = create_env(frame_stack_size)

        # Collect 1 fresh episode with random policy
        test_data = collect_experience(
            env,
            num_episodes=1,
            max_steps_per_episode=1000,
            frame_stack_size=frame_stack_size,
            exploration_rate=0.5,
            seed=int(time.time()),  # Different seed for fresh data
        )

        # Diagnostic: Check data format
        obs = test_data["obs"]
        print(f"\n=== Fresh Test Episode Diagnostics ===")
        print(f"Obs shape: {obs.shape}")
        print(f"Actions shape: {test_data['actions'].shape}")
        print(f"Episode boundaries: {test_data['episode_boundaries']}")
        print(f"Total transitions: {len(obs)}")

        compare_real_vs_model(
            num_steps=50000,
            render_scale=6,
            obs=test_data["obs"],
            actions=test_data["actions"],
            normalization_stats=norm_stats,
            boundaries=test_data["episode_boundaries"],
            env=env,
            starting_step=min(start_idx, len(obs) - 1),
            steps_into_future=10,
            frame_stack_size=frame_stack_size,
            model_path=checkpoint_path,
            clock_speed=100,
            print_error=False,  # Set to False for faster rendering
        )

    else:
        print(f"Unknown command: {command}")
        print("Use: collect, train, or render")


if __name__ == "__main__":
    main()
