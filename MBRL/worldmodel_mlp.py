"""
Lightweight MLP World Model for Pong

A simple, fast world model using:
- Frame stacking (4 frames) for temporal information
- Simple MLP architecture (no recurrent state)
- Life-aware batching (sequences don't cross ball respawns)
- Full JAX JIT compilation for speed
"""

import jax
import jax.numpy as jnp
import optax
import pickle
import os
import sys
from tqdm import tqdm
from typing import Tuple, Any

# Import from jaxatari for environment
from jaxatari.games.jax_pong import JaxPong
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper

# Import from existing codebase
from model_architectures import PongMLPDeep, PongMLPLight


MODEL_SCALE_FACTOR = 1  # Keep at 1 for speed

# ============================================================================
# MLP World Model
# ============================================================================

def create_world_model(model_scale_factor=MODEL_SCALE_FACTOR, use_deep=True):
    """
    Create the world model.
    use_deep=True uses PongMLPDeep (4 layers with LayerNorm and residual connections)
    use_deep=False uses PongMLPLight (2 layers, faster but less expressive)
    """
    if use_deep:
        return PongMLPDeep(model_scale_factor)
    return PongMLPLight(model_scale_factor)


# ============================================================================
# Helper Functions
# ============================================================================

def flatten_obs(state, single_state: bool = False) -> Tuple[jnp.ndarray, Any]:
    """Flatten the state PyTree into a single array.

    Output format is INTERLEAVED: [feat0_f0, feat0_f1, ..., feat0_f3, feat1_f0, ...]
    This matches the format from jax.flatten_util.ravel_pytree on single observations.
    """
    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener

    # Get all leaves - each has shape (..., 4) where 4 is frame stack
    # Concatenate along last axis to get (..., 56)
    leaves = jax.tree_util.tree_leaves(state)
    flat_state = jnp.concatenate(leaves, axis=-1)

    return flat_state, None


def detect_life_boundaries_vectorized(obs, next_obs, frame_stack_size=4):
    """
    Detect scoring events in a batch of transitions (vectorized).

    Data is INTERLEAVED format: [feat0_f0, feat0_f1, ..., feat0_f3, feat1_f0, ...]
    For feature i at frame f: index = i * frame_stack_size + f

    Args:
        obs: (N, 56) current observations
        next_obs: (N, 56) next observations

    Returns:
        Boolean array of shape (N,) indicating score changes
    """
    # Feature indices (0-based): score_player=12, score_enemy=13
    # Last frame index: feature_idx * 4 + 3
    score_player_idx = 12 * frame_stack_size + (frame_stack_size - 1)  # = 51
    score_enemy_idx = 13 * frame_stack_size + (frame_stack_size - 1)   # = 55

    score_changed = (
        (obs[:, score_player_idx] != next_obs[:, score_player_idx]) |
        (obs[:, score_enemy_idx] != next_obs[:, score_enemy_idx])
    )

    return score_changed


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
):
    """
    Collect experience data using JAX vmap and scan for speed.

    Uses a ball-tracking policy with exploration.
    """
    rng = jax.random.PRNGKey(seed)

    def perfect_policy(obs, rng):
        """Ball tracking policy with exploration."""
        rng, action_key = jax.random.split(rng)
        do_random = jax.random.uniform(action_key) < exploration_rate
        random_action = jax.random.randint(action_key, (), 0, 6)

        # Ball tracking: compare ball y with player y (from last frame)
        perfect_action = jax.lax.cond(
            obs.player.y[frame_stack_size - 1] > obs.ball.y[frame_stack_size - 1],
            lambda _: jnp.array(4),  # Up
            lambda _: jax.lax.cond(
                obs.player.y[frame_stack_size - 1] < obs.ball.y[frame_stack_size - 1],
                lambda _: jnp.array(3),  # Down
                lambda _: jnp.array(0),  # Noop
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

            def continue_step(_):
                rng_new, _ = jax.random.split(rng)
                action = perfect_policy(obs, rng_new)

                next_obs, next_state, reward, next_done, _ = env.step(state, action)
                transition = (obs, action, jnp.float32(reward), next_done, ~done)

                return (rng_new, next_obs, next_state, next_done), transition

            def skip_step(_):
                dummy_action = jnp.array(0, dtype=jnp.int32)
                dummy_reward = jnp.array(0.0, dtype=jnp.float32)
                dummy_done = jnp.array(False, dtype=jnp.bool_)
                dummy_valid = jnp.array(False, dtype=jnp.bool_)
                dummy_transition = (obs, dummy_action, dummy_reward, dummy_done, dummy_valid)
                return (rng, obs, state, done), dummy_transition

            return jax.lax.cond(done, skip_step, continue_step, None)

        initial_carry = (step_key, obs, state, jnp.array(False))
        _, transitions = jax.lax.scan(step_fn, initial_carry, None, length=max_steps_per_episode)

        observations, actions, rewards, dones, valid_mask = transitions
        episode_length = jnp.sum(valid_mask)

        return observations, actions, rewards, dones, valid_mask, episode_length

    # Generate episode keys and run in parallel
    episode_keys = jax.random.split(rng, num_episodes)

    print(f"Collecting {num_episodes} episodes with vmap + scan...")
    vmapped_episode_fn = jax.vmap(run_single_episode)
    observations, actions, rewards, _, _, episode_lengths = vmapped_episode_fn(episode_keys)

    print("Processing collected data...")

    # Flatten all observations at once (vectorized)
    flat_obs_all, _ = flatten_obs(observations)  # (num_episodes, max_steps, 56)

    # Process episodes
    all_obs = []
    all_actions = []
    all_next_obs = []
    all_rewards = []
    episode_boundaries = []
    life_boundaries = []
    cumulative_steps = 0

    for ep_idx in range(num_episodes):
        ep_length = int(episode_lengths[ep_idx])

        if ep_length > 1:
            flat_obs = flat_obs_all[ep_idx, :ep_length]
            valid_actions = actions[ep_idx, :ep_length]
            valid_rewards = rewards[ep_idx, :ep_length]

            # obs[i] -> next_obs[i] = obs[i+1]
            obs_slice = flat_obs[:-1]
            next_obs_slice = flat_obs[1:]
            actions_slice = valid_actions[:-1]
            rewards_slice = valid_rewards[:-1]

            # Vectorized life boundary detection (INTERLEAVED format)
            # For feature i at last frame: index = i * frame_stack_size + (frame_stack_size - 1)
            score_player_idx = 12 * frame_stack_size + (frame_stack_size - 1)  # = 51
            score_enemy_idx = 13 * frame_stack_size + (frame_stack_size - 1)   # = 55

            score_changes = (
                (obs_slice[:, score_player_idx] != next_obs_slice[:, score_player_idx]) |
                (obs_slice[:, score_enemy_idx] != next_obs_slice[:, score_enemy_idx])
            )
            boundary_indices = jnp.where(score_changes)[0] + cumulative_steps
            life_boundaries.extend(boundary_indices.tolist())

            all_obs.append(obs_slice)
            all_actions.append(actions_slice)
            all_next_obs.append(next_obs_slice)
            all_rewards.append(rewards_slice)

            cumulative_steps += ep_length - 1
            episode_boundaries.append(cumulative_steps)

    # Concatenate all episodes
    all_obs = jnp.concatenate(all_obs, axis=0)
    all_actions = jnp.concatenate(all_actions, axis=0)
    all_next_obs = jnp.concatenate(all_next_obs, axis=0)
    all_rewards = jnp.concatenate(all_rewards, axis=0)

    print(f"Collected {len(all_obs)} total transitions")
    print(f"Found {len(life_boundaries)} scoring events (life boundaries)")
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
    print(f"Valid training indices: {len(valid_indices)} / {len(obs)} "
          f"({100 * len(valid_indices) / len(obs):.1f}%)")

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
    print(f"Valid sequence starts (len={sequence_length}): {len(valid_sequence_starts)} / {len(obs)} "
          f"({100 * len(valid_sequence_starts) / len(obs):.1f}%)")

    return valid_sequence_starts


def train_world_model(
    data,
    learning_rate=1e-3,
    num_epochs=100,
    batch_size=256,
    model_scale_factor=MODEL_SCALE_FACTOR,
    checkpoint_path="worldmodel_mlp.pkl",
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

    params = model.init(rng, dummy_obs, dummy_action, None)

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {num_params:,}")

    # Optimizer - CONSTANT learning rate, NO weight decay
    # Previous schedule/decay was causing plateau - model couldn't escape local minimum
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate),  # Pure Adam, no schedule, no weight decay
    )
    opt_state = optimizer.init(params)


    # Ball-specific loss weighting - ball position errors compound fastest
    # INTERLEAVED format: for feature i, frame f: index = i * 4 + f
    # Ball features: ball_x (feat 8), ball_y (feat 9)
    # All frames for ball_x: [32, 33, 34, 35], ball_y: [36, 37, 38, 39]
    feature_weights = jnp.ones(56)
    # INCREASED: Weight ball position features MUCH higher (10x weight instead of 3x)
    # This forces the model to prioritize ball physics accuracy
    ball_x_indices = jnp.array([32, 33, 34, 35])
    ball_y_indices = jnp.array([36, 37, 38, 39])
    feature_weights = feature_weights.at[ball_x_indices].set(10.0)
    feature_weights = feature_weights.at[ball_y_indices].set(10.0)

    # Ball velocity features are also critical (5x weight instead of 2x)
    # ball_x_direction (feat 4): [16, 17, 18, 19]
    # ball_y_direction (feat 5): [20, 21, 22, 23]
    ball_vx_indices = jnp.array([16, 17, 18, 19])
    ball_vy_indices = jnp.array([20, 21, 22, 23])
    feature_weights = feature_weights.at[ball_vx_indices].set(5.0)
    feature_weights = feature_weights.at[ball_vy_indices].set(5.0)

    # Player position also matters (2x weight)
    player_y_indices = jnp.array([4, 5, 6, 7])  # player_y all frames
    feature_weights = feature_weights.at[player_y_indices].set(2.0)

    # Simple single-step training with ball-weighted loss
    @jax.jit
    def train_step_simple(params, opt_state, obs_batch, action_batch, target_batch):
        """
        Simple single-step prediction training.
        obs_batch: (batch, 56) observations
        action_batch: (batch,) actions
        target_batch: (batch, 56) target next observations
        """
        def loss_fn(params):
            def single_forward(obs, action):
                pred, _ = model.apply(params, None, obs, jnp.array([action]), None)
                return pred.squeeze()

            # Vectorized prediction
            predictions = jax.vmap(single_forward)(obs_batch, action_batch)

            # Weighted MSE loss - emphasize ball position/velocity errors
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
    def train_step_multistep(params, opt_state, obs_batch, actions_batch, targets_batch):
        """
        Multi-step consistency training.
        obs_batch: (batch, 56) starting observations
        actions_batch: (batch, 2) actions for step 1 and 2
        targets_batch: (batch, 2, 56) targets for step 1 and 2
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
                # Increased from 0.3 to put more emphasis on rollout consistency
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
            train_actions.append(actions[idx:idx + 2])
            train_targets.append(next_obs_normalized[idx:idx + 2])
        train_obs = jnp.array(train_obs)
        train_actions = jnp.array(train_actions)
        train_targets = jnp.array(train_targets)
        print(f"Prepared {len(train_obs)} 2-step sequences")

        # JIT warmup
        print("Warming up JIT (multi-step)...")
        _ = train_step_multistep(params, opt_state,
                                 train_obs[:batch_size],
                                 train_actions[:batch_size],
                                 train_targets[:batch_size])
        print("JIT warmup complete")
    else:
        # Use valid indices for single-step training
        train_obs = obs_normalized[valid_indices]
        train_actions = actions[valid_indices]
        train_targets = next_obs_normalized[valid_indices]

        # JIT warmup
        print("Warming up JIT (single-step)...")
        _ = train_step_simple(params, opt_state,
                             train_obs[:batch_size],
                             train_actions[:batch_size],
                             train_targets[:batch_size])
        print("JIT warmup complete")

    # Debug: compute approximate unnormalized MSE scale
    avg_std_sq = float(jnp.mean(state_std ** 2))
    print(f"DEBUG: avg(std^2) = {avg_std_sq:.2f}")
    print(f"Expected normalized loss if predicting mean: ~1.0")

    best_loss = float('inf')
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
                    params, opt_state,
                    train_obs[batch_perm],
                    train_actions[batch_perm],
                    train_targets[batch_perm]
                )
                epoch_step1_losses.append(step1_loss)
            else:
                params, opt_state, loss = train_step_simple(
                    params, opt_state,
                    train_obs[batch_perm],
                    train_actions[batch_perm],
                    train_targets[batch_perm]
                )
            epoch_losses.append(loss)

        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            if use_multistep:
                avg_step1 = float(jnp.mean(jnp.array(epoch_step1_losses)))
                print(f"Epoch {epoch + 1}: TotalLoss={avg_loss:.6f}, Step1Loss={avg_step1:.6f}, Best={best_loss:.6f}")
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
                model_scale_factor,
                use_deep,
            )

    # Final save
    save_checkpoint(
        checkpoint_path,
        params,
        {"mean": state_mean, "std": state_std},
        num_epochs,
        best_loss,
        model_scale_factor,
        use_deep,
    )

    print(f"Training complete! Best loss: {best_loss:.6f}")
    if use_multistep:
        print(f"Multi-step training complete. Model trained on 1-step + 2-step predictions.")
    else:
        print(f"Single-step training complete. You can now test rollout performance.")
    print(f"Improvements applied: ball-weighted loss, residual=0.2, {'multi-step' if use_multistep else 'single-step'}")
    return params, {"mean": state_mean, "std": state_std}


def save_checkpoint(path, params, normalization_stats, epoch, loss, model_scale_factor=MODEL_SCALE_FACTOR, use_deep=True):
    """Save model checkpoint (compatible with pong_agent.py)."""
    model_type = "PongMLPDeep" if use_deep else "PongMLPLight"
    with open(path, "wb") as f:
        pickle.dump({
            "params": params,
            "dynamics_params": params,  # Alias for compatibility
            "normalization_stats": normalization_stats,
            "epoch": epoch,
            "loss": float(loss),
            "model_scale_factor": model_scale_factor,
            "model_type": model_type,
            "use_deep": use_deep,
        }, f)
    print(f"Saved checkpoint to {path} (epoch {epoch}, loss {loss:.6f})")


def load_checkpoint(path):
    """Load model checkpoint."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# ============================================================================
# Visualization - uses compare_real_vs_model from worldmodelPong
# ============================================================================

import worldmodelPong
from worldmodelPong import compare_real_vs_model


def set_model_architecture(use_deep=True):
    """Set the MODEL_ARCHITECTURE in worldmodelPong based on checkpoint."""
    if use_deep:
        worldmodelPong.MODEL_ARCHITECTURE = PongMLPDeep
    else:
        worldmodelPong.MODEL_ARCHITECTURE = PongMLPLight


# ============================================================================
# Main CLI
# ============================================================================

def create_env(frame_stack_size=4):
    """Create the Pong environment with wrappers."""
    game = JaxPong()
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
        print("Lightweight MLP World Model for Pong")
        print()
        print("Usage:")
        print("  python worldmodel_mlp.py collect [num_episodes]  - Collect experience data")
        print("  python worldmodel_mlp.py train [num_epochs]      - Train the world model")
        print("  python worldmodel_mlp.py render [start_idx]      - Visualize predictions")
        print()
        print("Files:")
        print("  experience_mlp.pkl   - Collected experience data")
        print("  worldmodel_mlp.pkl   - Trained model checkpoint")
        return

    command = args[0]
    frame_stack_size = 4

    if command == "collect":
        num_episodes = int(args[1]) if len(args) > 1 else 100

        env = create_env(frame_stack_size)
        data = collect_experience(
            env,
            num_episodes=num_episodes,
            frame_stack_size=frame_stack_size,
        )

        save_path = "experience_mlp.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved experience to {save_path}")

    elif command == "train":
        num_epochs = int(args[1]) if len(args) > 1 else 100

        # Load experience
        experience_path = "experience_mlp.pkl"
        if not os.path.exists(experience_path):
            print(f"No experience data found at {experience_path}")
            print("Run 'python worldmodel_mlp.py collect' first")
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
            learning_rate=3e-4,  # Reduced from 1e-3 - was too high, causing early plateau
            batch_size=512,  # Increased from 256 for more stable gradients
            checkpoint_path="worldmodel_mlp.pkl",
            use_multistep=False,  # Disabled - preparation too slow
        )

    elif command == "render":
        start_idx = int(args[1]) if len(args) > 1 else 0

        # Load model
        checkpoint_path = "worldmodel_mlp.pkl"
        if not os.path.exists(checkpoint_path):
            print(f"No model found at {checkpoint_path}")
            print("Run 'python worldmodel_mlp.py train' first")
            return

        checkpoint = load_checkpoint(checkpoint_path)
        norm_stats = checkpoint["normalization_stats"]
        model_scale_factor = checkpoint.get("model_scale_factor", 1)
        use_deep = checkpoint.get("use_deep", False)  # Default False for old checkpoints
        set_model_architecture(use_deep)
        print(f"Using {'deep' if use_deep else 'light'} model from checkpoint")
        # Note: params loaded from checkpoint by compare_real_vs_model via model_path

        # Load experience for rendering
        experience_path = "experience_mlp.pkl"
        if not os.path.exists(experience_path):
            print(f"No experience data found at {experience_path}")
            return

        print(f"Loading from: {os.path.abspath(experience_path)}")
        print(f"File mtime: {os.path.getmtime(experience_path)}")

        with open(experience_path, "rb") as f:
            data = pickle.load(f)

        # Diagnostic: Check data format
        obs = data["obs"]
        print(f"\n=== Data Diagnostics ===")
        print(f"Obs shape: {obs.shape}")
        print(f"Actions shape: {data['actions'].shape}")
        print(f"Episode boundaries: {data['episode_boundaries'][:5]}...")

        # Check first observation values
        # INTERLEAVED format: for feature i, frame f: index = i * 4 + f
        sample_obs = obs[start_idx]
        print(f"\nSample obs at idx {start_idx} (INTERLEAVED format):")
        print(f"  Frame stacking: {frame_stack_size} frames x 14 features = {frame_stack_size * 14}")
        for frame in range(frame_stack_size):
            # Feature indices: player_y=1, ball_x=8, ball_y=9, score_player=12, score_enemy=13
            player_y_idx = 1 * frame_stack_size + frame
            ball_x_idx = 8 * frame_stack_size + frame
            ball_y_idx = 9 * frame_stack_size + frame
            score_p_idx = 12 * frame_stack_size + frame
            score_e_idx = 13 * frame_stack_size + frame
            print(f"  Frame {frame}: player_y={sample_obs[player_y_idx]:.1f}, "
                  f"ball_x={sample_obs[ball_x_idx]:.1f}, ball_y={sample_obs[ball_y_idx]:.1f}, "
                  f"score_p={sample_obs[score_p_idx]:.0f}, score_e={sample_obs[score_e_idx]:.0f}")

        print(f"\nFormat: [feat0_f0..f3, feat1_f0..f3, ...] - last frame uses idx i*4+3")
        print("=" * 30)

        env = create_env(frame_stack_size)

        # Debug: verify data before passing
        print("\n=== DEBUG: Data being passed to compare_real_vs_model ===")
        print(f"obs type: {type(data['obs'])}")
        print(f"obs shape: {data['obs'].shape}")
        print(f"obs[10000]: {data['obs'][10000][:20]}...")
        print("=" * 50)

        compare_real_vs_model(
            num_steps=500,
            render_scale=6,
            obs=data["obs"],
            actions=data["actions"],
            normalization_stats=norm_stats,
            boundaries=data["episode_boundaries"],
            env=env,
            starting_step=start_idx,
            steps_into_future=10,
            frame_stack_size=frame_stack_size,
            model_scale_factor=MODEL_SCALE_FACTOR,
            model_path=checkpoint_path,
        )

    else:
        print(f"Unknown command: {command}")
        print("Use: collect, train, or render")


if __name__ == "__main__":
    main()
