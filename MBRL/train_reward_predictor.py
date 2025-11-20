"""
Standalone reward predictor training script.
Trains the reward predictor to predict score-based rewards from position features only.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
from tqdm import tqdm
from model_architectures import RewardPredictorMLPPositionOnly


def calculate_score_based_reward(flat_obs, next_flat_obs):
    """
    Calculate reward based on score difference between observations.
    Returns +1 if player scored, -1 if enemy scored, 0 otherwise.
    """
    # flat_obs[-5] is player score, flat_obs[-1] is enemy score
    player_score_old = flat_obs[..., -5]
    enemy_score_old = flat_obs[..., -1]
    player_score_new = next_flat_obs[..., -5]
    enemy_score_new = next_flat_obs[..., -1]

    # Calculate score changes
    player_scored = player_score_new - player_score_old
    enemy_scored = enemy_score_new - enemy_score_old

    # Reward is +1 if player scored, -1 if enemy scored, 0 otherwise
    score_reward = player_scored - enemy_scored

    # Filter out rewards where abs > 1 (safety check)
    score_reward = jnp.where(jnp.abs(score_reward) > 1, 0.0, score_reward)

    return score_reward


def calculate_position_based_reward(flat_obs, next_flat_obs, frame_stack_size=4):
    """
    Calculate reward based on ball reaching extreme positions.

    Ball x ranges 0-16. We assign reward when ball is at edges:
    - Ball at very high x (>12): ball near/past enemy paddle → +1
    - Ball at very low x (0): ball near/past player paddle → -1

    This ensures model learns to associate extreme positions with scoring,
    and will never predict scores when ball is in the middle.
    """
    # Get ball_x from last frame
    base = (frame_stack_size - 1) * 14
    ball_x_curr = flat_obs[..., base + 8]
    ball_x_next = next_flat_obs[..., base + 8]

    # Calculate the change in ball position
    ball_x_delta = ball_x_next - ball_x_curr

    # Detect scoring based on large backward jumps (resets)
    # All scoring events cause ball to reset with a large negative delta
    is_reset = ball_x_delta < -3.0

    # Classify by where ball was before reset
    # Ball at high x (>6) before reset: ball was on far side → player scored
    # Ball at moderate-low x (2-6) before reset: ball was on near side → enemy scored
    player_scored = is_reset & (ball_x_curr > 6.0)
    enemy_scored = is_reset & (ball_x_curr >= 2.0) & (ball_x_curr <= 6.0)

    # Convert to reward
    reward = jnp.where(player_scored, 1.0,
                       jnp.where(enemy_scored, -1.0, 0.0))

    return reward


def calculate_edge_crossing_reward(flat_obs, next_flat_obs, frame_stack_size=4):
    """
    Calculate reward based on ball being at extreme positions.

    This labels frames where the ball is at the edge of the field,
    indicating a score just happened or is about to happen.

    Ball x ranges 0-16:
    - x near 0: player's side (left)
    - x near 16: enemy's side (right)

    Scoring detection (simple position thresholds):
    - Ball at very high x (>14): player scored (+1)
    - Ball at very low x (<1): enemy scored (-1)

    This ensures the model only predicts scores when ball is at extreme edges,
    never when ball is in the middle (hitting paddle).
    """
    # Get ball_x from last frame
    base = (frame_stack_size - 1) * 14
    ball_x_curr = flat_obs[..., base + 8]

    # Simple position-based detection
    # Player scores when ball reaches enemy edge
    player_scored = ball_x_curr > 14.0

    # Enemy scores when ball reaches player edge
    enemy_scored = ball_x_curr < 1.0

    # Convert to reward
    reward = jnp.where(player_scored, 1.0,
                       jnp.where(enemy_scored, -1.0, 0.0))

    return reward


def load_experience_data(num_files=5):
    """Load experience data from pickle files."""
    all_obs = []
    all_next_obs = []

    for i in range(num_files):
        experience_path = f"experience_data_LSTM_pong_{i}.pkl"
        if not os.path.exists(experience_path):
            print(f"Warning: {experience_path} not found, skipping")
            continue

        print(f"Loading {experience_path}...")
        with open(experience_path, "rb") as f:
            data = pickle.load(f)
            all_obs.append(data["obs"])
            all_next_obs.append(data["next_obs"])

    # Concatenate all data and convert to float32
    obs = jnp.concatenate(all_obs, axis=0).astype(jnp.float32)
    next_obs = jnp.concatenate(all_next_obs, axis=0).astype(jnp.float32)

    print(f"Loaded {len(obs)} transitions")
    return obs, next_obs


def create_training_data(obs, next_obs, frame_stack_size=4):
    """
    Create input/output arrays for reward predictor training.

    Input: (current_obs, next_obs) pairs
    Output: position-based reward (-1, 0, or +1) based on ball crossing edges
    """
    # Debug: find actual ball_x range in data
    base = (frame_stack_size - 1) * 14
    all_ball_x = jnp.concatenate([obs[:, base + 8], next_obs[:, base + 8]])
    print(f"\nBall X statistics across all data:")
    print(f"  Min: {jnp.min(all_ball_x):.1f}")
    print(f"  Max: {jnp.max(all_ball_x):.1f}")
    print(f"  Mean: {jnp.mean(all_ball_x):.1f}")

    # Check score-based rewards to understand actual scoring events
    score_rewards = calculate_score_based_reward(obs, next_obs)
    score_pos_mask = score_rewards > 0
    score_neg_mask = score_rewards < 0

    print(f"\nScore-based reward analysis:")
    print(f"  +1 events (player scored): {jnp.sum(score_pos_mask)}")
    print(f"  -1 events (enemy scored): {jnp.sum(score_neg_mask)}")

    ball_x_curr = obs[:, base + 8]
    ball_x_next = next_obs[:, base + 8]

    if jnp.sum(score_pos_mask) > 0:
        print(f"\nWhen player scores (+1):")
        print(f"  Current ball_x: min={jnp.min(ball_x_curr[score_pos_mask]):.1f}, max={jnp.max(ball_x_curr[score_pos_mask]):.1f}, mean={jnp.mean(ball_x_curr[score_pos_mask]):.1f}")
        print(f"  Next ball_x: min={jnp.min(ball_x_next[score_pos_mask]):.1f}, max={jnp.max(ball_x_next[score_pos_mask]):.1f}, mean={jnp.mean(ball_x_next[score_pos_mask]):.1f}")

    if jnp.sum(score_neg_mask) > 0:
        print(f"\nWhen enemy scores (-1):")
        print(f"  Current ball_x: min={jnp.min(ball_x_curr[score_neg_mask]):.1f}, max={jnp.max(ball_x_curr[score_neg_mask]):.1f}, mean={jnp.mean(ball_x_curr[score_neg_mask]):.1f}")
        print(f"  Next ball_x: min={jnp.min(ball_x_next[score_neg_mask]):.1f}, max={jnp.max(ball_x_next[score_neg_mask]):.1f}, mean={jnp.mean(ball_x_next[score_neg_mask]):.1f}")

    # Analyze ball_x jumps (resets)
    ball_x_delta = ball_x_next - ball_x_curr
    print(f"\nBall X delta (next - curr) statistics:")
    print(f"  Min: {jnp.min(ball_x_delta):.1f}")
    print(f"  Max: {jnp.max(ball_x_delta):.1f}")

    # Find large jumps
    large_neg = ball_x_delta < -5
    large_pos = ball_x_delta > 5
    print(f"  Large negative jumps (<-5): {jnp.sum(large_neg)}")
    print(f"  Large positive jumps (>5): {jnp.sum(large_pos)}")

    # Check transitions to 0
    reset_to_zero = ball_x_next == 0
    print(f"\nTransitions to ball_x=0: {jnp.sum(reset_to_zero)}")
    if jnp.sum(reset_to_zero) > 0:
        reset_curr = ball_x_curr[reset_to_zero]
        print(f"  Prior ball_x: min={jnp.min(reset_curr):.1f}, max={jnp.max(reset_curr):.1f}, mean={jnp.mean(reset_curr):.1f}")

    # Check transitions to high values
    reset_to_high = ball_x_next > 14
    print(f"Transitions to ball_x>14: {jnp.sum(reset_to_high)}")
    if jnp.sum(reset_to_high) > 0:
        high_curr = ball_x_curr[reset_to_high]
        print(f"  Prior ball_x: min={jnp.min(high_curr):.1f}, max={jnp.max(high_curr):.1f}, mean={jnp.mean(high_curr):.1f}")

    # Calculate rewards using score-based detection
    # This correctly identifies scoring events by actual score changes
    # (edge-crossing detection gives false positives when ball_x=0 during game start/reset)
    rewards = calculate_score_based_reward(obs, next_obs)

    # Compare with score-based for diagnostics
    score_rewards = calculate_score_based_reward(obs, next_obs)
    edge_pos = jnp.sum(rewards > 0)
    edge_neg = jnp.sum(rewards < 0)
    score_pos = jnp.sum(score_rewards > 0)
    score_neg = jnp.sum(score_rewards < 0)

    print(f"\nEdge-crossing vs Score-based comparison:")
    print(f"  Edge +1 events: {edge_pos}, Score +1 events: {score_pos}")
    print(f"  Edge -1 events: {edge_neg}, Score -1 events: {score_neg}")

    # Count reward distribution
    num_positive = jnp.sum(rewards > 0)
    num_negative = jnp.sum(rewards < 0)
    num_zero = jnp.sum(rewards == 0)

    print(f"Reward distribution (score-based):")
    print(f"  +1 (player scored): {num_positive}")
    print(f"  -1 (enemy scored): {num_negative}")
    print(f"   0 (no score): {num_zero}")

    # Debug: show ball positions for +1 vs -1 cases
    base = (frame_stack_size - 1) * 14  # Last frame

    pos_mask = rewards > 0
    neg_mask = rewards < 0

    if jnp.sum(pos_mask) > 0:
        pos_ball_x_curr = obs[pos_mask, base + 8]
        pos_ball_x_next = next_obs[pos_mask, base + 8]
        print(f"\n+1 cases (player scored):")
        print(f"  Current ball_x: min={jnp.min(pos_ball_x_curr):.1f}, max={jnp.max(pos_ball_x_curr):.1f}, mean={jnp.mean(pos_ball_x_curr):.1f}")
        print(f"  Next ball_x: min={jnp.min(pos_ball_x_next):.1f}, max={jnp.max(pos_ball_x_next):.1f}, mean={jnp.mean(pos_ball_x_next):.1f}")

    if jnp.sum(neg_mask) > 0:
        neg_ball_x_curr = obs[neg_mask, base + 8]
        neg_ball_x_next = next_obs[neg_mask, base + 8]
        print(f"\n-1 cases (enemy scored):")
        print(f"  Current ball_x: min={jnp.min(neg_ball_x_curr):.1f}, max={jnp.max(neg_ball_x_curr):.1f}, mean={jnp.mean(neg_ball_x_curr):.1f}")
        print(f"  Next ball_x: min={jnp.min(neg_ball_x_next):.1f}, max={jnp.max(neg_ball_x_next):.1f}, mean={jnp.mean(neg_ball_x_next):.1f}")

    return obs, next_obs, rewards


def train_reward_predictor(
    obs,
    next_obs,
    rewards,
    learning_rate=1e-3,
    batch_size=256,
    num_epochs=1000,
    frame_stack_size=4,
    save_path="reward_predictor_standalone.pkl",
):
    """Train the reward predictor."""

    # Create model
    reward_model = RewardPredictorMLPPositionOnly(model_scale_factor=1, frame_stack_size=frame_stack_size)

    # Initialize
    rng = jax.random.PRNGKey(42)
    dummy_obs = obs[:1]
    dummy_next_obs = next_obs[:1]
    dummy_action = jnp.zeros(1, dtype=jnp.int32)  # Not used but needed for signature

    params = reward_model.init(rng, dummy_obs, dummy_action, dummy_next_obs)

    # Optimizer with learning rate schedule
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=num_epochs,
        alpha=0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(params)

    # Split data 80/20
    num_samples = len(obs)
    num_train = int(0.8 * num_samples)

    # Shuffle indices
    shuffle_key = jax.random.PRNGKey(123)
    indices = jax.random.permutation(shuffle_key, num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_obs = obs[train_indices]
    train_next_obs = next_obs[train_indices]
    train_rewards = rewards[train_indices]

    val_obs = obs[val_indices]
    val_next_obs = next_obs[val_indices]
    val_rewards = rewards[val_indices]

    print(f"Training samples: {len(train_obs)}")
    print(f"Validation samples: {len(val_obs)}")

    # Compute class weights based on frequency (inverse frequency weighting)
    num_pos = jnp.sum(train_rewards > 0)
    num_neg = jnp.sum(train_rewards < 0)
    num_zero = jnp.sum(train_rewards == 0)
    total = len(train_rewards)

    # Weight = total / (3 * class_count) to balance classes
    weight_pos = total / (3 * jnp.maximum(num_pos, 1))
    weight_neg = total / (3 * jnp.maximum(num_neg, 1))
    weight_zero = total / (3 * jnp.maximum(num_zero, 1))

    print(f"Class weights: +1={weight_pos:.1f}, -1={weight_neg:.1f}, 0={weight_zero:.1f}")

    # Training step with weighted loss
    @jax.jit
    def train_step(params, opt_state, batch_obs, batch_next_obs, batch_rewards):
        def loss_fn(p):
            # Dummy action (not used in position-only model after user's edit)
            dummy_actions = jnp.zeros(len(batch_obs), dtype=jnp.int32)

            predicted = reward_model.apply(p, rng, batch_obs, dummy_actions, batch_next_obs)

            # Weighted MSE loss to handle class imbalance
            errors = (predicted - batch_rewards) ** 2

            # Assign weights based on true reward class
            weights = jnp.where(batch_rewards > 0, weight_pos,
                               jnp.where(batch_rewards < 0, weight_neg, weight_zero))

            loss = jnp.mean(errors * weights)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def compute_metrics(params, batch_obs, batch_next_obs, batch_rewards):
        """Compute loss and accuracy metrics."""
        dummy_actions = jnp.zeros(len(batch_obs), dtype=jnp.int32)
        predicted = reward_model.apply(params, rng, batch_obs, dummy_actions, batch_next_obs)

        # MSE loss
        loss = jnp.mean((predicted - batch_rewards) ** 2)

        # Classification accuracy (round predictions to -1, 0, +1)
        predicted_class = jnp.round(jnp.clip(predicted, -1.0, 1.0))
        accuracy = jnp.mean(predicted_class == batch_rewards)

        # Per-class accuracy
        pos_mask = batch_rewards > 0
        neg_mask = batch_rewards < 0
        zero_mask = batch_rewards == 0

        pos_acc = jnp.where(jnp.sum(pos_mask) > 0,
                           jnp.mean((predicted_class == batch_rewards) * pos_mask) / jnp.mean(pos_mask),
                           0.0)
        neg_acc = jnp.where(jnp.sum(neg_mask) > 0,
                           jnp.mean((predicted_class == batch_rewards) * neg_mask) / jnp.mean(neg_mask),
                           0.0)
        zero_acc = jnp.where(jnp.sum(zero_mask) > 0,
                            jnp.mean((predicted_class == batch_rewards) * zero_mask) / jnp.mean(zero_mask),
                            0.0)

        return loss, accuracy, pos_acc, neg_acc, zero_acc

    # Training loop
    best_val_loss = float('inf')
    best_params = params

    num_batches = (len(train_obs) + batch_size - 1) // batch_size

    pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in pbar:
        # Shuffle training data each epoch
        epoch_key = jax.random.fold_in(shuffle_key, epoch)
        epoch_indices = jax.random.permutation(epoch_key, len(train_obs))

        epoch_losses = []

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(train_obs))

            batch_indices = epoch_indices[start:end]
            batch_obs = train_obs[batch_indices]
            batch_next_obs = train_next_obs[batch_indices]
            batch_rewards = train_rewards[batch_indices]

            params, opt_state, loss = train_step(params, opt_state, batch_obs, batch_next_obs, batch_rewards)
            epoch_losses.append(loss)

        train_loss = jnp.mean(jnp.array(epoch_losses))

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_loss, val_acc, pos_acc, neg_acc, zero_acc = compute_metrics(
                params, val_obs, val_next_obs, val_rewards
            )

            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'acc': f'{val_acc:.3f}',
                '+1': f'{pos_acc:.3f}',
                '-1': f'{neg_acc:.3f}',
                '0': f'{zero_acc:.3f}'
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params

                # Save best model
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "params": best_params,
                        "val_loss": best_val_loss,
                        "epoch": epoch,
                    }, f)
        else:
            pbar.set_postfix({'train_loss': f'{train_loss:.4f}'})

    # Final validation metrics
    val_loss, val_acc, pos_acc, neg_acc, zero_acc = compute_metrics(
        best_params, val_obs, val_next_obs, val_rewards
    )

    print(f"\nFinal Results:")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Overall accuracy: {val_acc:.3f}")
    print(f"  +1 accuracy: {pos_acc:.3f}")
    print(f"  -1 accuracy: {neg_acc:.3f}")
    print(f"   0 accuracy: {zero_acc:.3f}")
    print(f"  Model saved to: {save_path}")

    return best_params


def main():
    print("=" * 60)
    print("Reward Predictor Training")
    print("=" * 60)

    # Load data
    obs, next_obs = load_experience_data(num_files=5)

    # Create training data
    obs, next_obs, rewards = create_training_data(obs, next_obs)

    # Train
    params = train_reward_predictor(
        obs=obs,
        next_obs=next_obs,
        rewards=rewards,
        learning_rate=1e-3,
        batch_size=512,
        num_epochs=500,
        frame_stack_size=4,
        save_path="reward_predictor_standalone.pkl",
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
