"""
Standalone reward predictor training script.
Trains the reward predictor to predict position-based rewards:
  - +1 when ball goes past enemy paddle (x < 16) → Player scores
  - -1 when ball goes past player paddle (x > 140) → Enemy scores
  -  0 when ball is in play (16 <= x <= 140)

Coordinate system: WIDTH=160 pixels
  - Enemy paddle: x=16 (left side)
  - Player paddle: x=140 (right side)

This trains the model to recognize scoring events purely from ball position.
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
from model_architectures import RewardPredictorMLPTransition


def calculate_position_based_reward(next_flat_obs, frame_stack_size=4):
    """
    Calculate reward PURELY based on ball position in next_obs.

    Ball x coordinate system (pixel coordinates, WIDTH=160):
    - Enemy paddle at x=16 (left side)
    - Player paddle at x=140 (right side)
    - Ball center at x=78

    Scoring logic:
    - x < 16: Ball past enemy paddle → Enemy misses → PLAYER SCORED (+1)
    - x > 140: Ball past player paddle → Player misses → ENEMY SCORED (-1)
    - Otherwise: Ball in play → 0

    This is the ONLY reward function used for training.
    The model learns to predict scoring purely from ball positions.

    Frame-stacked observations are INTERLEAVED:
    [f1_e0, f2_e0, f3_e0, f4_e0, f1_e1, f2_e1, ...]
    To extract the last frame, take every frame_stack_size'th element
    starting from (frame_stack_size - 1).
    """
    # Extract the latest frame from the interleaved observation
    # Takes indices [3, 7, 11, 15, ...] for frame_stack_size=4
    last_frame = next_flat_obs[..., (frame_stack_size - 1) :: frame_stack_size]

    # Ball x is at index 8 in the flat observation structure:
    # [player.x, player.y, player.width, player.height,  # 0-3
    #  enemy.x, enemy.y, enemy.width, enemy.height,      # 4-7
    #  ball.x, ball.y, ball.width, ball.height,          # 8-11
    #  score_player, score_enemy]                         # 12-13
    ball_x = last_frame[..., 8]

    # Player scores when ball goes past enemy paddle (left side)
    player_scored = ball_x < 16.0

    # Enemy scores when ball goes past player paddle (right side)
    enemy_scored = ball_x > 140.0

    # Convert to reward
    reward = jnp.where(player_scored, 1.0, jnp.where(enemy_scored, -1.0, 0.0))

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
    Output: position-based reward (-1, 0, or +1) based on ball position in next_obs

    Reward assignment (pixel coordinates, WIDTH=160):
    - +1 when ball_x < 16 (past enemy paddle = player scored)
    - -1 when ball_x > 140 (past player paddle = enemy scored)
    -  0 otherwise (ball in play)
    """
    # Calculate rewards using ONLY ball position in next_obs
    rewards = calculate_position_based_reward(next_obs, frame_stack_size)

    # Count reward distribution
    num_positive = jnp.sum(rewards > 0)
    num_negative = jnp.sum(rewards < 0)
    num_zero = jnp.sum(rewards == 0)

    print(f"\nPosition-based reward distribution:")
    print(f"  +1 (ball past enemy paddle, x<16): {num_positive}")
    print(f"  -1 (ball past player paddle, x>140): {num_negative}")
    print(f"   0 (ball in play, 16<=x<=140): {num_zero}")

    # Show ball position statistics for labeled cases
    # Extract last frame and get ball_x (same as in reward calculation)
    last_frame_next = next_obs[:, (frame_stack_size - 1) :: frame_stack_size]
    ball_x_next = last_frame_next[:, 8]

    pos_mask = rewards > 0
    neg_mask = rewards < 0

    if jnp.sum(pos_mask) > 0:
        print(f"\n+1 cases (player scored):")
        print(
            f"  Ball X range: {jnp.min(ball_x_next[pos_mask]):.1f} - {jnp.max(ball_x_next[pos_mask]):.1f}"
        )
        print(f"  Mean: {jnp.mean(ball_x_next[pos_mask]):.1f}")

    if jnp.sum(neg_mask) > 0:
        print(f"\n-1 cases (enemy scored):")
        print(
            f"  Ball X range: {jnp.min(ball_x_next[neg_mask]):.1f} - {jnp.max(ball_x_next[neg_mask]):.1f}"
        )
        print(f"  Mean: {jnp.mean(ball_x_next[neg_mask]):.1f}")

    print(f"\nOverall ball_x statistics:")
    print(f"  Min: {jnp.min(ball_x_next):.1f}")
    print(f"  Max: {jnp.max(ball_x_next):.1f}")
    print(f"  Mean: {jnp.mean(ball_x_next):.1f}")

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
    reward_model = RewardPredictorMLPTransition(model_scale_factor=1)

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

    print(
        f"Class weights: +1={weight_pos:.1f}, -1={weight_neg:.1f}, 0={weight_zero:.1f}"
    )

    # Training step with weighted loss
    @jax.jit
    def train_step(params, opt_state, batch_obs, batch_next_obs, batch_rewards):
        def loss_fn(p):
            # Dummy action (not used in position-only model after user's edit)
            dummy_actions = jnp.zeros(len(batch_obs), dtype=jnp.int32)

            predicted = reward_model.apply(
                p, rng, batch_obs, dummy_actions, batch_next_obs
            )

            # Weighted MSE loss to handle class imbalance
            errors = (predicted - batch_rewards) ** 2

            # Assign weights based on true reward class
            weights = jnp.where(
                batch_rewards > 0,
                weight_pos,
                jnp.where(batch_rewards < 0, weight_neg, weight_zero),
            )

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
        predicted = reward_model.apply(
            params, rng, batch_obs, dummy_actions, batch_next_obs
        )

        # MSE loss
        loss = jnp.mean((predicted - batch_rewards) ** 2)

        # Classification accuracy (round predictions to -1, 0, +1)
        predicted_class = jnp.round(jnp.clip(predicted, -1.0, 1.0))
        accuracy = jnp.mean(predicted_class == batch_rewards)

        # Per-class accuracy
        pos_mask = batch_rewards > 0
        neg_mask = batch_rewards < 0
        zero_mask = batch_rewards == 0

        pos_acc = jnp.where(
            jnp.sum(pos_mask) > 0,
            jnp.mean((predicted_class == batch_rewards) * pos_mask)
            / jnp.mean(pos_mask),
            0.0,
        )
        neg_acc = jnp.where(
            jnp.sum(neg_mask) > 0,
            jnp.mean((predicted_class == batch_rewards) * neg_mask)
            / jnp.mean(neg_mask),
            0.0,
        )
        zero_acc = jnp.where(
            jnp.sum(zero_mask) > 0,
            jnp.mean((predicted_class == batch_rewards) * zero_mask)
            / jnp.mean(zero_mask),
            0.0,
        )

        return loss, accuracy, pos_acc, neg_acc, zero_acc

    # Training loop
    best_val_loss = float("inf")
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

            params, opt_state, loss = train_step(
                params, opt_state, batch_obs, batch_next_obs, batch_rewards
            )
            epoch_losses.append(loss)

        train_loss = jnp.mean(jnp.array(epoch_losses))

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_loss, val_acc, pos_acc, neg_acc, zero_acc = compute_metrics(
                params, val_obs, val_next_obs, val_rewards
            )

            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "acc": f"{val_acc:.3f}",
                    "+1": f"{pos_acc:.3f}",
                    "-1": f"{neg_acc:.3f}",
                    "0": f"{zero_acc:.3f}",
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params

                # Save best model
                with open(save_path, "wb") as f:
                    pickle.dump(
                        {
                            "params": best_params,
                            "val_loss": best_val_loss,
                            "epoch": epoch,
                        },
                        f,
                    )
        else:
            pbar.set_postfix({"train_loss": f"{train_loss:.4f}"})

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
    print("Reward Predictor Training (Position-Based)")
    print("=" * 60)
    print("Reward definition:")
    print("  +1: Ball past enemy paddle (x < 16) → Player scores")
    print("  -1: Ball past player paddle (x > 140) → Enemy scores")
    print("   0: Ball in play (16 <= x <= 140)")
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
        num_epochs=100,
        frame_stack_size=4,
        save_path="reward_predictor_standalone.pkl",
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
