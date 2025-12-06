import jax
import jax.numpy as jnp
import haiku as hk
from typing import NamedTuple, Optional, Tuple









def PongMLPDeep(model_scale_factor=1):
    """
    Deeper MLP with LayerNorm and residual connections.
    4 layers with skip connections for better gradient flow.

    NOTE: Removes score features (last 8 features) from observations to prevent
    out-of-distribution issues during model-based rollouts.
    """

    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        # Remove last 8 features (score_player and score_enemy for 4 frames)
        # Original: 56 features (14 per frame × 4 frames)
        # New: 48 features (12 per frame × 4 frames)
        flat_state = flat_state_full[..., :-8]

        action_one_hot = jax.nn.one_hot(action, num_classes=6)
        x = jnp.concatenate([flat_state, action_one_hot], axis=-1)

        hidden_size = int(256 * model_scale_factor)

        # Layer 1
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)

        # Layer 2 with residual
        residual = x
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = x + residual

        # Layer 3 with residual
        residual = x
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = x + residual

        # Layer 4 with residual
        residual = x
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = x + residual

        # Output layer - predict NEW frame only (not full state)
        # For frame stacking, we need to SHIFT frames and add new frame
        # Features per frame: 12 (excluding scores), Frames: 4, Total: 48
        num_features_per_frame = 12  # Removed score features
        num_frames = 4

        # Predict only the NEW frame (12 values, no scores)
        new_frame = hk.Linear(num_features_per_frame)(x)

        # Shift frames: [f0, f1, f2, f3] -> [f1, f2, f3, NEW]
        # Interleaved format: for feature i, frame f: index = i * num_frames + f

        # Build index arrays for vectorized shifting
        # For each feature, we want to shift frames [1,2,3] to positions [0,1,2]
        shifted_frames = []

        for feat_idx in range(num_features_per_frame):
            # Get frames 1, 2, 3 for this feature (shift left)
            old_indices = jnp.array(
                [feat_idx * num_frames + f for f in range(1, num_frames)]
            )
            old_frames = flat_state[..., old_indices]  # Shape: (batch, 3)

            # Append the new predicted frame
            new_frame_value = new_frame[
                ..., feat_idx : feat_idx + 1
            ]  # Shape: (batch, 1)

            # Concatenate: [frame1, frame2, frame3, NEW]
            feature_frames = jnp.concatenate(
                [old_frames, new_frame_value], axis=-1
            )  # Shape: (batch, 4)
            shifted_frames.append(feature_frames)

        # Stack all features in interleaved format
        # shifted_frames is list of 12 arrays, each (batch, 4)
        # We need to interleave them: [f0_frames, f1_frames, ..., f11_frames]
        shifted_prediction = jnp.concatenate(
            shifted_frames, axis=-1
        )  # Shape: (batch, 48)

        return shifted_prediction, None

    return hk.transform(forward)


def improved_pong_reward(obs, action, frame_stack_size=4):
    """
    Improved reward function for Pong ball tracking.

    Args:
        obs: Flattened observation (should be shape [56] for 4-frame stack)
        action: Action taken
        frame_stack_size: Number of stacked frames
    """
    if frame_stack_size > 1:
        curr_obs = obs[(frame_stack_size - 1) :: frame_stack_size]
        prev_obs = obs[
            (frame_stack_size - 2) :: frame_stack_size
        ]  # Fixed: was -3, should be -2
    else:
        curr_obs = obs
        prev_obs = obs

    # Extract ball and player positions from the latest frame
    # Flat array order: [player_x, player_y, player_h, player_w,
    #                    enemy_x, enemy_y, enemy_h, enemy_w,
    #                    ball_x, ball_y, ball_h, ball_w,
    #                    score_player, score_enemy]
    player_x = curr_obs[0]  # Player X position (pixel coords, ~140)
    player_y = curr_obs[1]  # Player Y position (pixel coords, 0-210)
    enemy_x = curr_obs[4]  # Enemy X position (pixel coords, ~16)
    enemy_y = curr_obs[5]  # Enemy Y position (pixel coords, 0-210)
    ball_x = curr_obs[8]  # Ball X position (pixel coords, 0-160)
    ball_y = curr_obs[9]  # Ball Y position (pixel coords, 0-210)

    prev_ball_x = prev_obs[8]  # Ball X position from previous frame

    # DEBUG: Print ball positions to find correct thresholds
    # Uncomment these lines to see ball_x values when running
    # jax.debug.print("ball_x={bx:.1f}, prev_ball_x={pbx:.1f}, player_x={px:.1f}, enemy_x={ex:.1f}",
    #                 bx=ball_x, pbx=prev_ball_x, px=player_x, ex=enemy_x)

    # 1. Primary reward: Track ball vertically with exponential distance penalty
    y_distance = jnp.abs(ball_y - player_y)

    # Exponential reward that's stronger when closer to ball
    # Range: [0, 1] where 1 is perfect alignment
    tracking_reward = jnp.exp(-y_distance * 3.0)

    # 2. Bonus for being aligned when ball is on player's side
    ball_on_player_side = ball_x < 0.5  # Assuming player is on left side
    alignment_bonus = jnp.where(
        ball_on_player_side & (y_distance < 0.2),
        0.5,  # Strong bonus for good positioning
        0.0,
    )

    # 3. Small encouragement for movement actions (helps exploration)
    # Actions 2 (RIGHT), 3 (LEFT), 4 (RIGHTFIRE), 5 (LEFTFIRE) are movement actions
    movement_reward = jnp.where(
        (action == 2) | (action == 3) | (action == 4) | (action == 5),
        0.05,  # Small bonus
        0.0,
    )

    # Detect crossing events using FIXED boundaries (not paddle positions)
    # In Pong pixel coordinates (0-160):
    #   - Enemy paddle is on LEFT at x=16
    #   - Player paddle is on RIGHT at x=140
    #   - Ball goes out of bounds past the paddles

    # Scoring boundaries (ball goes past these to score)
    # Enemy side (left): ball_x < ~10 means player missed (ball went left past enemy)
    # Player side (right): ball_x > ~150 means enemy missed (ball went right past player)
    ENEMY_WALL_X = 10  # Left boundary - if ball crosses, enemy scored
    PLAYER_WALL_X = 150  # Right boundary - if ball crosses, player scored

    margin = 3

    if frame_stack_size > 1:
        # Ball crossed LEFT wall (enemy scored, player missed)
        ball_crossed_enemy_wall = (prev_ball_x >= ENEMY_WALL_X - margin) & (
            ball_x < ENEMY_WALL_X + margin
        )
        # Ball crossed RIGHT wall (player scored, enemy missed)
        ball_crossed_player_wall = (prev_ball_x <= PLAYER_WALL_X + margin) & (
            ball_x > PLAYER_WALL_X - margin
        )

        margin = 5

        # Check if ball is near enemy paddle but enemy is far away in y (missed ball)
        enemy_far_from_ball = jnp.abs(ball_y - enemy_y) > 15
        enemy_missed = (ball_x < enemy_x) & enemy_far_from_ball

        score_reward = jnp.where(
            enemy_missed,
            1.0,  # Player Scored (ball past enemy)
            jnp.where(ball_x > player_x + margin, -1.0, 0.0),  # Enemy scored
        )

    else:
        score_reward = 0.0

    # Combine rewards with appropriate scaling
    total_reward = score_reward * 2.0  # + movement_reward

    # Scale to reasonable range for learning
    return total_reward


def SeaquestMLPDeep(model_scale_factor=1):
    """
    Deeper MLP with LayerNorm and residual connections for Seaquest.
    4 layers with skip connections for better gradient flow.

    Adapted from PongMLPDeep for Seaquest's observation space and 18 actions.
    """

    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state = state

        # Seaquest has 18 actions (vs Pong's 6)
        action_one_hot = jax.nn.one_hot(action, num_classes=18)
        x = jnp.concatenate([flat_state, action_one_hot], axis=-1)

        hidden_size = int(256 * model_scale_factor)

        # Layer 1
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)

        # Layer 2 with residual
        residual = x
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = x + residual

        # Layer 3 with residual
        residual = x
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = x + residual

        # Layer 4 with residual
        residual = x
        x = hk.Linear(hidden_size)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = x + residual

        # Output layer - predict NEW frame only (not full state)
        # For frame stacking, we need to SHIFT frames and add new frame
        # Seaquest observation size depends on the flattened state structure
        # We'll determine features per frame dynamically from input
        num_frames = 4
        total_features = flat_state.shape[-1]
        num_features_per_frame = total_features // num_frames

        # Predict only the NEW frame
        new_frame = hk.Linear(num_features_per_frame)(x)

        # Shift frames: [f0, f1, f2, f3] -> [f1, f2, f3, NEW]
        # Interleaved format: for feature i, frame f: index = i * num_frames + f

        # Build index arrays for vectorized shifting
        shifted_frames = []

        for feat_idx in range(num_features_per_frame):
            # Get frames 1, 2, 3 for this feature (shift left)
            old_indices = jnp.array(
                [feat_idx * num_frames + f for f in range(1, num_frames)]
            )
            old_frames = flat_state[..., old_indices]  # Shape: (batch, 3)

            # Append the new predicted frame
            new_frame_value = new_frame[
                ..., feat_idx : feat_idx + 1
            ]  # Shape: (batch, 1)

            # Concatenate: [frame1, frame2, frame3, NEW]
            feature_frames = jnp.concatenate(
                [old_frames, new_frame_value], axis=-1
            )  # Shape: (batch, 4)
            shifted_frames.append(feature_frames)

        # Stack all features in interleaved format
        shifted_prediction = jnp.concatenate(
            shifted_frames, axis=-1
        )  # Shape: (batch, total_features)

        return shifted_prediction, None

    return hk.transform(forward)


def RewardPredictorMLPTransition(model_scale_factor=1):
    """
    Transition-based reward predictor that takes (current_state, action, next_state).

    This provides temporal context about the transition, making it more robust to
    world model errors. Based on successful PyTorch world model reference.

    Args:
        current_state: Current observation
        action: Action taken
        next_state: Next observation

    Returns:
        Predicted scalar reward for the transition
    """

    def forward(current_state, action, next_state):
        batch_size = current_state.shape[0] if len(current_state.shape) > 1 else 1

        # Ensure states are 2D
        if len(current_state.shape) == 1:
            current_state = current_state.reshape(1, -1)
        if len(next_state.shape) == 1:
            next_state = next_state.reshape(1, -1)

        # One-hot encode action
        action_one_hot = jax.nn.one_hot(action, num_classes=6)
        if len(action_one_hot.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, -1)

        # Concatenate all inputs: [current_state, action, next_state]
        x = jnp.concatenate([current_state, action_one_hot, next_state], axis=-1)

        # MLP architecture - memory-efficient version
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(128 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(64 * model_scale_factor))(x)
        x = jax.nn.relu(x)

        # Output layer - single scalar reward
        reward = hk.Linear(1)(x)
        reward = jnp.squeeze(
            reward, axis=-1
        )  # Remove last dimension to get scalar per batch

        return reward

    return hk.transform(forward)