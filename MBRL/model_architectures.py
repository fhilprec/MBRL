import jax
import jax.numpy as jnp
import haiku as hk
from typing import NamedTuple, Optional, Tuple




def V2_LSTM(model_scale_factor = 1):
    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]
        action_one_hot = jax.nn.one_hot(action, num_classes=18)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, 18)

        # Separate static and dynamic features
        # Indices 170-178 appear to be dynamic (bullets)
        static_features = jnp.concatenate(
            [flat_state[..., :170], flat_state[..., 179:]], axis=-1
        )
        dynamic_features = flat_state[..., 170:179]

        # Static feature processing (player position, lives, etc.)
        static_branch = hk.Linear(int(256 * model_scale_factor))(static_features)
        static_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            static_branch
        )
        static_branch = jax.nn.gelu(static_branch)
        static_branch = hk.Linear(int(128 * model_scale_factor))(static_branch)
        static_branch = jax.nn.gelu(static_branch)

        # Dynamic feature processing (bullets, enemies, etc.)
        dynamic_branch = hk.Linear(int(128 * model_scale_factor))(dynamic_features)
        dynamic_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            dynamic_branch
        )
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        dynamic_branch = hk.Linear(int(64 * model_scale_factor))(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)

        # Action processing with enhanced representation
        action_features = hk.Linear(int(64 * model_scale_factor))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(32 * model_scale_factor))(action_features)
        action_features = jax.nn.gelu(action_features)

        # Combine all features
        combined = jnp.concatenate(
            [static_branch, dynamic_branch, action_features], axis=1
        )

        # Enhanced feature mixing
        x = hk.Linear(int(512 * model_scale_factor))(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.gelu(x)

        lstm1 = hk.LSTM(int(512 * model_scale_factor))
        lstm2 = hk.LSTM(int(256 * model_scale_factor))

        if lstm_state is None:
            lstm1_state = lstm1.initial_state(batch_size)
            lstm2_state = lstm2.initial_state(batch_size)
        else:

            lstm1_state, lstm2_state = lstm_state

        # First LSTM layer
        lstm1_out, new_lstm1_state = lstm1(x, lstm1_state)

        # ADD THIS: Clip LSTM1 states to prevent explosion
        new_lstm1_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm1_state.hidden, -5.0, 5.0),
            cell=jnp.clip(new_lstm1_state.cell, -5.0, 5.0),
        )

        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            lstm1_out
        )

        # Second LSTM layer
        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)

        new_lstm2_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm2_state.hidden, -5.0, 5.0),
            cell=jnp.clip(new_lstm2_state.cell, -5.0, 5.0),
        )

        # Separate prediction heads for static vs dynamic features
        static_head = hk.Linear(int(256 * model_scale_factor))(lstm2_out)
        static_head = jax.nn.gelu(static_head)
        static_pred = hk.Linear(len(static_features[0]))(static_head)

        dynamic_head = hk.Linear(int(128 * model_scale_factor))(lstm2_out)
        dynamic_head = jax.nn.gelu(dynamic_head)
        dynamic_pred = hk.Linear(9)(dynamic_head)  # 170:179 = 9 features

        # Combine predictions
        full_prediction = jnp.concatenate(
            [static_pred[..., :170], dynamic_pred, static_pred[..., 170:]], axis=-1
        )

        # Apply different residual connections
        # Stronger residual for static features, weaker for dynamic
        static_residual = 0.8
        dynamic_residual = 0.3

        residual_weights = jnp.concatenate(
            [
                jnp.full((170,), static_residual),
                jnp.full((9,), dynamic_residual),
                jnp.full((flat_state.shape[-1] - 179,), static_residual),
            ]
        )

        prediction = full_prediction + residual_weights * flat_state

        # Stability constraints for dynamic features
        # prediction = prediction.at[..., 170:179].set(
        #     jnp.clip(prediction[..., 170:179], -50, 200)
        # )

        new_lstm_state = (new_lstm1_state, new_lstm2_state)
        return prediction, new_lstm_state

    return hk.transform(forward)


def V2_NO_SEP(model_scale_factor = 1):
    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]
        action_one_hot = jax.nn.one_hot(action, num_classes=6)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, 6)

        # Unified feature processing
        state_features = hk.Linear(int(256 * model_scale_factor))(flat_state)
        state_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            state_features
        )
        state_features = jax.nn.gelu(state_features)
        state_features = hk.Linear(int(128 * model_scale_factor))(state_features)
        state_features = jax.nn.gelu(state_features)

        # Action processing
        action_features = hk.Linear(int(64 * model_scale_factor))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(32 * model_scale_factor))(action_features)
        action_features = jax.nn.gelu(action_features)

        # Combine features
        combined = jnp.concatenate([state_features, action_features], axis=1)

        # Feature mixing
        x = hk.Linear(int(256 * model_scale_factor))(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(int(128 * model_scale_factor))(x)
        x = jax.nn.gelu(x)

        lstm1 = hk.LSTM(int(256 * model_scale_factor))
        lstm2 = hk.LSTM(int(128 * model_scale_factor))

        if lstm_state is None:
            lstm1_state = lstm1.initial_state(batch_size)
            lstm2_state = lstm2.initial_state(batch_size)
        else:
            lstm1_state, lstm2_state = lstm_state

        # First LSTM layer
        lstm1_out, new_lstm1_state = lstm1(x, lstm1_state)

        # Clip LSTM1 states to prevent explosion
        new_lstm1_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm1_state.hidden, -1.0, 1.0),
            cell=jnp.clip(new_lstm1_state.cell, -1.0, 1.0),
        )

        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            lstm1_out
        )

        # Second LSTM layer
        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)

        new_lstm2_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm2_state.hidden, -1.0, 1.0),
            cell=jnp.clip(new_lstm2_state.cell, -1.0, 1.0),
        )

        # Unified prediction head
        prediction_head = hk.Linear(int(128 * model_scale_factor))(lstm2_out)
        prediction_head = jax.nn.gelu(prediction_head)
        prediction = hk.Linear(flat_state.shape[-1])(prediction_head)

        # Simple residual connection
        prediction = prediction + 0.5 * flat_state

        new_lstm_state = (new_lstm1_state, new_lstm2_state)
        return prediction, new_lstm_state

    return hk.transform(forward)


def MLP(model_scale_factor = 1):
    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]
        action_one_hot = jax.nn.one_hot(action, num_classes=6)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, 6)

        # State processing branch
        state_features = hk.Linear(int(512 * model_scale_factor))(flat_state)
        state_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            state_features
        )
        state_features = jax.nn.gelu(state_features)
        state_features = hk.Linear(int(256 * model_scale_factor))(state_features)
        state_features = jax.nn.gelu(state_features)

        # Action processing branch
        action_features = hk.Linear(int(128 * model_scale_factor))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(64 * model_scale_factor))(action_features)
        action_features = jax.nn.gelu(action_features)

        # Combine features
        combined = jnp.concatenate([state_features, action_features], axis=1)
        x = hk.Linear(int(512 * model_scale_factor))(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)

        # Additional MLP layers instead of LSTM
        x = hk.Linear(int(1024 * model_scale_factor))(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(int(512 * model_scale_factor))(x)
        x = jax.nn.gelu(x)

        # Multi-layer output processing
        output = hk.Linear(int(512 * model_scale_factor))(x)
        output = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(output)
        output = jax.nn.gelu(output)
        output = hk.Linear(int(256 * model_scale_factor))(output)
        output = jax.nn.gelu(output)

        # Final prediction with residual connection
        prediction = hk.Linear(flat_state.shape[-1])(output)

        # Add residual connection for stability
        prediction = prediction + flat_state

        # Return dummy LSTM state (None to maintain interface compatibility)
        dummy_lstm_state = None

        return prediction, dummy_lstm_state

    return hk.transform(forward)


def PongLSTM(model_scale_factor = 1):
    def forward(state, action, lstm_state=None):
        # Much simpler architecture for Pong
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]

        # Normalize and embed state
        action_one_hot = jax.nn.one_hot(
            action, num_classes=6
        )  # Pong has 6 discrete actions

        # Concatenate state and action directly
        x = jnp.concatenate([flat_state, action_one_hot], axis=-1)

        # Feedforward encoder before LSTM
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(512 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)

        # Single LSTM layer
        lstm = hk.LSTM(int(256 * model_scale_factor))
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_out, new_lstm_state = lstm(x, lstm_state)

        # Direct prediction
        prediction = hk.Linear(state.shape[-1])(lstm_out)
        prediction = prediction + 0.8 * state  # Strong residual for stable features

        return prediction, new_lstm_state

    return hk.transform(forward)


class SimpleDreamerState(NamedTuple):
    """Simplified state for easier integration"""

    h: jnp.ndarray
    z: jnp.ndarray


def PongDreamer(model_scale_factor = 1):
    """Much more stable Dreamer implementation"""

    def forward(state, action, dreamer_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state = state

        action_one_hot = jax.nn.one_hot(action, num_classes=6)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, 6)

        # CRITICAL FIX 1: Input normalization and clipping
        # Clip extreme values and normalize
        flat_state_clipped = jnp.clip(flat_state, -100, 300)
        flat_state_normalized = jnp.tanh(
            flat_state_clipped / 50.0
        )  # Normalize to [-1, 1]

        # Much smaller dimensions to prevent explosion
        h_dim = int(32 * model_scale_factor)  # Reduced from 128
        z_dim = int(8 * model_scale_factor)  # Reduced from 32

        # Initialize or unpack state
        if dreamer_state is None:
            h_prev = jnp.zeros((batch_size, h_dim))
            z_prev = jnp.zeros((batch_size, z_dim))
        else:
            h_prev, z_prev = dreamer_state
            # CRITICAL FIX 2: Clip previous states
            h_prev = jnp.clip(h_prev, -1.0, 1.0)
            z_prev = jnp.clip(z_prev, -1.0, 1.0)

        # OBSERVATION ENCODER - much simpler
        obs_encoded = hk.Sequential(
            [
                hk.Linear(int(32 * model_scale_factor)),
                jax.nn.tanh,  # Use tanh instead of swish for bounded output
                hk.Linear(int(16 * model_scale_factor)),
                jax.nn.tanh,
            ]
        )(flat_state_normalized)

        # DETERMINISTIC RECURRENT STATE - simplified
        transition_input = jnp.concatenate([h_prev, z_prev, action_one_hot], axis=-1)

        h_current = hk.Sequential(
            [
                hk.Linear(int(64 * model_scale_factor)),
                jax.nn.tanh,
                hk.Linear(h_dim),
                jax.nn.tanh,  # Ensure bounded output
            ]
        )(transition_input)

        # CRITICAL FIX 3: Aggressive clipping of hidden state
        h_current = jnp.clip(h_current, -0.5, 0.5)

        # STOCHASTIC LATENT STATE - much simpler
        posterior_input = jnp.concatenate([h_current, obs_encoded], axis=-1)
        z_mean = hk.Linear(z_dim, name="z_mean")(posterior_input)
        z_mean = jnp.tanh(z_mean)  # Bound the mean

        # For deterministic rollout, use mean (no noise)
        z_current = jnp.clip(z_mean, -0.5, 0.5)

        # DECODER - much simpler and more constrained
        decoder_input = jnp.concatenate([h_current, z_current], axis=-1)

        decoded = hk.Sequential(
            [
                hk.Linear(int(32 * model_scale_factor)),
                jax.nn.tanh,
                hk.Linear(int(16 * model_scale_factor)),
                jax.nn.tanh,
                hk.Linear(flat_state.shape[-1]),
                jax.nn.tanh,  # CRITICAL: Bound the output
            ]
        )(decoder_input)

        # CRITICAL FIX 4: Much stronger residual connection and bounded changes
        # Scale the decoded output to be small changes only
        decoded_scaled = decoded * 2.0  # Max change of ±2 per feature

        # Very strong residual connection (99.5% original)
        residual_strength = 0.995
        prediction = (
            residual_strength * flat_state + (1 - residual_strength) * decoded_scaled
        )

        new_dreamer_state = SimpleDreamerState(h=h_current, z=z_current)

        return prediction, new_dreamer_state

    return hk.transform(forward)


def PongLSTMStable(model_scale_factor = 1):
    """Ultra-stable LSTM version for comparison"""

    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state = state

        # AGGRESSIVE INPUT STABILIZATION
        # Clip extreme outliers
        flat_state = jnp.clip(flat_state, -50, 250)

        # Normalize to [-1, 1] range
        state_normalized = jnp.tanh(flat_state / 50.0)

        action_one_hot = jax.nn.one_hot(action, num_classes=6)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, 6)

        # SIMPLE FEATURE PROCESSING
        x = jnp.concatenate([state_normalized, action_one_hot], axis=-1)

        # Much smaller network
        x = hk.Linear(int(32 * model_scale_factor))(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(int(64 * model_scale_factor))(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(int(32 * model_scale_factor))(x)
        x = jax.nn.tanh(x)

        # TINY LSTM
        lstm = hk.LSTM(int(32 * model_scale_factor))
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_out, new_lstm_state = lstm(x, lstm_state)

        # ULTRA-AGGRESSIVE CLIPPING
        new_lstm_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm_state.hidden, -0.5, 0.5),
            cell=jnp.clip(new_lstm_state.cell, -1.0, 1.0),
        )

        # MINIMAL OUTPUT CHANGE
        change = hk.Linear(int(16 * model_scale_factor))(lstm_out)
        change = jax.nn.tanh(change)
        change = hk.Linear(flat_state.shape[-1])(change)

        # Bounded change magnitude
        change = jnp.tanh(change) * 2.0  # Max change of ±2 per feature

        # MAXIMUM RESIDUAL CONNECTION (99.5% original)
        prediction = flat_state + 0.005 * change

        return prediction, new_lstm_state

    return hk.transform(forward)


def PongLSTMFixed(model_scale_factor = 1):
    """Ultra-stable version - should work with your existing training code"""

    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]

        # KEY FIX 1: Much more aggressive input stabilization
        # Remove extreme outliers first
        flat_state = jnp.clip(
            flat_state, jnp.percentile(flat_state, 1), jnp.percentile(flat_state, 99)
        )

        # Then normalize to a small range
        flat_state_normalized = jnp.tanh(flat_state / 100.0)

        action_one_hot = jax.nn.one_hot(action, num_classes=6)

        # Concatenate normalized state and action
        x = jnp.concatenate([flat_state_normalized, action_one_hot], axis=-1)

        # KEY FIX 2: Much smaller network
        x = hk.Linear(int(16 * model_scale_factor))(x)  # Very small
        x = jax.nn.tanh(x)  # Bounded
        x = hk.Linear(int(32 * model_scale_factor))(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(int(16 * model_scale_factor))(x)
        x = jax.nn.tanh(x)

        # KEY FIX 3: Tiny LSTM with extreme clipping
        lstm = hk.LSTM(int(16 * model_scale_factor))
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_out, new_lstm_state = lstm(x, lstm_state)

        # KEY FIX 4: Extreme state clipping
        new_lstm_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm_state.hidden, -0.1, 0.1),  # Very tight
            cell=jnp.clip(new_lstm_state.cell, -0.2, 0.2),
        )

        # KEY FIX 5: Tiny output changes only
        delta = hk.Linear(int(8 * model_scale_factor))(lstm_out)
        delta = jax.nn.tanh(delta)
        delta = hk.Linear(state.shape[-1])(delta)

        # Extremely small changes
        delta = jnp.tanh(delta) * 0.5  # Max change of ±0.5 per feature

        # KEY FIX 6: 99.9% residual connection
        prediction = flat_state + 0.001 * delta

        return prediction, new_lstm_state

    return hk.transform(forward)
