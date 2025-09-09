import jax
import jax.numpy as jnp
import haiku as hk
from typing import NamedTuple, Optional, Tuple


def V2_LSTM(model_scale_factor=1):
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

        static_features = jnp.concatenate(
            [flat_state[..., :170], flat_state[..., 179:]], axis=-1
        )
        dynamic_features = flat_state[..., 170:179]

        static_branch = hk.Linear(int(256 * model_scale_factor))(static_features)
        static_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            static_branch
        )
        static_branch = jax.nn.gelu(static_branch)
        static_branch = hk.Linear(int(128 * model_scale_factor))(static_branch)
        static_branch = jax.nn.gelu(static_branch)

        dynamic_branch = hk.Linear(int(128 * model_scale_factor))(dynamic_features)
        dynamic_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            dynamic_branch
        )
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        dynamic_branch = hk.Linear(int(64 * model_scale_factor))(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)

        action_features = hk.Linear(int(64 * model_scale_factor))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(32 * model_scale_factor))(action_features)
        action_features = jax.nn.gelu(action_features)

        combined = jnp.concatenate(
            [static_branch, dynamic_branch, action_features], axis=1
        )

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

        lstm1_out, new_lstm1_state = lstm1(x, lstm1_state)

        new_lstm1_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm1_state.hidden, -5.0, 5.0),
            cell=jnp.clip(new_lstm1_state.cell, -5.0, 5.0),
        )

        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            lstm1_out
        )

        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)

        new_lstm2_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm2_state.hidden, -5.0, 5.0),
            cell=jnp.clip(new_lstm2_state.cell, -5.0, 5.0),
        )

        static_head = hk.Linear(int(256 * model_scale_factor))(lstm2_out)
        static_head = jax.nn.gelu(static_head)
        static_pred = hk.Linear(len(static_features[0]))(static_head)

        dynamic_head = hk.Linear(int(128 * model_scale_factor))(lstm2_out)
        dynamic_head = jax.nn.gelu(dynamic_head)
        dynamic_pred = hk.Linear(9)(dynamic_head)

        full_prediction = jnp.concatenate(
            [static_pred[..., :170], dynamic_pred, static_pred[..., 170:]], axis=-1
        )

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

        new_lstm_state = (new_lstm1_state, new_lstm2_state)
        return prediction, new_lstm_state

    return hk.transform(forward)


def PongActionAwareLSTM(model_scale_factor=1):
    """LSTM that explicitly learns action effects"""

    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]

        # Process action more explicitly
        action_one_hot = jax.nn.one_hot(action, num_classes=6)

        # Separate state and action processing
        state_features = hk.Linear(int(256 * model_scale_factor))(flat_state)
        state_features = jax.nn.relu(state_features)

        action_features = hk.Linear(int(64 * model_scale_factor))(action_one_hot)
        action_features = jax.nn.relu(action_features)
        action_features = hk.Linear(int(64 * model_scale_factor))(action_features)
        action_features = jax.nn.relu(action_features)

        # Combine with attention to action
        combined = jnp.concatenate([state_features, action_features], axis=-1)

        x = hk.Linear(int(512 * model_scale_factor))(combined)
        x = jax.nn.relu(x)
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)

        lstm = hk.LSTM(int(256 * model_scale_factor))
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_out, new_lstm_state = lstm(x, lstm_state)

        # Predict state change (delta) instead of full state
        delta = hk.Linear(state.shape[-1])(lstm_out)

        # Apply action-dependent scaling
        action_scale = hk.Linear(state.shape[-1])(action_features)
        action_scale = jax.nn.sigmoid(action_scale) * 0.2  # Scale factor 0-0.2

        scaled_delta = delta * (0.1 + action_scale)
        prediction = flat_state + scaled_delta

        return prediction, new_lstm_state

    return hk.transform(forward)


def V2_NO_SEP(model_scale_factor=1):
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

        state_features = hk.Linear(int(256 * model_scale_factor))(flat_state)
        state_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            state_features
        )
        state_features = jax.nn.gelu(state_features)
        state_features = hk.Linear(int(128 * model_scale_factor))(state_features)
        state_features = jax.nn.gelu(state_features)

        action_features = hk.Linear(int(64 * model_scale_factor))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(32 * model_scale_factor))(action_features)
        action_features = jax.nn.gelu(action_features)

        combined = jnp.concatenate([state_features, action_features], axis=1)

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

        lstm1_out, new_lstm1_state = lstm1(x, lstm1_state)

        new_lstm1_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm1_state.hidden, -1.0, 1.0),
            cell=jnp.clip(new_lstm1_state.cell, -1.0, 1.0),
        )

        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            lstm1_out
        )

        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)

        new_lstm2_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm2_state.hidden, -1.0, 1.0),
            cell=jnp.clip(new_lstm2_state.cell, -1.0, 1.0),
        )

        prediction_head = hk.Linear(int(128 * model_scale_factor))(lstm2_out)
        prediction_head = jax.nn.gelu(prediction_head)
        prediction = hk.Linear(flat_state.shape[-1])(prediction_head)

        prediction = prediction + 0.5 * flat_state

        new_lstm_state = (new_lstm1_state, new_lstm2_state)
        return prediction, new_lstm_state

    return hk.transform(forward)


def MLP(model_scale_factor=1):
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

        state_features = hk.Linear(int(512 * model_scale_factor))(flat_state)
        state_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            state_features
        )
        state_features = jax.nn.gelu(state_features)
        state_features = hk.Linear(int(256 * model_scale_factor))(state_features)
        state_features = jax.nn.gelu(state_features)

        action_features = hk.Linear(int(128 * model_scale_factor))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(64 * model_scale_factor))(action_features)
        action_features = jax.nn.gelu(action_features)

        combined = jnp.concatenate([state_features, action_features], axis=1)
        x = hk.Linear(int(512 * model_scale_factor))(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)

        x = hk.Linear(int(1024 * model_scale_factor))(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(int(512 * model_scale_factor))(x)
        x = jax.nn.gelu(x)

        output = hk.Linear(int(512 * model_scale_factor))(x)
        output = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(output)
        output = jax.nn.gelu(output)
        output = hk.Linear(int(256 * model_scale_factor))(output)
        output = jax.nn.gelu(output)

        prediction = hk.Linear(flat_state.shape[-1])(output)

        prediction = prediction + flat_state

        dummy_lstm_state = None

        return prediction, dummy_lstm_state

    return hk.transform(forward)


def PongLSTM(model_scale_factor=1):
    def forward(state, action, lstm_state=None):

        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]

        action_one_hot = jax.nn.one_hot(action, num_classes=6)

        action_embed = hk.Linear(32)(action_one_hot)
        x = jnp.concatenate([flat_state, action_embed], axis=-1)

        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(512 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)

        lstm = hk.LSTM(int(256 * model_scale_factor))
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_out, new_lstm_state = lstm(x, lstm_state)

        delta = hk.Linear(state.shape[-1])(lstm_out)
        prediction = state + delta

        return prediction, new_lstm_state

    return hk.transform(forward)


def PongMLP(model_scale_factor=1):
    def forward(state, action, lstm_state=None):

        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]

        action_one_hot = jax.nn.one_hot(action, num_classes=6)

        x = jnp.concatenate([flat_state, action_one_hot], axis=-1)

        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(512 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)

        prediction = hk.Linear(state.shape[-1])(x)
        return prediction, None

    return hk.transform(forward)


def PongMLP2(model_scale_factor=1):
    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]

        action_one_hot = jax.nn.one_hot(action, num_classes=6)

        x = jnp.concatenate([flat_state, action_one_hot], axis=-1)

        # Deeper network for better feature extraction
        x = hk.Linear(int(512 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(1024 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(512 * model_scale_factor))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(int(256 * model_scale_factor))(x)
        x = jax.nn.relu(x)

        # Predict change rather than full state
        delta = hk.Linear(state.shape[-1])(x)

        # Add residual connection for stability
        prediction = flat_state + 0.1 * delta

        return prediction, None

    return hk.transform(forward)


class SimpleDreamerState(NamedTuple):
    """Simplified state for easier integration"""

    h: jnp.ndarray
    z: jnp.ndarray


def PongDreamer(model_scale_factor=1):
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

        flat_state_clipped = jnp.clip(flat_state, -100, 300)
        flat_state_normalized = jnp.tanh(flat_state_clipped / 50.0)

        h_dim = int(32 * model_scale_factor)
        z_dim = int(8 * model_scale_factor)

        if dreamer_state is None:
            h_prev = jnp.zeros((batch_size, h_dim))
            z_prev = jnp.zeros((batch_size, z_dim))
        else:
            h_prev, z_prev = dreamer_state

            h_prev = jnp.clip(h_prev, -1.0, 1.0)
            z_prev = jnp.clip(z_prev, -1.0, 1.0)

        obs_encoded = hk.Sequential(
            [
                hk.Linear(int(32 * model_scale_factor)),
                jax.nn.tanh,
                hk.Linear(int(16 * model_scale_factor)),
                jax.nn.tanh,
            ]
        )(flat_state_normalized)

        transition_input = jnp.concatenate([h_prev, z_prev, action_one_hot], axis=-1)

        h_current = hk.Sequential(
            [
                hk.Linear(int(64 * model_scale_factor)),
                jax.nn.tanh,
                hk.Linear(h_dim),
                jax.nn.tanh,
            ]
        )(transition_input)

        h_current = jnp.clip(h_current, -0.5, 0.5)

        posterior_input = jnp.concatenate([h_current, obs_encoded], axis=-1)
        z_mean = hk.Linear(z_dim, name="z_mean")(posterior_input)
        z_mean = jnp.tanh(z_mean)

        z_current = jnp.clip(z_mean, -0.5, 0.5)

        decoder_input = jnp.concatenate([h_current, z_current], axis=-1)

        decoded = hk.Sequential(
            [
                hk.Linear(int(32 * model_scale_factor)),
                jax.nn.tanh,
                hk.Linear(int(16 * model_scale_factor)),
                jax.nn.tanh,
                hk.Linear(flat_state.shape[-1]),
                jax.nn.tanh,
            ]
        )(decoder_input)

        decoded_scaled = decoded * 2.0

        residual_strength = 0.995
        prediction = (
            residual_strength * flat_state + (1 - residual_strength) * decoded_scaled
        )

        new_dreamer_state = SimpleDreamerState(h=h_current, z=z_current)

        return prediction, new_dreamer_state

    return hk.transform(forward)


def PongLSTMStable(model_scale_factor=1):
    """Ultra-stable LSTM version for comparison"""

    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state = state

        flat_state = jnp.clip(flat_state, -50, 250)

        state_normalized = jnp.tanh(flat_state / 50.0)

        action_one_hot = jax.nn.one_hot(action, num_classes=6)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, 6)

        x = jnp.concatenate([state_normalized, action_one_hot], axis=-1)

        x = hk.Linear(int(32 * model_scale_factor))(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(int(64 * model_scale_factor))(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(int(32 * model_scale_factor))(x)
        x = jax.nn.tanh(x)

        lstm = hk.LSTM(int(32 * model_scale_factor))
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_out, new_lstm_state = lstm(x, lstm_state)

        new_lstm_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm_state.hidden, -0.5, 0.5),
            cell=jnp.clip(new_lstm_state.cell, -1.0, 1.0),
        )

        change = hk.Linear(int(16 * model_scale_factor))(lstm_out)
        change = jax.nn.tanh(change)
        change = hk.Linear(flat_state.shape[-1])(change)

        change = jnp.tanh(change) * 2.0

        prediction = flat_state + 0.005 * change

        return prediction, new_lstm_state

    return hk.transform(forward)


def PongLSTMFixed(model_scale_factor=1):
    """Ultra-stable version - should work with your existing training code"""

    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :]

        flat_state = jnp.clip(
            flat_state, jnp.percentile(flat_state, 1), jnp.percentile(flat_state, 99)
        )

        flat_state_normalized = jnp.tanh(flat_state / 100.0)

        action_one_hot = jax.nn.one_hot(action, num_classes=6)

        x = jnp.concatenate([flat_state_normalized, action_one_hot], axis=-1)

        x = hk.Linear(int(16 * model_scale_factor))(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(int(32 * model_scale_factor))(x)
        x = jax.nn.tanh(x)
        x = hk.Linear(int(16 * model_scale_factor))(x)
        x = jax.nn.tanh(x)

        lstm = hk.LSTM(int(16 * model_scale_factor))
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_out, new_lstm_state = lstm(x, lstm_state)

        new_lstm_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm_state.hidden, -0.1, 0.1),
            cell=jnp.clip(new_lstm_state.cell, -0.2, 0.2),
        )

        delta = hk.Linear(int(8 * model_scale_factor))(lstm_out)
        delta = jax.nn.tanh(delta)
        delta = hk.Linear(state.shape[-1])(delta)

        delta = jnp.tanh(delta) * 0.5

        prediction = flat_state + 0.001 * delta

        return prediction, new_lstm_state

    return hk.transform(forward)


def PongDreamerRSSM(model_scale_factor=1):
    """Simplified Dreamer RSSM implementation for Pong"""

    def forward(state, action, rssm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        # Flatten state handling
        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state = state

        # One-hot encode action
        action_one_hot = jax.nn.one_hot(action, num_classes=6)

        # RSSM parameters
        deter_size = int(128 * model_scale_factor)  # Deterministic state size
        stoch_size = int(32 * model_scale_factor)  # Stochastic state size
        hidden_size = int(128 * model_scale_factor)

        # Initialize RSSM state if None
        if rssm_state is None:
            deter = jnp.zeros((batch_size, deter_size))
            stoch = jnp.zeros((batch_size, stoch_size))
            rssm_state = {"deter": deter, "stoch": stoch}
        else:
            deter = rssm_state["deter"]
            stoch = rssm_state["stoch"]

        # === PRIOR STEP (Imagination) ===
        # Combine previous stochastic state with action
        prior_input = jnp.concatenate([stoch, action_one_hot], axis=-1)

        # Process through MLP before GRU
        x = hk.Linear(hidden_size)(prior_input)
        x = jax.nn.elu(x)

        # GRU cell for deterministic state
        gru = hk.GRU(deter_size)
        new_deter, _ = gru(x, deter)

        # Generate prior distribution parameters
        prior_hidden = hk.Linear(hidden_size)(new_deter)
        prior_hidden = jax.nn.elu(prior_hidden)

        # Split into mean and std for stochastic state
        prior_params = hk.Linear(2 * stoch_size)(prior_hidden)
        prior_mean, prior_std_logit = jnp.split(prior_params, 2, axis=-1)
        prior_std = jax.nn.softplus(prior_std_logit) + 0.1

        # Sample from prior - use hk.next_rng_key() instead of manual key
        rng = hk.next_rng_key()
        prior_stoch = prior_mean + prior_std * jax.random.normal(rng, prior_mean.shape)

        # === POSTERIOR STEP (Observation update) ===
        # Encode current observation
        obs_features = hk.Linear(int(256 * model_scale_factor))(flat_state)
        obs_features = jax.nn.elu(obs_features)
        obs_features = hk.Linear(int(128 * model_scale_factor))(obs_features)
        obs_features = jax.nn.elu(obs_features)

        # Combine deterministic state with observation features
        post_input = jnp.concatenate([new_deter, obs_features], axis=-1)
        post_hidden = hk.Linear(hidden_size)(post_input)
        post_hidden = jax.nn.elu(post_hidden)

        # Generate posterior distribution parameters
        post_params = hk.Linear(2 * stoch_size)(post_hidden)
        post_mean, post_std_logit = jnp.split(post_params, 2, axis=-1)
        post_std = jax.nn.softplus(post_std_logit) + 0.1

        # Sample from posterior
        rng2 = hk.next_rng_key()
        post_stoch = post_mean + post_std * jax.random.normal(rng2, post_mean.shape)

        # === DECODER ===
        # Combine deterministic and stochastic states for feature
        feature = jnp.concatenate([new_deter, post_stoch], axis=-1)

        # Decode to next state prediction
        decoded = hk.Linear(int(512 * model_scale_factor))(feature)
        decoded = jax.nn.elu(decoded)
        decoded = hk.Linear(int(256 * model_scale_factor))(decoded)
        decoded = jax.nn.elu(decoded)

        # Output layer
        prediction = hk.Linear(flat_state.shape[-1])(decoded)

        # Add residual connection
        prediction = flat_state + 0.1 * prediction

        # KL divergence loss between posterior and prior (for training)
        kl_loss = 0.5 * jnp.sum(
            (post_mean - prior_mean) ** 2 / (prior_std**2)
            + (post_std**2) / (prior_std**2)
            - 1
            - 2 * jnp.log(post_std / prior_std),
            axis=-1,
        )

        # New RSSM state
        new_rssm_state = {
            "deter": new_deter,
            "stoch": post_stoch,
            "kl_loss": kl_loss,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "post_mean": post_mean,
            "post_std": post_std,
        }

        return prediction, new_rssm_state

    return hk.transform(forward)


################ Reward functions ############################
def get_dense_pong_reward(obs, action, frame_stack_size=4):
    """
    Dense reward function for Pong that provides continuous feedback.
    Designed to work well with actor-critic methods.
    """

    if frame_stack_size > 1:
        obs = obs[(frame_stack_size - 1) :: frame_stack_size]

    ball_x = obs[8]
    ball_y = obs[9]
    left_paddle_x = obs[4]
    left_paddle_y = obs[5]
    right_paddle_x = obs[0]

    # Component 1: Ball tracking reward (most important)
    # Reward being vertically aligned with the ball
    y_distance = jnp.abs(ball_y - left_paddle_y)
    tracking_reward = jnp.exp(-y_distance / 20.0) * 0.3  # 0-0.3 range

    # Component 2: Defensive positioning
    # Reward being in good defensive position
    defense_reward = jnp.where(
        ball_x < left_paddle_x + 20,  # Ball approaching our side
        0.2,  # Good defense
        0.1,  # Baseline reward for being ready
    )

    # Component 3: Ball proximity reward
    # Reward being close to ball horizontally when it's on our side
    x_distance = jnp.abs(ball_x - left_paddle_x)
    proximity_reward = jnp.where(
        ball_x < left_paddle_x + 30,  # Ball on our side
        jnp.exp(-x_distance / 30.0) * 0.2,  # 0-0.2 range
        0.0,
    )

    # Component 4: Movement reward
    # Encourage active play, but not random movement
    movement_reward = jnp.where(
        (action == 3) | (action == 4),  # LEFT or RIGHTFIRE actions
        0.1,
        -0.05,  # Small penalty for no-op
    )

    # Component 5: Critical events (sparse but important)
    # Big rewards for actual game events
    critical_reward = jnp.where(
        ball_x < left_paddle_x + 5,  # Ball very close - potential save
        jnp.where(
            y_distance < 10,  # And we're aligned
            1.0,  # Big reward for good save position
            -0.5,  # Penalty for missing the ball when close
        ),
        jnp.where(
            ball_x > right_paddle_x - 5,  # Ball at opponent's side
            0.5,  # Moderate reward for offensive position
            0.0,
        ),
    )

    # Combine all components
    total_reward = (
        tracking_reward
        + defense_reward
        + proximity_reward
        + movement_reward
        + critical_reward
    )

    # Ensure rewards are in reasonable range and mostly positive
    # Most of the time, total should be positive (0.1 + 0.3 + 0.1 = 0.5 baseline)
    return total_reward


def get_simple_dense_reward(obs, action, frame_stack_size=4):
    if frame_stack_size > 1:
        obs = obs[(frame_stack_size - 1) :: frame_stack_size]

    # print("hi")
    # print(obs)
    ball_y = obs[9]
    player_y = obs[1]

    y_distance = jnp.abs(ball_y - player_y)
    tracking_reward = 1 - jnp.minimum(y_distance / 50.0, 1.0)

    movement_bonus = jnp.where((action == 3) | (action == 4), 0.1, -0.1)
    base_reward = 0.2

    total = base_reward + tracking_reward + movement_bonus

    # jax.debug.print("Ball Y: {:.1f}, Paddle Y: {:.1f}, Distance: {:.1f}, Tracking: {:.2f}", ball_y, player_y, y_distance, tracking_reward)

    return total * 0.5  # Scale down to keep rewards manageable


def stricter_reward(obs, action, frame_stack_size=4):
    if frame_stack_size > 1:
        obs = obs[(frame_stack_size - 1) :: frame_stack_size]

    ball_y = obs[9]
    player_y = obs[1]

    y_distance = jnp.abs(ball_y - player_y)
    tracking_reward = jnp.exp(-y_distance / 15.0)  # Sharper dropoff

    movement_bonus = jnp.where((action == 3) | (action == 4), 0.05, -0.1)

    return tracking_reward + movement_bonus  # No base reward


# Integration example - replace your reward computation with:
def enhanced_reward_integration(obs, action, frame_stack_size=4):
    """
    How to integrate into your rollout functions
    """
    # Option 1: Use dense reward
    reward = get_dense_pong_reward(obs, action, frame_stack_size)

    # Option 2: Blend with original if you want to keep some original signal
    # original_reward = get_reward_from_ball_position(obs, frame_stack_size)
    # dense_reward = get_dense_pong_reward(obs, action, frame_stack_size)
    # reward = 0.3 * original_reward + 0.7 * dense_reward

    # No need for tanh scaling - rewards are already in good range
    return reward


def improved_pong_reward(obs, action, frame_stack_size=4):
    """
    Improved reward function for Pong ball tracking.

    Args:
        obs: Flattened observation (should be shape [56] for 4-frame stack)
        action: Action taken
        frame_stack_size: Number of stacked frames
    """
    if frame_stack_size > 1:
        obs = obs[(frame_stack_size - 1) :: frame_stack_size]

    # Extract ball and player positions from the latest frame
    # Assuming standard Pong state format: [player_x, player_y, ..., ball_x, ball_y, ...]
    player_y = obs[1]  # Player Y position
    ball_x = obs[8]  # Ball X position
    ball_y = obs[9]  # Ball Y position

    # 1. Primary reward: Track ball vertically
    
    tracking_reward = 10 - abs(obs[9] - obs[1])  # ball_y - player_y
    # y_distance = jnp.abs(ball_y - player_y)
    # tracking_reward = jnp.exp(-y_distance / 20.0)  # Exponential decay

    # 2. Bonus for being in the right horizontal zone when ball approaches
    # Encourage positioning when ball is on player's side
    ball_approaching = ball_x > 0.5  # Assuming normalized coordinates
    # positioning_bonus = jnp.where(ball_approaching & (y_distance < 0.1), 0.2, 0.0)

    # 3. Action rewards - encourage movement actions over no-op
    # Actions 3 (LEFT) and 4 (RIGHTFIRE) are movement actions
    movement_bonus = jnp.where(
        (action == 3) | (action == 4),  # LEFT or RIGHTFIRE
        0.05,  # Small bonus for movement
        -0.02,  # Small penalty for no-op/other actions
    )

    # 4. Prevent extreme movements (optional stabilization)
    # Penalize if player is at screen edges unnecessarily
    edge_penalty = jnp.where((player_y < 0.1) | (player_y > 0.9), -0.1, 0.0)

    total_reward = tracking_reward * 0.5 + movement_bonus * 5

    # Clip to reasonable range to prevent training instability
    return jnp.clip(total_reward, -0.5, 1.5)


def simple_movement_reward(obs, action, frame_stack_size=4):
    # Only reward movement actions that reduce distance
    movement_bonus = jnp.where(
        (action == 3) | (action == 4),  # LEFT or RIGHTFIRE
        1,  # Small bonus for movement
        -1,  # Small penalty for no-op/other actions
    )

    if frame_stack_size > 1:
        obs = obs[(frame_stack_size - 1) :: frame_stack_size]

    # Simple distance-based reward

    distance = -30 + abs(obs[9] - obs[1])  # ball_y - player_y
    return -(distance / 10) + movement_bonus * 5  # Closer = better


# In worldmodelPong.py, modify the reward function:
def get_reward_from_ball_position(obs, frame_stack_size=4):
    """Give rewards based on ball position and movement."""

    if frame_stack_size > 1:
        obs = obs[(frame_stack_size - 1) :: frame_stack_size]

    ball_x = obs[8]
    left_paddle_x = obs[4]
    right_paddle_x = obs[0]

    # More balanced rewards - encourage active play
    reward = jnp.where(
        ball_x < left_paddle_x + 15,
        5.0,  # Higher reward for defending left
        jnp.where(
            ball_x > right_paddle_x - 15,
            -5.0,  # Penalty for ball near right
            0,  # Small penalty for inaction (ball in middle)
        ),
    )
    return reward


def get_enhanced_reward(obs, action, frame_stack_size=4):
    # Original ball position reward
    ball_reward = get_reward_from_ball_position(obs, frame_stack_size)

    # Ball tracking bonus - reward being near ball vertically
    if frame_stack_size > 1:
        obs_current = obs[(frame_stack_size - 1) :: frame_stack_size]

    ball_y = obs_current[9]  # Ball Y position
    paddle_y = obs_current[5]  # Left paddle Y position

    # Reward being close to ball Y position
    distance_bonus = jnp.exp(-jnp.abs(ball_y - paddle_y) / 20.0) * 0.1

    # Movement variety bonus
    movement_bonus = jnp.where((action == 3) | (action == 4), 0.05, -0.01)

    return ball_reward + distance_bonus + movement_bonus
