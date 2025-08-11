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

        x = jnp.concatenate([flat_state, action_one_hot], axis=-1)

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

        prediction = hk.Linear(state.shape[-1])(lstm_out)
        prediction = prediction + 0.8 * state

        return prediction, new_lstm_state

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
