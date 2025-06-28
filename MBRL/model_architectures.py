
import jax
import jax.numpy as jnp
import haiku as hk

def LSTM():
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

        inputs = jnp.concatenate([flat_state, action_one_hot], axis=1)
        x = hk.Linear(512)(inputs)
        x = jax.nn.relu(x)
        lstm = hk.LSTM(1024)

        # Use provided lstm_state or initialize if None
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        output, new_lstm_state = lstm(x, lstm_state)
        output = jax.nn.relu(output)
        output = hk.Linear(flat_state.shape[-1])(output)
        # output = jnp.clip(output, -10.0, 10.0)


        return output, new_lstm_state  # Return both prediction and new state

    return hk.transform(forward)

def LSTM_with_split_action():
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

        # Improved architecture with residual connections and better processing
        inputs = jnp.concatenate([flat_state, action_one_hot], axis=1)
        
        # State processing branch
        state_features = hk.Linear(512)(flat_state)
        state_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(state_features)
        state_features = jax.nn.gelu(state_features)
        state_features = hk.Linear(256)(state_features)
        state_features = jax.nn.gelu(state_features)
        
        # Action processing branch  
        action_features = hk.Linear(128)(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(64)(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine features
        combined = jnp.concatenate([state_features, action_features], axis=1)
        x = hk.Linear(512)(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        
        # LSTM with larger hidden size
        lstm = hk.LSTM(1024)

        # Use provided lstm_state or initialize if None
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)

        lstm_output, new_lstm_state = lstm(x, lstm_state)
        
        # Multi-layer output processing with skip connection
        output = hk.Linear(512)(lstm_output)
        output = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(output)
        output = jax.nn.gelu(output)
        output = hk.Linear(256)(output)
        output = jax.nn.gelu(output)
        
        # Final prediction with residual connection
        prediction = hk.Linear(flat_state.shape[-1])(output)
        
        # Add residual connection for stability
        prediction = prediction + flat_state
        
        return prediction, new_lstm_state

    return hk.transform(forward)


def MLP():
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

        # State processing branch
        state_features = hk.Linear(512)(flat_state)
        state_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(state_features)
        state_features = jax.nn.gelu(state_features)
        state_features = hk.Linear(256)(state_features)
        state_features = jax.nn.gelu(state_features)
        
        # Action processing branch  
        action_features = hk.Linear(128)(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(64)(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine features
        combined = jnp.concatenate([state_features, action_features], axis=1)
        x = hk.Linear(512)(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        
        # Additional MLP layers instead of LSTM
        x = hk.Linear(1024)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.gelu(x)
        
        # Multi-layer output processing
        output = hk.Linear(512)(x)
        output = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(output)
        output = jax.nn.gelu(output)
        output = hk.Linear(256)(output)
        output = jax.nn.gelu(output)
        
        # Final prediction with residual connection
        prediction = hk.Linear(flat_state.shape[-1])(output)
        
        # Add residual connection for stability
        prediction = prediction + flat_state
        
        # Return dummy LSTM state (None to maintain interface compatibility)
        dummy_lstm_state = None
        
        return prediction, dummy_lstm_state

    return hk.transform(forward)
