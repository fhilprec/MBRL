
import jax
import jax.numpy as jnp
import haiku as hk


MODEL_SCALE_FACTOR = 2  

def V2_LSTM():
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
        static_features = jnp.concatenate([flat_state[..., :170], flat_state[..., 179:]], axis=-1)
        dynamic_features = flat_state[..., 170:179]
        
        # Static feature processing (player position, lives, etc.)
        static_branch = hk.Linear(int(256 * MODEL_SCALE_FACTOR))(static_features)
        static_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        static_branch = hk.Linear(int(128 * MODEL_SCALE_FACTOR))(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        
        # Dynamic feature processing (bullets, enemies, etc.)
        dynamic_branch = hk.Linear(int(128 * MODEL_SCALE_FACTOR))(dynamic_features)
        dynamic_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        dynamic_branch = hk.Linear(int(64 * MODEL_SCALE_FACTOR))(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        
        # Action processing with enhanced representation
        action_features = hk.Linear(int(64 * MODEL_SCALE_FACTOR))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(32 * MODEL_SCALE_FACTOR))(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine all features
        combined = jnp.concatenate([static_branch, dynamic_branch, action_features], axis=1)
        
        # Enhanced feature mixing
        x = hk.Linear(int(512 * MODEL_SCALE_FACTOR))(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(int(256 * MODEL_SCALE_FACTOR))(x)
        x = jax.nn.gelu(x)
        
        lstm1 = hk.LSTM(int(512 * MODEL_SCALE_FACTOR))
        lstm2 = hk.LSTM(int(256 * MODEL_SCALE_FACTOR))

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
            cell=jnp.clip(new_lstm1_state.cell, -5.0, 5.0)
        )

        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(lstm1_out)

        # Second LSTM layer
        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)

        
        new_lstm2_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm2_state.hidden, -5.0, 5.0),
            cell=jnp.clip(new_lstm2_state.cell, -5.0, 5.0)
        )

        # Separate prediction heads for static vs dynamic features
        static_head = hk.Linear(int(256 * MODEL_SCALE_FACTOR))(lstm2_out)
        static_head = jax.nn.gelu(static_head)
        static_pred = hk.Linear(len(static_features[0]))(static_head)
        
        dynamic_head = hk.Linear(int(128 * MODEL_SCALE_FACTOR))(lstm2_out)
        dynamic_head = jax.nn.gelu(dynamic_head)
        dynamic_pred = hk.Linear(9)(dynamic_head)  # 170:179 = 9 features
        
        # Combine predictions
        full_prediction = jnp.concatenate([
            static_pred[..., :170], 
            dynamic_pred, 
            static_pred[..., 170:]
        ], axis=-1)
        
        # Apply different residual connections
        # Stronger residual for static features, weaker for dynamic
        static_residual = 0.8
        dynamic_residual = 0.3
        
        residual_weights = jnp.concatenate([
            jnp.full((170,), static_residual),
            jnp.full((9,), dynamic_residual),
            jnp.full((flat_state.shape[-1] - 179,), static_residual)
        ])
        
        prediction = full_prediction + residual_weights * flat_state
        
        # Stability constraints for dynamic features
        # prediction = prediction.at[..., 170:179].set(
        #     jnp.clip(prediction[..., 170:179], -50, 200)
        # )
        
        new_lstm_state = (new_lstm1_state, new_lstm2_state)
        return prediction, new_lstm_state

    return hk.transform(forward)


def V2_NO_SEP():
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

        # Unified feature processing
        state_features = hk.Linear(int(256 * MODEL_SCALE_FACTOR))(flat_state)
        state_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(state_features)
        state_features = jax.nn.gelu(state_features)
        state_features = hk.Linear(int(128 * MODEL_SCALE_FACTOR))(state_features)
        state_features = jax.nn.gelu(state_features)
        
        # Action processing
        action_features = hk.Linear(int(64 * MODEL_SCALE_FACTOR))(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(int(32 * MODEL_SCALE_FACTOR))(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine features
        combined = jnp.concatenate([state_features, action_features], axis=1)
        
        # Feature mixing
        x = hk.Linear(int(256 * MODEL_SCALE_FACTOR))(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(int(128 * MODEL_SCALE_FACTOR))(x)
        x = jax.nn.gelu(x)
        
        lstm1 = hk.LSTM(int(256 * MODEL_SCALE_FACTOR))
        lstm2 = hk.LSTM(int(128 * MODEL_SCALE_FACTOR))

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
            cell=jnp.clip(new_lstm1_state.cell, -1.0, 1.0)
        )

        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(lstm1_out)

        # Second LSTM layer
        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)

        new_lstm2_state = hk.LSTMState(
            hidden=jnp.clip(new_lstm2_state.hidden, -1.0, 1.0),
            cell=jnp.clip(new_lstm2_state.cell, -1.0, 1.0)
        )

        # Unified prediction head
        prediction_head = hk.Linear(int(128 * MODEL_SCALE_FACTOR))(lstm2_out)
        prediction_head = jax.nn.gelu(prediction_head)
        prediction = hk.Linear(flat_state.shape[-1])(prediction_head)
        
        # Simple residual connection
        prediction = prediction + 0.5 * flat_state
        
        new_lstm_state = (new_lstm1_state, new_lstm2_state)
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
