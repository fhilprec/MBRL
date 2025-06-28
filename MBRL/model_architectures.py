
import jax
import jax.numpy as jnp
import haiku as hk


def V4_LSTM():
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

        # Separate static and dynamic features (same as V2)
        static_features = jnp.concatenate([flat_state[..., :170], flat_state[..., 179:]], axis=-1)
        dynamic_features = flat_state[..., 170:179]
        
        # Static feature processing (same as V2)
        static_branch = hk.Linear(256)(static_features)
        static_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        static_branch = hk.Linear(128)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        
        # Dynamic feature processing (same as V2)
        dynamic_branch = hk.Linear(128)(dynamic_features)
        dynamic_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        dynamic_branch = hk.Linear(64)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        
        # Action processing (same as V2)
        action_features = hk.Linear(64)(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(32)(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine all features
        combined = jnp.concatenate([static_branch, dynamic_branch, action_features], axis=1)
        
        # Simpler feature mixing (less layers than V3)
        x = hk.Linear(256)(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        
        # Single LSTM (V2 complexity) but with better residual handling
        lstm = hk.LSTM(256)
        
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)
            
        lstm_out, new_lstm_state = lstm(x, lstm_state)
        lstm_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(lstm_out)
        
        # Simple prediction heads (like V2 but with better residuals)
        static_pred = hk.Linear(len(static_features[0]))(lstm_out)
        dynamic_pred = hk.Linear(9)(lstm_out)
        
        # Combine predictions
        full_prediction = jnp.concatenate([
            static_pred[..., :170], 
            dynamic_pred, 
            static_pred[..., 170:]
        ], axis=-1)
        
        # Better residual weights (key improvement over V2)
        static_residual = 0.95  # Very strong for static features
        dynamic_residual = 0.05  # Very weak for dynamic features
        
        residual_weights = jnp.concatenate([
            jnp.full((170,), static_residual),
            jnp.full((9,), dynamic_residual),
            jnp.full((flat_state.shape[-1] - 179,), static_residual)
        ])
        
        prediction = full_prediction + residual_weights * flat_state
        
        # Tighter clipping for dynamic features only
        prediction = prediction.at[..., 170:179].set(
            jnp.clip(prediction[..., 170:179], -10, 100)
        )
        
        return prediction, new_lstm_state

    return hk.transform(forward)

def V3_LSTM():
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
        static_features = jnp.concatenate([flat_state[..., :170], flat_state[..., 179:]], axis=-1)
        dynamic_features = flat_state[..., 170:179]
        
        # Static feature processing (player position, lives, etc.)
        static_branch = hk.Linear(256)(static_features)
        static_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        static_branch = hk.Linear(128)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        
        # Dynamic feature processing (bullets, enemies, etc.)
        dynamic_branch = hk.Linear(128)(dynamic_features)
        dynamic_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        dynamic_branch = hk.Linear(64)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        
        # Action processing with enhanced representation
        action_features = hk.Linear(64)(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(32)(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine all features
        combined = jnp.concatenate([static_branch, dynamic_branch, action_features], axis=1)
        
        # Enhanced feature mixing
        x = hk.Linear(512)(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(512)(x)  # Changed from 256 to 512 to match LSTM1
        x = jax.nn.gelu(x)
        
        # Multi-layer LSTM for better temporal modeling
        lstm1 = hk.LSTM(512)
        lstm2 = hk.LSTM(256)
        
        if lstm_state is None:
            lstm1_state = lstm1.initial_state(batch_size)
            lstm2_state = lstm2.initial_state(batch_size)
        else:
            lstm1_state, lstm2_state = lstm_state
            
        # First LSTM layer with residual connection
        lstm1_out, new_lstm1_state = lstm1(x, lstm1_state)
        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(lstm1_out)
        lstm1_out = lstm1_out + x  # Now dimensions match: (1, 512) + (1, 512)
        
        # Second LSTM layer with projection for residual connection
        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)
        # Project lstm1_out to match lstm2_out dimensions for skip connection
        lstm1_projected = hk.Linear(256)(lstm1_out)
        lstm2_out = lstm2_out + lstm1_projected  # Now: (1, 256) + (1, 256)
        
        # Separate prediction heads for static vs dynamic features
        static_head = hk.Linear(256)(lstm2_out)
        static_head = jax.nn.gelu(static_head)
        static_head = hk.Linear(128)(static_head)  # Extra layer for better representation
        static_head = jax.nn.gelu(static_head)
        static_pred = hk.Linear(len(static_features[0]))(static_head)
        
        dynamic_head = hk.Linear(128)(lstm2_out)
        dynamic_head = jax.nn.gelu(dynamic_head)
        dynamic_head = hk.Linear(64)(dynamic_head)  # Extra layer for better representation
        dynamic_head = jax.nn.gelu(dynamic_head)
        dynamic_pred = hk.Linear(9)(dynamic_head)  # 170:179 = 9 features
        
        # Combine predictions
        full_prediction = jnp.concatenate([
            static_pred[..., :170], 
            dynamic_pred, 
            static_pred[..., 170:]
        ], axis=-1)
        
        # Improved residual connections with learnable weights
        static_residual = 0.9  # Increased for more stability
        dynamic_residual = 0.1  # Decreased to allow more learning
        
        residual_weights = jnp.concatenate([
            jnp.full((170,), static_residual),
            jnp.full((9,), dynamic_residual),
            jnp.full((flat_state.shape[-1] - 179,), static_residual)
        ])
        
        prediction = full_prediction + residual_weights * flat_state
        
        # Tighter stability constraints for dynamic features
        prediction = prediction.at[..., 170:179].set(
            jnp.clip(prediction[..., 170:179], -20, 150)  # Tighter bounds
        )
        
        new_lstm_state = (new_lstm1_state, new_lstm2_state)
        return prediction, new_lstm_state

    return hk.transform(forward)







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
        # Indices 170-178 appear to be dynamic (bullets, enemies)
        static_features = jnp.concatenate([flat_state[..., :170], flat_state[..., 179:]], axis=-1)
        dynamic_features = flat_state[..., 170:179]
        
        # Static feature processing (player position, lives, etc.)
        static_branch = hk.Linear(256)(static_features)
        static_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        static_branch = hk.Linear(128)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        
        # Dynamic feature processing (bullets, enemies, etc.)
        dynamic_branch = hk.Linear(128)(dynamic_features)
        dynamic_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        dynamic_branch = hk.Linear(64)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        
        # Action processing with enhanced representation
        action_features = hk.Linear(64)(action_one_hot)
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(32)(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine all features
        combined = jnp.concatenate([static_branch, dynamic_branch, action_features], axis=1)
        
        # Enhanced feature mixing
        x = hk.Linear(512)(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(256)(x)
        x = jax.nn.gelu(x)
        
        # Multi-layer LSTM for better temporal modeling
        lstm1 = hk.LSTM(512)
        lstm2 = hk.LSTM(256)
        
        if lstm_state is None:
            lstm1_state = lstm1.initial_state(batch_size)
            lstm2_state = lstm2.initial_state(batch_size)
        else:
            lstm1_state, lstm2_state = lstm_state
            
        # First LSTM layer
        lstm1_out, new_lstm1_state = lstm1(x, lstm1_state)
        lstm1_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, new_lstm2_state = lstm2(lstm1_out, lstm2_state)
        
        # Separate prediction heads for static vs dynamic features
        static_head = hk.Linear(256)(lstm2_out)
        static_head = jax.nn.gelu(static_head)
        static_pred = hk.Linear(len(static_features[0]))(static_head)
        
        dynamic_head = hk.Linear(128)(lstm2_out)
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
        prediction = prediction.at[..., 170:179].set(
            jnp.clip(prediction[..., 170:179], -50, 200)
        )
        
        new_lstm_state = (new_lstm1_state, new_lstm2_state)
        return prediction, new_lstm_state

    return hk.transform(forward)

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
