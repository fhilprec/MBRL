
import jax
import jax.numpy as jnp
import haiku as hk


# ...existing code...

def V5_Physics_Aware_LSTM():
    """Enhanced model with physics-aware processing for different entity types"""
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

        # Separate features by physics characteristics
        player_features = flat_state[..., :5]  # Player position/state
        entity_positions = flat_state[..., 5:170]  # All entity positions (sharks, subs, divers)
        missile_features = flat_state[..., 170:175]  # Player missile (fast dynamics)
        game_state = flat_state[..., 175:179]  # Score, lives, oxygen, divers
        
        # Physics-aware processing branches
        # 1. Player dynamics (medium speed, action-dependent)
        player_branch = hk.Linear(64)(jnp.concatenate([player_features, action_one_hot], axis=-1))
        player_branch = jax.nn.gelu(player_branch)
        player_branch = hk.Linear(32)(player_branch)
        
        # 2. Entity positions (slow dynamics, mostly independent movement)
        entity_branch = hk.Linear(256)(entity_positions)
        entity_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(entity_branch)
        entity_branch = jax.nn.gelu(entity_branch)
        entity_branch = hk.Linear(128)(entity_branch)
        
        # 3. Missile dynamics (fast, physics-based)
        missile_action_context = jnp.concatenate([missile_features, action_one_hot[:, 10:]], axis=-1)  # Fire actions
        missile_branch = hk.Linear(32)(missile_action_context)
        missile_branch = jax.nn.gelu(missile_branch)
        missile_branch = hk.Linear(16)(missile_branch)
        
        # 4. Game state (discrete updates, event-driven)
        game_branch = hk.Linear(32)(game_state)
        game_branch = jax.nn.gelu(game_branch)
        game_branch = hk.Linear(16)(game_branch)
        
        # Combine with attention-like weighting
        combined = jnp.concatenate([player_branch, entity_branch, missile_branch, game_branch], axis=-1)
        
        # Temporal processing with specialized LSTM
        x = hk.Linear(256)(combined)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        
        # Single LSTM but with more capacity
        lstm = hk.LSTM(512)
        
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)
            
        lstm_out, new_lstm_state = lstm(x, lstm_state)
        lstm_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(lstm_out)
        
        # Specialized prediction heads
        player_pred = hk.Linear(5)(lstm_out)
        entity_pred = hk.Linear(165)(lstm_out)  # 5:170
        missile_pred = hk.Linear(5)(lstm_out)   # 170:175
        game_pred = hk.Linear(4)(lstm_out)      # 175:179
        
        # Combine predictions
        full_prediction = jnp.concatenate([
            player_pred, entity_pred, missile_pred, game_pred
        ], axis=-1)
        
        # Physics-informed residual connections
        player_residual = 0.7    # Medium residual for player
        entity_residual = 0.95   # High residual for slow entities
        missile_residual = 0.1   # Low residual for fast missiles
        game_residual = 0.8      # High residual for discrete game state
        
        residual_weights = jnp.concatenate([
            jnp.full((5,), player_residual),
            jnp.full((165,), entity_residual),
            jnp.full((5,), missile_residual),
            jnp.full((4,), game_residual)
        ])
        
        prediction = full_prediction + residual_weights * flat_state
        
        # Specialized constraints for different feature types
        # Missile positions can change rapidly
        prediction = prediction.at[..., 170:175].set(
            jnp.clip(prediction[..., 170:175], -20, 200)
        )
        
        # Score should only increase or stay same
        prediction = prediction.at[..., 176].set(
            jnp.maximum(prediction[..., 176], flat_state[..., 176])
        )
        
        return prediction, new_lstm_state

    return hk.transform(forward)

def V6_Hierarchical_LSTM():
    """Hierarchical model with separate timescales for different game aspects"""
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

        # Separate by update frequency
        fast_features = jnp.concatenate([
            flat_state[..., :5],      # Player position
            flat_state[..., 170:175]  # Missile position
        ], axis=-1)
        
        medium_features = flat_state[..., 5:170]     # Entity positions
        slow_features = flat_state[..., 175:179]     # Game state
        
        # Fast timescale processing (every frame)
        fast_input = jnp.concatenate([fast_features, action_one_hot], axis=-1)
        fast_processed = hk.Linear(128)(fast_input)
        fast_processed = jax.nn.gelu(fast_processed)
        
        # Medium timescale processing 
        medium_processed = hk.Linear(256)(medium_features)
        medium_processed = jax.nn.gelu(medium_processed)
        medium_processed = hk.Linear(128)(medium_processed)
        
        # Slow timescale processing
        slow_processed = hk.Linear(64)(slow_features)
        slow_processed = jax.nn.gelu(slow_processed)
        
        # Hierarchical LSTM architecture
        fast_lstm = hk.LSTM(256)
        medium_lstm = hk.LSTM(128) 
        slow_lstm = hk.LSTM(64)
        
        if lstm_state is None:
            fast_state = fast_lstm.initial_state(batch_size)
            medium_state = medium_lstm.initial_state(batch_size)
            slow_state = slow_lstm.initial_state(batch_size)
        else:
            fast_state, medium_state, slow_state = lstm_state
        
        # Process at different timescales
        fast_out, new_fast_state = fast_lstm(fast_processed, fast_state)
        medium_out, new_medium_state = medium_lstm(medium_processed, medium_state)
        slow_out, new_slow_state = slow_lstm(slow_processed, slow_state)
        
        # Combine hierarchical outputs
        combined = jnp.concatenate([fast_out, medium_out, slow_out], axis=-1)
        combined = hk.Linear(256)(combined)
        combined = jax.nn.gelu(combined)
        
        # Separate prediction heads
        fast_pred = hk.Linear(10)(fast_out)     # Player + missile
        medium_pred = hk.Linear(165)(medium_out) # Entities
        slow_pred = hk.Linear(4)(slow_out)       # Game state
        
        # Combine predictions
        full_prediction = jnp.concatenate([
            fast_pred[..., :5],      # Player
            medium_pred,             # Entities  
            fast_pred[..., 5:],      # Missile
            slow_pred               # Game state
        ], axis=-1)
        
        # Timescale-appropriate residuals
        prediction = full_prediction + jnp.concatenate([
            jnp.full((5,), 0.3),    # Low residual for fast features
            jnp.full((165,), 0.9),  # High residual for medium features
            jnp.full((5,), 0.1),    # Very low residual for missiles
            jnp.full((4,), 0.95)    # Very high residual for slow features
        ]) * flat_state
        
        new_lstm_state = (new_fast_state, new_medium_state, new_slow_state)
        return prediction, new_lstm_state

    return hk.transform(forward)

def V7_Entity_Aware_LSTM():
    """Model that explicitly handles entity lifecycles and interactions"""
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

        # Process entities by type with awareness of their properties
        player_features = flat_state[..., :5]
        
        # Reshape entities into (entity_type, entity_id, properties)
        sharks = flat_state[..., 5:65].reshape(batch_size, 12, 5)    # 12 sharks, 5 props each
        subs = flat_state[..., 65:125].reshape(batch_size, 12, 5)    # 12 subs, 5 props each  
        divers = flat_state[..., 125:145].reshape(batch_size, 4, 5)   # 4 divers, 5 props each
        missiles = flat_state[..., 145:165].reshape(batch_size, 4, 5) # 4 enemy missiles, 5 props each
        
        surface_sub = flat_state[..., 165:170]
        player_missile = flat_state[..., 170:175]
        game_state = flat_state[..., 175:179]
        
        # Entity-type-specific processing
        def process_entity_group(entities, entity_type_embedding):
            # entities: (batch, num_entities, 5)
            batch_s, num_entities, props = entities.shape
            
            # Add entity type embedding
            entities_flat = entities.reshape(batch_s, -1)
            type_embed = jnp.tile(entity_type_embedding, (batch_s, num_entities))
            
            combined = jnp.concatenate([entities_flat, type_embed], axis=-1)
            processed = hk.Linear(64)(combined)
            processed = jax.nn.gelu(processed)
            return processed
        
        # Process each entity type
        shark_processed = process_entity_group(sharks, jnp.array([1, 0, 0, 0]))
        sub_processed = process_entity_group(subs, jnp.array([0, 1, 0, 0]))
        diver_processed = process_entity_group(divers, jnp.array([0, 0, 1, 0]))
        emissile_processed = process_entity_group(missiles, jnp.array([0, 0, 0, 1]))
        
        # Player and special entities
        player_processed = hk.Linear(32)(jnp.concatenate([player_features, action_one_hot], axis=-1))
        surface_processed = hk.Linear(16)(surface_sub)
        pmissile_processed = hk.Linear(16)(jnp.concatenate([player_missile, action_one_hot[:, 10:]], axis=-1))
        game_processed = hk.Linear(16)(game_state)
        
        # Combine all processed features
        all_processed = jnp.concatenate([
            player_processed, shark_processed, sub_processed, diver_processed,
            emissile_processed, surface_processed, pmissile_processed, game_processed
        ], axis=-1)
        
        # Main processing
        x = hk.Linear(512)(all_processed)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(256)(x)
        x = jax.nn.gelu(x)
        
        # LSTM for temporal modeling
        lstm = hk.LSTM(512)
        
        if lstm_state is None:
            lstm_state = lstm.initial_state(batch_size)
            
        lstm_out, new_lstm_state = lstm(x, lstm_state)
        
        # Entity-aware prediction heads
        player_head = hk.Linear(5)(lstm_out)
        entity_head = hk.Linear(160)(lstm_out)  # All entities 5:165
        special_head = hk.Linear(10)(lstm_out)  # Surface sub + player missile
        game_head = hk.Linear(4)(lstm_out)
        
        full_prediction = jnp.concatenate([
            player_head, entity_head, special_head, game_head
        ], axis=-1)
        
        # Entity-aware residuals
        residual_weights = jnp.concatenate([
            jnp.full((5,), 0.6),    # Player
            jnp.full((160,), 0.85), # Entities
            jnp.full((10,), 0.4),   # Special entities
            jnp.full((4,), 0.9)     # Game state
        ])
        
        prediction = full_prediction + residual_weights * flat_state
        
        # Entity lifecycle constraints
        # Ensure active flags are binary
        for start_idx in [9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64]:  # Active flags for sharks
            prediction = prediction.at[..., start_idx].set(
                jnp.round(jnp.clip(prediction[..., start_idx], 0, 1))
            )
        
        return prediction, new_lstm_state

    return hk.transform(forward)






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





def V2_Enhanced_LSTM():
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

        # Separate static and dynamic features (same structure)
        static_features = jnp.concatenate([flat_state[..., :170], flat_state[..., 179:]], axis=-1)
        dynamic_features = flat_state[..., 170:179]
        
        # Enhanced static feature processing - increased capacity
        static_branch = hk.Linear(512)(static_features)  # 256 -> 512
        static_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        static_branch = hk.Linear(256)(static_branch)  # 128 -> 256
        static_branch = jax.nn.gelu(static_branch)
        # Add extra layer for static features
        static_branch = hk.Linear(128)(static_branch)
        static_branch = jax.nn.gelu(static_branch)
        
        # Enhanced dynamic feature processing - increased capacity  
        dynamic_branch = hk.Linear(256)(dynamic_features)  # 128 -> 256
        dynamic_branch = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        dynamic_branch = hk.Linear(128)(dynamic_branch)  # 64 -> 128
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        # Add extra layer for dynamic features
        dynamic_branch = hk.Linear(64)(dynamic_branch)
        dynamic_branch = jax.nn.gelu(dynamic_branch)
        
        # Enhanced action processing - increased capacity
        action_features = hk.Linear(128)(action_one_hot)  # 64 -> 128
        action_features = jax.nn.gelu(action_features)
        action_features = hk.Linear(64)(action_features)  # 32 -> 64
        action_features = jax.nn.gelu(action_features)
        # Add extra layer for action features
        action_features = hk.Linear(32)(action_features)
        action_features = jax.nn.gelu(action_features)
        
        # Combine all features
        combined = jnp.concatenate([static_branch, dynamic_branch, action_features], axis=1)
        
        # Enhanced feature mixing with more capacity
        x = hk.Linear(1024)(combined)  # 512 -> 1024
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(512)(x)  # 256 -> 512
        x = jax.nn.gelu(x)
        # Add extra mixing layer
        x = hk.Linear(256)(x)
        x = jax.nn.gelu(x)
        
        # Enhanced multi-layer LSTM with increased capacity
        lstm1 = hk.LSTM(1024)  # 512 -> 1024
        lstm2 = hk.LSTM(512)   # 256 -> 512
        
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
        lstm2_out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(lstm2_out)
        
        # Enhanced prediction heads with more layers
        # Static head with increased capacity
        static_head = hk.Linear(512)(lstm2_out)  # 256 -> 512
        static_head = jax.nn.gelu(static_head)
        static_head = hk.Linear(256)(static_head)  # Add extra layer
        static_head = jax.nn.gelu(static_head)
        static_pred = hk.Linear(len(static_features[0]))(static_head)
        
        # Dynamic head with increased capacity
        dynamic_head = hk.Linear(256)(lstm2_out)  # 128 -> 256
        dynamic_head = jax.nn.gelu(dynamic_head)
        dynamic_head = hk.Linear(128)(dynamic_head)  # Add extra layer
        dynamic_head = jax.nn.gelu(dynamic_head)
        dynamic_pred = hk.Linear(9)(dynamic_head)  # 170:179 = 9 features
        
        # Combine predictions (same structure)
        full_prediction = jnp.concatenate([
            static_pred[..., :170], 
            dynamic_pred, 
            static_pred[..., 170:]
        ], axis=-1)
        
        # Apply different residual connections (same structure but tuned values)
        # Slightly stronger residual for static features, weaker for dynamic
        static_residual = 0.85   # 0.8 -> 0.85 (more stable)
        dynamic_residual = 0.2   # 0.3 -> 0.2 (allow more learning)
        
        residual_weights = jnp.concatenate([
            jnp.full((170,), static_residual),
            jnp.full((9,), dynamic_residual),
            jnp.full((flat_state.shape[-1] - 179,), static_residual)
        ])
        
        prediction = full_prediction + residual_weights * flat_state
        
        # Tighter stability constraints for dynamic features
        prediction = prediction.at[..., 170:179].set(
            jnp.clip(prediction[..., 170:179], -30, 150)  # Tighter: -50,200 -> -30,150
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
