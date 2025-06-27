import jax.numpy as jnp
import chex
from typing import NamedTuple
import jax

from jaxatari.games.jax_seaquest import SeaquestObservation, SeaquestState, SpawnState

def flat_observation_to_state(obs: SeaquestObservation, unflattener, rng_key: chex.PRNGKey = None) -> SeaquestState:
    """
    Convert a flattened observation to SeaquestState by first unflattening it.
    
    Args:
        obs: The flattened observation array to convert
        unflattener: Function to convert flat observation to structured observation
        rng_key: RNG key to use for the state (since obs doesn't contain this)
    
    Returns:
        SeaquestState constructed from the observation
    """

    obs = unflattener(obs)
    
    # Add debug 
    
    
    
    

    # just generate a rng_key if not provided
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    # Extract player position and direction from EntityPosition
    player_x = obs.player.x[0] if hasattr(obs.player.x, '__len__') and len(obs.player.x) > 0 else obs.player.x
    player_y = obs.player.y[0] if hasattr(obs.player.y, '__len__') and len(obs.player.y) > 0 else obs.player.y
    # We'll need to infer direction - this might need adjustment based on your logic
    player_direction = jnp.array(0)  # Default to right, you may need to track this differently
    
    
    
    # Extract basic game metrics - ensure they are scalars
    oxygen = obs.oxygen_level[0] if hasattr(obs.oxygen_level, '__len__') and len(obs.oxygen_level) > 0 else obs.oxygen_level
    divers_collected = obs.collected_divers[0] if hasattr(obs.collected_divers, '__len__') and len(obs.collected_divers) > 0 else obs.collected_divers
    score = obs.player_score[0] if hasattr(obs.player_score, '__len__') and len(obs.player_score) > 0 else obs.player_score
    lives = obs.lives[0] if hasattr(obs.lives, '__len__') and len(obs.lives) > 0 else obs.lives
    
    
    
    
    # Create default spawn state with proper initialization
    spawn_state = SpawnState(
        difficulty=jnp.array(0),
        lane_dependent_pattern=jnp.zeros(4, dtype=jnp.int32),
        to_be_spawned=jnp.zeros(12, dtype=jnp.int32),
        survived=jnp.zeros(12, dtype=jnp.int32),
        prev_sub=jnp.ones(4, dtype=jnp.int32),
        spawn_timers=jnp.array([277, 277, 277, 277 + 60], dtype=jnp.int32),
        diver_array=jnp.array([1, 1, 0, 0], dtype=jnp.int32),
        lane_directions=jnp.array([0, 0, 0, 1], dtype=jnp.int32)  # First wave directions
    )
    
    # Convert entity arrays to position arrays
    # Entity arrays come as (1, N, 5) from unflattener, need to squeeze and convert to (N, 3)
    # State expects (N, 3) - we'll take [x, y, direction] where direction is inferred from active
    
    def process_entity_array(entity_array, expected_rows, name):
        """Process entity array: squeeze batch dim and convert to position format"""
        
        
        # Remove batch dimension if present
        if len(entity_array.shape) == 3 and entity_array.shape[0] == 1:
            entity_array = entity_array.squeeze(0)  # Remove batch dimension
            
        
        # Convert to (N, 3) format: [x, y, direction]
        positions = jnp.column_stack([
            entity_array[:, 0],  # x
            entity_array[:, 1],  # y  
            entity_array[:, 4].astype(jnp.float32)  # direction (using active as placeholder)
        ])
        
        return positions
    
    # Process each entity type
    diver_positions = process_entity_array(obs.divers, 4, "divers")
    shark_positions = process_entity_array(obs.sharks, 12, "sharks") 
    sub_positions = process_entity_array(obs.submarines, 12, "submarines")
    enemy_missile_positions = process_entity_array(obs.enemy_missiles, 4, "enemy_missiles")
    
    # Surface submarine: Handle EntityPosition with array values
    
    
    
    # Extract scalar values from EntityPosition arrays
    surface_x = obs.surface_submarine.x[0] if hasattr(obs.surface_submarine.x, '__len__') else obs.surface_submarine.x
    surface_y = obs.surface_submarine.y[0] if hasattr(obs.surface_submarine.y, '__len__') else obs.surface_submarine.y
    surface_active = obs.surface_submarine.active[0] if hasattr(obs.surface_submarine.active, '__len__') else obs.surface_submarine.active
    
    surface_sub_position = jnp.array([
        float(surface_x), 
        float(surface_y), 
        float(jnp.where(surface_active, 1.0, 0.0))
    ]).astype(jnp.float32)
    
    
    # Player missile: Handle EntityPosition with array values
    
    
    # Extract scalar values from EntityPosition arrays
    missile_x = obs.player_missile.x[0] if hasattr(obs.player_missile.x, '__len__') else obs.player_missile.x
    missile_y = obs.player_missile.y[0] if hasattr(obs.player_missile.y, '__len__') else obs.player_missile.y
    missile_active = obs.player_missile.active[0] if hasattr(obs.player_missile.active, '__len__') else obs.player_missile.active
    
    player_missile_position = jnp.array([
        float(missile_x), 
        float(missile_y), 
        float(jnp.where(missile_active, 1.0, 0.0))
    ]).astype(jnp.float32)
    
    
    # Default values for state-only fields that aren't in observation
    step_counter = jnp.array(0)  # You might want to track this externally
    just_surfaced = jnp.array(False)  # Default to False
    successful_rescues = jnp.array(0)  # Could be inferred from score or tracked separately
    death_counter = jnp.array(0)  # Default to 0
    
    
    
    
    
    
    
    
    
    return SeaquestState(
        player_x=jnp.array(player_x),
        player_y=jnp.array(player_y),
        player_direction=player_direction,
        oxygen=jnp.array(oxygen),
        divers_collected=jnp.array(divers_collected),
        score=jnp.array(score),
        lives=jnp.array(lives),
        spawn_state=spawn_state,
        diver_positions=diver_positions,
        shark_positions=shark_positions,
        sub_positions=sub_positions,
        enemy_missile_positions=enemy_missile_positions,
        surface_sub_position=surface_sub_position,
        player_missile_position=player_missile_position,
        step_counter=step_counter,
        just_surfaced=just_surfaced,
        successful_rescues=successful_rescues,
        death_counter=death_counter,
        rng_key=rng_key
    )