import jax.numpy as jnp
import chex
from typing import NamedTuple
import jax

from jaxatari.games.jax_seaquest import SeaquestObservation, SeaquestState, SpawnState

def flat_observation_to_state(obs: SeaquestObservation, unflattener, rng_key: chex.PRNGKey = None, frame_stack_size = 4) -> SeaquestState:

    if frame_stack_size > 1:
        obs = unflattener(obs)
        obs = jax.tree_util.tree_map(lambda x: x[-1], obs)

    
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Extract player position and direction - now scalars after tree_map
        player_x = obs.player.x
        player_y = obs.player.y
        player_direction = obs.player.o
        
        # Extract basic game metrics - now scalars after tree_map
        oxygen = obs.oxygen_level
        divers_collected = obs.collected_divers
        score = obs.player_score
        lives = obs.lives
        
        # Create default spawn state with proper initialization
        spawn_state = SpawnState(
            difficulty=jnp.array(0),
            lane_dependent_pattern=jnp.zeros(4, dtype=jnp.int32),
            to_be_spawned=jnp.zeros(12, dtype=jnp.int32),
            survived=jnp.zeros(12, dtype=jnp.int32),
            prev_sub=jnp.ones(4, dtype=jnp.int32),
            spawn_timers=jnp.array([277, 277, 277, 277 + 60], dtype=jnp.int32),
            diver_array=jnp.array([1, 1, 0, 0], dtype=jnp.int32),
            lane_directions=jnp.array([0, 0, 0, 1], dtype=jnp.int32)
        )
        
        def process_entity_array(entity_array, name):
            """Process entity array: convert to position format"""
            
            # Convert to (N, 3) format: [x, y, direction]
            positions = jnp.column_stack([
                entity_array[:, 0],  # x
                entity_array[:, 1],  # y  
                entity_array[:, 4].astype(jnp.float32)  # active flag as direction
            ])
            
            return positions
        
        # Process each entity type - now single arrays after tree_map
        diver_positions = process_entity_array(obs.divers, "divers")
        shark_positions = process_entity_array(obs.sharks, "sharks") 
        sub_positions = process_entity_array(obs.submarines, "submarines")
        enemy_missile_positions = process_entity_array(obs.enemy_missiles, "enemy_missiles")
        
        # Surface submarine and player missile - now scalars after tree_map
        surface_sub_position = jnp.array([
            float(obs.surface_submarine.x), 
            float(obs.surface_submarine.y), 
            float(obs.surface_submarine.active)
        ]).astype(jnp.float32)
        
        player_missile_position = jnp.array([
            float(obs.player_missile.x), 
            float(obs.player_missile.y), 
            float(obs.player_missile.active)
        ]).astype(jnp.float32)
        
        # Default values for state-only fields
        step_counter = jnp.array(0)
        just_surfaced = jnp.array(False)
        successful_rescues = jnp.array(0)
        death_counter = jnp.array(0)
        
        return SeaquestState(
            player_x=jnp.array(player_x),
            player_y=jnp.array(player_y),
            player_direction=jnp.array(player_direction),
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
    

    else:
        obs = unflattener(obs)
    

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Extract player position and direction from EntityPosition
        player_x = obs.player.x[0] if hasattr(obs.player.x, '__len__') and len(obs.player.x) > 0 else obs.player.x
        player_y = obs.player.y[0] if hasattr(obs.player.y, '__len__') and len(obs.player.y) > 0 else obs.player.y
        # We'll need to infer direction - this might need adjustment based on your logic
        player_direction = obs.player.o.squeeze()  # Default to right, you may need to track this differently
        
        
        
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




'''
Indices 0-4: Player submarine
Indices 5-64: 12 Sharks (5 elements each)
Indices 65-124: 12 Enemy submarines (5 elements each)
Indices 125-144: 4 Divers (5 elements each)
Indices 145-164: 4 Enemy missiles (5 elements each)
Indices 165-169: Surface submarine
Indices 170-174: Player missile
Indices 175-178: Game state (divers collected, score, lives, oxygen)
'''

# ...existing code...

# ...existing code...

OBSERVATION_INDEX_MAP = {
    # Player EntityPosition (5 elements: x, y, width, height, active)
    0: "player_x",
    1: "player_y", 
    2: "player_width",
    3: "player_height",
    4: "player_active",
    
    # Sharks array (12 sharks × 5 elements each = 60 elements)
    # Shark 0
    5: "shark_0_x", 6: "shark_0_y", 7: "shark_0_width", 8: "shark_0_height", 9: "shark_0_active",
    # Shark 1
    10: "shark_1_x", 11: "shark_1_y", 12: "shark_1_width", 13: "shark_1_height", 14: "shark_1_active",
    # Shark 2
    15: "shark_2_x", 16: "shark_2_y", 17: "shark_2_width", 18: "shark_2_height", 19: "shark_2_active",
    # Shark 3
    20: "shark_3_x", 21: "shark_3_y", 22: "shark_3_width", 23: "shark_3_height", 24: "shark_3_active",
    # Shark 4
    25: "shark_4_x", 26: "shark_4_y", 27: "shark_4_width", 28: "shark_4_height", 29: "shark_4_active",
    # Shark 5
    30: "shark_5_x", 31: "shark_5_y", 32: "shark_5_width", 33: "shark_5_height", 34: "shark_5_active",
    # Shark 6
    35: "shark_6_x", 36: "shark_6_y", 37: "shark_6_width", 38: "shark_6_height", 39: "shark_6_active",
    # Shark 7
    40: "shark_7_x", 41: "shark_7_y", 42: "shark_7_width", 43: "shark_7_height", 44: "shark_7_active",
    # Shark 8
    45: "shark_8_x", 46: "shark_8_y", 47: "shark_8_width", 48: "shark_8_height", 49: "shark_8_active",
    # Shark 9
    50: "shark_9_x", 51: "shark_9_y", 52: "shark_9_width", 53: "shark_9_height", 54: "shark_9_active",
    # Shark 10
    55: "shark_10_x", 56: "shark_10_y", 57: "shark_10_width", 58: "shark_10_height", 59: "shark_10_active",
    # Shark 11
    60: "shark_11_x", 61: "shark_11_y", 62: "shark_11_width", 63: "shark_11_height", 64: "shark_11_active",
    
    # Submarines array (12 subs × 5 elements each = 60 elements)
    # Submarine 0
    65: "sub_0_x", 66: "sub_0_y", 67: "sub_0_width", 68: "sub_0_height", 69: "sub_0_active",
    # Submarine 1
    70: "sub_1_x", 71: "sub_1_y", 72: "sub_1_width", 73: "sub_1_height", 74: "sub_1_active",
    # Submarine 2
    75: "sub_2_x", 76: "sub_2_y", 77: "sub_2_width", 78: "sub_2_height", 79: "sub_2_active",
    # Submarine 3
    80: "sub_3_x", 81: "sub_3_y", 82: "sub_3_width", 83: "sub_3_height", 84: "sub_3_active",
    # Submarine 4
    85: "sub_4_x", 86: "sub_4_y", 87: "sub_4_width", 88: "sub_4_height", 89: "sub_4_active",
    # Submarine 5
    90: "sub_5_x", 91: "sub_5_y", 92: "sub_5_width", 93: "sub_5_height", 94: "sub_5_active",
    # Submarine 6
    95: "sub_6_x", 96: "sub_6_y", 97: "sub_6_width", 98: "sub_6_height", 99: "sub_6_active",
    # Submarine 7
    100: "sub_7_x", 101: "sub_7_y", 102: "sub_7_width", 103: "sub_7_height", 104: "sub_7_active",
    # Submarine 8
    105: "sub_8_x", 106: "sub_8_y", 107: "sub_8_width", 108: "sub_8_height", 109: "sub_8_active",
    # Submarine 9
    110: "sub_9_x", 111: "sub_9_y", 112: "sub_9_width", 113: "sub_9_height", 114: "sub_9_active",
    # Submarine 10
    115: "sub_10_x", 116: "sub_10_y", 117: "sub_10_width", 118: "sub_10_height", 119: "sub_10_active",
    # Submarine 11
    120: "sub_11_x", 121: "sub_11_y", 122: "sub_11_width", 123: "sub_11_height", 124: "sub_11_active",
    
    # Divers array (4 divers × 5 elements each = 20 elements)
    # Diver 0
    125: "diver_0_x", 126: "diver_0_y", 127: "diver_0_width", 128: "diver_0_height", 129: "diver_0_active",
    # Diver 1
    130: "diver_1_x", 131: "diver_1_y", 132: "diver_1_width", 133: "diver_1_height", 134: "diver_1_active",
    # Diver 2
    135: "diver_2_x", 136: "diver_2_y", 137: "diver_2_width", 138: "diver_2_height", 139: "diver_2_active",
    # Diver 3
    140: "diver_3_x", 141: "diver_3_y", 142: "diver_3_width", 143: "diver_3_height", 144: "diver_3_active",
    
    # Enemy missiles array (4 missiles × 5 elements each = 20 elements)
    # Enemy missile 0
    145: "enemy_missile_0_x", 146: "enemy_missile_0_y", 147: "enemy_missile_0_width", 148: "enemy_missile_0_height", 149: "enemy_missile_0_active",
    # Enemy missile 1
    150: "enemy_missile_1_x", 151: "enemy_missile_1_y", 152: "enemy_missile_1_width", 153: "enemy_missile_1_height", 154: "enemy_missile_1_active",
    # Enemy missile 2
    155: "enemy_missile_2_x", 156: "enemy_missile_2_y", 157: "enemy_missile_2_width", 158: "enemy_missile_2_height", 159: "enemy_missile_2_active",
    # Enemy missile 3
    160: "enemy_missile_3_x", 161: "enemy_missile_3_y", 162: "enemy_missile_3_width", 163: "enemy_missile_3_height", 164: "enemy_missile_3_active",
    
    # Surface submarine EntityPosition (5 elements)
    165: "surface_submarine_x",
    166: "surface_submarine_y",
    167: "surface_submarine_width",
    168: "surface_submarine_height",
    169: "surface_submarine_active",
    
    # Player missile EntityPosition (5 elements)
    170: "player_missile_x",
    171: "player_missile_y",
    172: "player_missile_width",
    173: "player_missile_height",
    174: "player_missile_active",
    
    # Game state scalars (4 elements)
    175: "collected_divers",
    176: "player_score",
    177: "lives",
    178: "oxygen_level",
    179: "terminal",  # Add the missing index 179
    
    # Total observation size: 180 elements
}

# ...existing code...
#