import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
import pygame
import time
import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle
from jaxatari.games.jax_seaquest import SeaquestRenderer, JaxSeaquest
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper

VERBOSE = True
model = None


def render_trajectory(
    states, num_frames: int = 100, render_scale: int = 3, delay: int = 50
):
    """
    Render a trajectory of states in a single window.
    Args:
        states: PyTree containing the collected states to visualize
        num_frames: Maximum number of frames to show
        render_scale: Scaling factor for rendering
        delay: Milliseconds to delay between frames
    """
    import pygame
    import time

    pygame.init()
    renderer = SeaquestRenderer()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("State Trajectory Visualization")
    surface = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont(None, 24)
    if isinstance(states, dict) or hasattr(states, "env_state"):
        total_frames = 1
    else:
        first_field = jax.tree_util.tree_leaves(states)[0]
        total_frames = first_field.shape[0] if hasattr(first_field, "shape") else 1
    frames_to_show = min(total_frames, num_frames)
    print(f"Rendering trajectory with {frames_to_show} frames...")
    running = True
    frame_idx = 0
    while running and frame_idx < frames_to_show:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if total_frames > 1:
            current_state = jax.tree.map(
                lambda x: (
                    x[frame_idx]
                    if hasattr(x, "shape") and x.shape[0] > frame_idx
                    else x
                ),
                states,
            )
        else:
            current_state = states
        try:
            raster = renderer.render(current_state)
            img = np.array(raster * 255, dtype=np.uint8)
            pygame.surfarray.blit_array(surface, img)
            screen.fill((0, 0, 0))
            scaled_surface = pygame.transform.scale(
                surface, (WIDTH * render_scale, HEIGHT * render_scale)
            )
            screen.blit(scaled_surface, (0, 0))
            frame_text = font.render(
                f"Frame: {frame_idx + 1}/{frames_to_show}", True, (255, 255, 255)
            )
            screen.blit(frame_text, (10, 10))
            pygame.display.flip()
        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            frame_idx += 1
            continue
        pygame.time.wait(delay)
        frame_idx += 1
    if running:
        pygame.time.wait(1000)
    pygame.quit()
    print(f"Rendered {frame_idx} frames from trajectory")


def flatten_state(
    state, single_state: bool = False, is_list=False
) -> Tuple[jnp.ndarray, Any]:
    """
    Flatten the state PyTree into a single array.
    This is useful for debugging and visualization.
    """
    # check whether it is a single state or a batch of states

    if type(state) == list:
        flat_states = []
        for s in state:
            flat_state, _ = jax.flatten_util.ravel_pytree(s)
            flat_states.append(flat_state)
        flat_states = jnp.stack(flat_states, axis=0)  # Shape: (1626, 160)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener
    batch_shape = state.player_x.shape[0]

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)
    return flat_state, unflattener


def build_world_model():
    def forward(state, action, lstm_state=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
        else:
            flat_state_full = state

        flat_state = flat_state_full[..., :-2]
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


def collect_experience_sequential(
    env, num_episodes: int = 1, max_steps_per_episode: int = 1000, episodic_life: bool = False
):
    """Collect experience data sequentially to ensure proper transitions."""
    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    boundaries = []

    dead = False
    total_steps = 0
    rng = jax.random.PRNGKey(42)

    for episode in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        _, state = env.reset(reset_key)

        for step in range(max_steps_per_episode):
            current_state = jax.tree.map(lambda x: x, state.env_state)

            # Choose a random action
            rng, action_key = jax.random.split(rng)
            action = (
                5
                if jax.random.uniform(action_key) < 0.2
                else jax.random.randint(action_key, (), 0, 18)
            )

            # Take a step in the environment
            rng, step_key = jax.random.split(rng)
            _, next_state, reward, done, _ = env.step(step_key, state, action)

            next_state_repr = jax.tree.map(lambda x: x, next_state.env_state)

            # Store the transition
            states.append(current_state)
            actions.append(action)
            next_states.append(next_state_repr)
            rewards.append(reward)
            dones.append(done)

            # If episode is done, reset the environment

            
            if not episodic_life: 
                if current_state.death_counter > 0 and not dead:
                    dead = True
                if not current_state.death_counter > 0 and dead:
                    dead = False
                    boundaries.append(total_steps)

            if done:
                print(f"Episode {episode+1} done after {step+1} steps")

                if episodic_life: 
                    if len(boundaries) == 0:
                        boundaries.append(step)
                    else:
                        boundaries.append(boundaries[-1] + step + 1)
                break
               

            # Update state for the next step
            state = next_state
            total_steps += 1

    # Convert to JAX arrays (but don't flatten the structure yet)
    # Use tree_map to maintain structure with jnp arrays

    # Stack states correctly to form batch
    # Step 1: Stack states across time
    batched_states = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

    # Step 2: Flatten into a single vector per state
    flat_states, unflattener = flatten_state(batched_states, single_state=False)

    batched_next_states = jax.tree.map(lambda *xs: jnp.stack(xs), *next_states)
    flat_next_states, _ = flatten_state(batched_next_states, single_state=False)

    actions_array = jnp.array(actions)
    rewards_array = jnp.array(rewards)
    dones_array = jnp.array(dones)

    print("Boundaries:")
    print(boundaries)

    return (
        flatten_state(states, is_list=True),
        actions_array,
        states,
        rewards_array,
        dones_array,
        boundaries,
    )





def train_world_model(
    states,
    actions,
    next_states,
    rewards,
    learning_rate=3e-4,
    batch_size=4,
    num_epochs=600,
    sequence_length=32,
    episode_boundaries=None,
):
    # Calculate normalization statistics from the flattened states
    state_mean = jnp.mean(states, axis=0)
    state_std = jnp.std(states, axis=0) + 1e-8

    # Store normalization stats for later use
    normalization_stats = {"mean": state_mean, "std": state_std}

    # Normalize states and next_states
    normalized_states = (states - state_mean) / state_std
    normalized_next_states = (next_states - state_mean) / state_std

    # Create sequential batches that respect episode boundaries
    def create_sequential_batches(batch_size=32):
        """
        Create batches of sequential data for training
        Args:
            batch_size: Number of sequences per batch
        Returns:
            List of batches, each containing (state_batch, action_batch, next_state_batch)
            where each has shape (batch_size, seq_len, feature_dim)
        """
        sequences = []
        
        # First, collect all sequences
        for i in range(len(episode_boundaries)-1):
            if i == 0:
                start_idx = 0
                end_idx = episode_boundaries[0]
            else:
                start_idx = episode_boundaries[i - 1]
                end_idx = episode_boundaries[i]
            
            # Create sequences within this episode
            for j in range(0, end_idx-start_idx-sequence_length+1): # Iterate over every possible starting point
            # for j in range(0, end_idx-start_idx-sequence_length+1, sequence_length // 4):
                if start_idx + j + sequence_length > end_idx:
                    break
                    
                sequences.append((
                    normalized_states[start_idx + j : start_idx + j + sequence_length],
                    actions[start_idx + j : start_idx + j + sequence_length], 
                    normalized_next_states[start_idx + j : start_idx + j + sequence_length]
                ))
        
        return sequences

    
    # Create sequential batches
    batches = create_sequential_batches()



    model = build_world_model()
    optimizer = optax.adam(learning_rate=1e-4)

    rng = jax.random.PRNGKey(42)
    dummy_state = normalized_states[:1]
    dummy_action = actions[:1]
    params = model.init(rng, dummy_state, dummy_action, None)
    opt_state = optimizer.init(params)

    def loss_function(params, state_batch, action_batch, next_state_batch):
        """
        Compute loss over sequences maintaining LSTM state
        Args:
            state_batch: (batch_size, seq_len, state_dim)
            action_batch: (batch_size, seq_len)
            next_state_batch: (batch_size, seq_len, state_dim)
        """
        print(state_batch.shape)
        seq_len, state_dim = state_batch.shape
        total_loss = 0.0

        # Process each sequence in the batch

        lstm_state = None  # Start with fresh state for each sequence
        sequence_loss = 0.0

        for t in range(seq_len):
            current_state = state_batch[
                 t : t + 1
            ]  # Keep batch dimension
            current_action = action_batch[ t : t + 1]
            target_next_state = next_state_batch[
                t, :-2
            ]  # Remove last 2 features

            # Forward pass
            pred_next_state, lstm_state = model.apply(
                params, None, current_state, current_action, lstm_state
            )

            # Compute loss for this timestep
            step_loss = jnp.mean(
                (target_next_state - pred_next_state.squeeze()) ** 2
            )
            sequence_loss += step_loss

        total_loss += sequence_loss / seq_len

        return total_loss

    @jax.jit
    def update_step(params, opt_state, state_batch, action_batch, next_state_batch):
        loss, grads = jax.value_and_grad(loss_function)(
            params, state_batch, action_batch, next_state_batch
        )

        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    

    for epoch in range(num_epochs):
        losses = []
        for batch in batches:
            state_batch, action_batch, next_state_batch = batch
            params, opt_state, loss = update_step(
                params, opt_state, state_batch, action_batch, next_state_batch
            )
            losses.append(loss)
        epoch_loss = jnp.mean(jnp.array(losses))
        # if VERBOSE and (epoch + 1) % (num_epochs / 10) == 0:
        #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        if VERBOSE and (epoch + 1) % (num_epochs/10) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    return params, {
        "final_loss": epoch_loss,
        "normalization_stats": normalization_stats,
    }


def compare_real_vs_model(
    num_steps: int = 10,
    render_scale: int = 2,
    states=None,
    actions=None,
    normalization_stats=None,
    steps_into_future: int = 10,
    clock_speed = 5
):
    
    # states = states[100:]
    # actions = actions[100:]
    
    # Add debugging to understand the model input/output formats
    def debug_states(step, real_state, pred_state):
        if step % 1 == 0:
            print(f"Step {step} debugging:")

            # Check ranges and statistics of the data
            print(
                f"Real state min/max/mean: {jnp.min(real_state):.2f}/{jnp.max(real_state):.2f}/{jnp.mean(real_state):.2f}"
            )
            print(
                f"Model state min/max/mean: {jnp.min(pred_state):.2f}/{jnp.max(pred_state):.2f}/{jnp.mean(pred_state):.2f}"
            )

            error = jnp.mean((real_state - pred_state) ** 2)
            print(f"Step {step_count}, Unnormalized Error: {error:.2f}")

    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]

    normalized_states = (states - state_mean) / state_std
    normalized_next_states = (next_states - state_mean) / state_std

    base_game = JaxSeaquest()
    real_env = AtariWrapper(
        base_game, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    renderer = SeaquestRenderer()
    model_path = "world_model_LSTM.pkl"
    if not os.path.exists(model_path):
        print(f"Error: World model not found at {model_path}")
        return
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        dynamics_params = model_data["dynamics_params"]
        scale_factor = model_data.get("scale_factor", 1 / 1)
        normalization_stats = model_data.get("normalization_stats", None)
    world_model = build_world_model()

    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(
        "Real Environment vs World Model (AtariWrapper Frame Stack)"
    )

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    running = True
    # step_count = 120 # Start from a later step to avoid initial noise
    step_count = 0
    clock = pygame.time.Clock()

    # This part is only here to get the real_start and for the unflattener
    base_game = JaxSeaquest()
    real_env = AtariWrapper(
        base_game, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    rng = jax.random.PRNGKey(int(time.time()))
    rng, reset_key = jax.random.split(rng)
    real_obs, real_state = real_env.reset(reset_key)

    first_state_flat = states[0 + step_count]
    _, unflattener = flatten_state(real_state.env_state, single_state=True)
    first_state_raw = unflattener(first_state_flat)
    real_state = real_state.replace(env_state=first_state_raw)
    model_state = real_state  # Start identical

    # Initialize LSTM state for model predictions
    lstm_state = None
    lsmt_real_state = None

    while running and step_count < min(num_steps, len(states) - 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Use the saved action
        action = actions[step_count]

        # Use the saved next state directly instead of environment stepping
        next_real_state_flat = states[step_count + 1]
        real_state = real_state.replace(env_state=unflattener(next_real_state_flat))

        # Get the flattened current state for the model
        flattened_model_state, _ = flatten_state(
            model_state.env_state, single_state=True
        )

        # Apply model prediction with normalization and LSTM state
        normalized_flattened_model_state = (
            flattened_model_state - state_mean
        ) / state_std




        # Use the stateful model (returns both prediction and new LSTM state)
        normalized_model_state_flattened, lstm_state = world_model.apply(
            dynamics_params,
            None,
            normalized_flattened_model_state,
            jnp.array([action]),
            lstm_state,
        )


        




        model_state_flattened = (
            normalized_model_state_flattened * state_std[:-2] + state_mean[:-2]
        )

        debug_states(step_count, next_real_state_flat[:-2], model_state_flattened)

        # Complete model state with additional fields
        model_state_flattened = jnp.concatenate(
            [model_state_flattened, jnp.zeros((model_state_flattened.shape[0], 2))],
            axis=-1,
        )
        model_state_flattened_1d = model_state_flattened.reshape(-1)

        # Update model state
        model_state = model_state.replace(
            env_state=unflattener(model_state_flattened_1d)
        )
        reconstructed_env_state = unflattener(model_state_flattened_1d)
        reconstructed_env_state = reconstructed_env_state._replace(
            step_counter=real_state.env_state.step_counter,
            rng_key=real_state.env_state.rng_key,
        )
        model_state = model_state.replace(env_state=reconstructed_env_state)

        if VERBOSE and step_count % 100 == 0:
            print(f"Step {step_count}: Real vs Model state comparison")
        real_base_state = real_state.env_state
        model_base_state = model_state.env_state

        # Rendering stuff -------------------------------------------------------
        real_raster = renderer.render(real_base_state)
        real_img = np.array(real_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(real_surface, real_img)
        model_raster = renderer.render(model_base_state)
        model_img = np.array(model_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(model_surface, model_img)
        screen.fill((0, 0, 0))
        scaled_real = pygame.transform.scale(
            real_surface, (WIDTH * render_scale, HEIGHT * render_scale)
        )
        screen.blit(scaled_real, (0, 0))
        scaled_model = pygame.transform.scale(
            model_surface, (WIDTH * render_scale, HEIGHT * render_scale)
        )
        screen.blit(scaled_model, (WIDTH * render_scale + 20, 0))
        font = pygame.font.SysFont(None, 24)
        real_text = font.render("Real Environment", True, (255, 255, 255))
        model_text = font.render("World Model (4 Frames)", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        pygame.display.flip()







        #seperate prediction just to have the lstm state for the current real trajectory at all times
        flattened_real_state, unflattener = flatten_state(
            real_state.env_state, single_state=True
        )
        normalized_real_state_flat = (
            flattened_real_state - state_mean
        ) / state_std
        _, lsmt_real_state = world_model.apply(
            dynamics_params,
            None,
            normalized_real_state_flat,
            jnp.array([action]),
            lsmt_real_state,
        )

        if step_count % steps_into_future == 0:
            model_state = real_state
            lstm_state = lsmt_real_state



        step_count += 1
        # print(states[step_count][:-2])
        clock.tick(clock_speed)
        # Rendering stuff end -------------------------------------------------------

    pygame.quit()
    print("Comparison completed")


if __name__ == "__main__":



    game = JaxSeaquest()
    env = AtariWrapper(
        game, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    env = FlattenObservationWrapper(env)

    save_path = "world_model_LSTM.pkl"
    experience_data_path = "experience_data.pkl"
    model = build_world_model()
    normalization_stats = None

    # print(next_states[300][:-2])
    # pred = model.apply(dynamics_params, None, states[300], actions[300])
    # print(pred)

    # print(((next_states[300][:-2] - pred) ** 2))

    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, "rb") as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data["dynamics_params"]
            normalization_stats = saved_data.get("normalization_stats", None)
    else:
        print("No existing model found. Training a new model...")

        # Define a file path for the experience data

        # Check if experience data file exists
        if os.path.exists(experience_data_path):
            print(f"Loading existing experience data from {experience_data_path}...")
            with open(experience_data_path, "rb") as f:
                saved_data = pickle.load(f)
                states = saved_data["states"]
                actions = saved_data["actions"]
                next_states = saved_data["next_states"]
                rewards = saved_data["rewards"]
                boundaries = saved_data["boundaries"]
        else:
            print(
                "No existing experience data found. Collecting new experience data..."
            )
            # Collect experience data (AtariWrapper handles frame stacking automatically)
            flattened_states, actions, _, rewards, _, boundaries = (
                collect_experience_sequential(
                    env, num_episodes=5, max_steps_per_episode=1000
                )
            )
            next_states = flattened_states[
                1:
            ]  # Next states are just the next frame in the sequence
            states = flattened_states[:-1]  # Current states are all but the last frame

            # render_trajectory(states, num_frames=1000, render_scale=2, delay=10)
            # I want to check whether the next_state is equal to the current state + 1
            print(states.shape)
            print(next_states.shape)

            if VERBOSE:
                print("Checking if next_state is equal to current state + 1...")
                for i in range(100):
                    if not jnp.allclose(states[i + 1][:-2], next_states[i][:-2]):
                        print(
                            f"Mismatch at index {i}: {states[i][:-2]} != {next_states[i][:-2]}"
                        )
                        print(states[i + 1][:-2])
                        print(next_states[i][:-2])
                        print(states[i + 1][:-2] - next_states[i][:-2])
                        exit(1)
                print("All states match the expected transition.")

            # Save the collected experience data
            with open(experience_data_path, "wb") as f:
                pickle.dump(
                    {
                        "states": states,
                        "actions": actions,
                        "next_states": next_states,
                        "rewards": rewards,
                        "boundaries": boundaries,
                    },
                    f,
                )
            print(f"Experience data saved to {experience_data_path}")
            print(boundaries)

        # Train world model
        dynamics_params, training_info = train_world_model(
            states,
            actions,
            next_states,
            rewards,
            episode_boundaries=boundaries,
        )
        normalization_stats = training_info.get("normalization_stats", None)

        # Save the model and scaling factor
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "dynamics_params": dynamics_params,
                    "normalization_stats": training_info.get(
                        "normalization_stats", None
                    ),
                },
                f,
            )
        print(f"Model saved to {save_path}")

    states, actions, boundaries = None, None, None
    if os.path.exists(experience_data_path):
        with open(experience_data_path, "rb") as f:
            saved_data = pickle.load(f)
            states = saved_data["states"]
            actions = saved_data["actions"]
            next_states = saved_data["next_states"]
            rewards = saved_data["rewards"]
            boundaries = saved_data["boundaries"]

    world_model = build_world_model()
    losses = []

    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]

    lstm_state = None

    normalized_states = (states - state_mean) / state_std
    normalized_next_states = (next_states - state_mean) / state_std

    # for i in range(len(states)):
    #     prediction, lstm_state = world_model.apply(
    #             dynamics_params, None, normalized_states[0+i], jnp.array([actions[0+i]]), lstm_state
    #         )
    #     prediction
        
    #     loss = jnp.mean((prediction - normalized_next_states[0+i][:-2])**2)
    #     # print(loss)
    #     losses.append(loss)
    #     if loss > 0.01:
    #         # print('-----------------------------------------------------------------------------------------------------------------')
    #         print(f"Step {i}:")
    #         print(f"Loss : {loss}")
    #         print("Indexes where difference > 3:")
    #         for j in range(len(prediction[0])):
    #             if jnp.abs(prediction[0][j] - normalized_states[1+i][j]) > 1:
    #                 print(f"Index {j}: {prediction[0][j]} vs {normalized_states[1+i][j]}")
    #         # print(f"Difference: {prediction - normalized_states[1+i][:-2]}")
    #         # print(f"State {normalized_states[i]}")
    #         # print("Negative values in state:")
    #         # print(jnp.any(normalized_states[i][:-2] < -1))
    #         # print(f"Prediction: {prediction}")
    #         # print(f"Actual Next State {normalized_states[i+1]}")
    #         # print all indexes where the difference it greater than 10

            
    #     if i == 2048:
    #         break
    #     # print(f"Loss : {jnp.mean((prediction - states[1+i][:-2])**2)}")

    # print(f"Average loss: {jnp.mean(jnp.array(losses))}")

    # exit()
    compare_real_vs_model(
        num_steps=5000,
        render_scale=6,
        states=states,
        actions=actions,
        normalization_stats=normalization_stats,
    )
