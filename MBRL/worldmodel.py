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


def flatten_state(state, single_state: bool = False, is_list = False) -> Tuple[jnp.ndarray, Any]:
    """
    Flatten the state PyTree into a single array.
    This is useful for debugging and visualization.
    """
    #check whether it is a single state or a batch of states

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
    flat_state = flat_state.reshape(
        batch_shape, -1
    )  
    return flat_state, unflattener

def build_world_model():
    def forward(state, action, normalization_stats=None):
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(batch_size, feature_size // batch_size)
            if normalization_stats is not None:
                flat_state_full = (flat_state_full - normalization_stats['mean']) / normalization_stats['std']
        else:
            flat_state_full = state
            if normalization_stats is not None:
                flat_state_full = (flat_state_full - normalization_stats['mean']) / normalization_stats['std']
            
        # Extract the actual state features (excluding last 2 columns)
        flat_state = flat_state_full[..., :-2]
        
        # Apply normalization if normalization stats are provided
        

        # Process action as before
        action_one_hot = jax.nn.one_hot(action, num_classes=18)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1, 18)
            
        inputs = jnp.concatenate([flat_state, action_one_hot], axis=1)
        
        # First layer
        x = hk.Linear(1024)(inputs)
        x = jax.nn.relu(x)
        
        # Residual block 1
        residual = x
        x = hk.Linear(1024)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1024)(x)
        x = x + residual  # Residual connection
        x = jax.nn.relu(x)
        
        # Residual block 2
        residual = x
        x = hk.Linear(1024)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1024)(x)
        x = x + residual  # Residual connection
        x = jax.nn.relu(x)
        
        # Output layer
        output = hk.Linear(flat_state.shape[-1])(x)
        
        # Add clipping to prevent extreme values
        output = jnp.clip(output, -10.0, 10.0)
        
        # If normalization was used, denormalize the output
        if normalization_stats is not None:
            output = output * normalization_stats['std'][:-2] + normalization_stats['mean'][:-2]
            
        return output

    return hk.transform(forward)

def collect_experience_sequential(
    env, num_episodes: int = 1, max_steps_per_episode: int = 1000
):
    """Collect experience data sequentially to ensure proper transitions."""
    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    
    rng = jax.random.PRNGKey(42)
    
    for episode in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        _, state = env.reset(reset_key)
        
        for step in range(max_steps_per_episode):
            current_state = jax.tree.map(lambda x: x, state.env_state)
            
            # Choose a random action
            rng, action_key = jax.random.split(rng)
            action = 5 if jax.random.uniform(action_key) < 0.2 else jax.random.randint(action_key, (), 0, 18)
            
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
            if done:
                print(f"Episode {episode+1} done after {step+1} steps")
                break
                
            # Update state for the next step
            state = next_state
    
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


    return flatten_state(states, is_list=True), actions_array, states, rewards_array, dones_array

def collect_experience(
    env, num_episodes: int = 100, max_steps_per_episode: int = 1000, num_envs: int = 512
) -> Tuple[List, List, List]:

    print(type(env))

    print(f"Collecting experience data from {num_envs} parallel environments...")
    # print("Note: AtariWrapper provides 4 stacked frames automatically")
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
        jax.random.split(rng, n_envs)
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(env.step)(
        jax.random.split(rng, n_envs), env_state, action
    )
    states = []
    next_states = []
    actions = []
    rewards = []
    rng = jax.random.PRNGKey(42)
    jitted_reset = jax.jit(vmap_reset(num_envs))
    jitted_step = jax.jit(vmap_step(num_envs))
    rng, reset_rng = jax.random.split(rng)
    _, state = jitted_reset(reset_rng)
    total_steps = 0
    total_episodes = 0
    while total_episodes < num_episodes * num_envs:
        current_state_repr = jax.tree.map(lambda x: x, state.env_state)
        rng, action_rng = jax.random.split(rng)
        action_batch = jax.random.randint(action_rng, (num_envs,), 0, 18)
        rng, step_rng = jax.random.split(rng)
        _, next_state, reward_batch, done_batch, _ = jitted_step(
            step_rng, state, action_batch
        )
        if jnp.any(done_batch):
            rng, reset_rng = jax.random.split(rng)
            _, reset_states = jitted_reset(reset_rng)

            def update_where_done(old_state, new_state, done_mask):
                """Update states only where done_mask is True."""

                def where_with_correct_broadcasting(x, y, mask):
                    if hasattr(x, "shape") and hasattr(y, "shape"):
                        if x.ndim > 1:
                            new_shape = (mask.shape[0],) + (1,) * (x.ndim - 1)
                            reshaped_mask = mask.reshape(new_shape)
                            return jnp.where(reshaped_mask, y, x)
                        else:
                            return jnp.where(mask, y, x)
                    else:
                        return x

                return jax.tree.map(
                    lambda x, y: where_with_correct_broadcasting(x, y, done_mask),
                    old_state,
                    new_state,
                )

            next_state = update_where_done(next_state, reset_states, done_batch)
        next_state_repr = jax.tree.map(lambda x: x, next_state.env_state)
        states.append(current_state_repr)
        actions.append(action_batch)
        next_states.append(next_state_repr)
        rewards.append(reward_batch)
        newly_completed = jnp.sum(done_batch)
        total_episodes += newly_completed
        total_steps += num_envs
        state = next_state
        if total_episodes >= num_episodes * num_envs:
            break
    if VERBOSE:
        print(f"Experience collection completed:")
        print(f"- Total steps: {total_steps}")
        print(f"- Total episodes: {total_episodes}")
        print(f"- Total transitions: {len(states)}")
    states = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *states)
    actions = jnp.concatenate(actions, axis=0)
    next_states = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *next_states)
    rewards = jnp.concatenate(rewards, axis=0)
    if VERBOSE:
        print(
            f"Final flattened shape: states: {jax.tree.map(lambda x: x.shape, states)}"
        )
        print(f"Actions shape: {actions.shape}, Rewards shape: {rewards.shape}")
    return states, actions, next_states, rewards


def train_world_model(
    states,
    actions,
    next_states,
    rewards,
    learning_rate=3e-4,
    batch_size=256,
    num_epochs=10,
):
    # Calculate normalization statistics from the flattened states
    # States should be shape (num_samples, feature_dim)
    state_mean = jnp.mean(states, axis=0)  # Shape (feature_dim,)
    state_std = jnp.std(states, axis=0) + 1e-8  # Shape (feature_dim,)
    
    # Store normalization stats for later use
    normalization_stats = {'mean': state_mean, 'std': state_std}
    
    # Normalize states and next_states
    normalized_states = (states - state_mean) / state_std
    normalized_next_states = (next_states - state_mean) / state_std
    


    # Use normalized data for training
    model = build_world_model()
    optimizer = optax.adam(learning_rate=1e-4)
    
    rng = jax.random.PRNGKey(42)
    dummy_state = normalized_states[:1]  # Take first normalized state as example
    dummy_action = actions[:1]
    params = model.init(rng, dummy_state, dummy_action)
    opt_state = optimizer.init(params)

    # Update loss_function to work with normalized states
    def loss_function(params, state_batch, action_batch, next_state_batch):
        # Model now works with normalized states
        pred_next_state = model.apply(params, None, state_batch, action_batch)
        
        # Next state is also normalized
        target_next_state = next_state_batch[:, :-2]
        
        # Simple MSE loss on normalized values
        mse = jnp.mean((target_next_state - pred_next_state)**2)
        return mse

    @jax.jit
    def update_step(params, opt_state, state_batch, action_batch, next_state_batch):
        loss, grads = jax.value_and_grad(loss_function)(
            params, state_batch, action_batch, next_state_batch
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    num_batches = len(actions) // batch_size
    batches = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        state_batch = normalized_states[start_idx:end_idx]
        action_batch = actions[start_idx:end_idx]
        next_state_batch = normalized_next_states[start_idx:end_idx]
        batches.append((state_batch, action_batch, next_state_batch))
    batches = jax.device_put(batches)
    for epoch in range(num_epochs): #SCAN 
        losses = []
        for batch in batches: #maybe include in one scan
            state_batch, action_batch, next_state_batch = batch
            params, opt_state, loss = update_step(
                params, opt_state, state_batch, action_batch, next_state_batch
            )
            losses.append(loss)
        epoch_loss = jnp.mean(jnp.array(losses))
        if VERBOSE and (epoch + 1) % (num_epochs/10) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    return params, {
        "final_loss": epoch_loss, 
        "normalization_stats": normalization_stats
    }


def compare_real_vs_model(num_steps: int = 10, render_scale: int = 2, states=None, actions=None):
    # Add debugging to understand the model input/output formats
    def debug_states(step, real_state, pred_state):
        if step % 1 == 0:
            print(f"Step {step} debugging:")
            
            # Check ranges and statistics of the data
            print(f"Real state min/max/mean: {jnp.min(real_state):.2f}/{jnp.max(real_state):.2f}/{jnp.mean(real_state):.2f}")
            print(f"Model state min/max/mean: {jnp.min(pred_state):.2f}/{jnp.max(pred_state):.2f}/{jnp.mean(pred_state):.2f}")

            error = jnp.mean((real_state - pred_state) ** 2)
            print(f"Step {step_count}, Error: {error:.2f}")

    base_game = JaxSeaquest()
    real_env = AtariWrapper(
        base_game, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    renderer = SeaquestRenderer()
    model_path = "world_model.pkl"
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

    first_state_flat = states[0]
    _, unflattener = flatten_state(real_state.env_state, single_state=True)
    first_state_raw = unflattener(first_state_flat)
    real_state = real_state.replace(env_state=first_state_raw)
    model_state = real_state  # Start identical

    # Add reset interval to prevent error accumulation

    while running and step_count < min(num_steps, len(states)-1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Reset model state periodically to prevent excessive error accumulation

                
        # Use the saved action
        action = actions[step_count]
        
        # Use the saved next state directly instead of environment stepping
        next_real_state_flat = states[step_count + 1]
        real_state = real_state.replace(env_state=unflattener(next_real_state_flat))
        
        # Get the flattened current state for the model
        flattened_model_state, _ = flatten_state(model_state.env_state, single_state=True)
        
        # Apply model prediction with normalization
        if normalization_stats is not None:
            model_state_flattened = world_model.apply(
                dynamics_params, None, flattened_model_state, jnp.array([action]), normalization_stats
            )
        else:
            model_state_flattened = world_model.apply(
                dynamics_params, None, flattened_model_state, jnp.array([action])
            )

        debug_states(step_count, next_real_state_flat[:-2], model_state_flattened)

        # Complete model state with additional fields
        model_state_flattened = jnp.concatenate(
            [model_state_flattened, jnp.zeros((model_state_flattened.shape[0], 2))],
            axis=-1,
        )
        model_state_flattened_1d = model_state_flattened.reshape(-1)
        
        # Update model state
        model_state = model_state.replace(env_state=unflattener(model_state_flattened_1d))
        reconstructed_env_state = unflattener(model_state_flattened_1d)
        reconstructed_env_state = reconstructed_env_state._replace(
            step_counter=real_state.env_state.step_counter,
            rng_key=real_state.env_state.rng_key
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
       
        step_count += 1
        clock.tick(30)
        # Rendering stuff end -------------------------------------------------------

    pygame.quit()
    print("Comparison completed")


if __name__ == "__main__":

    batch_size = 128

    game = JaxSeaquest()
    env = AtariWrapper(
        game, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    env = FlattenObservationWrapper(env)

    save_path = "world_model.pkl"
    experience_data_path = "experience_data.pkl"
    model = build_world_model()
    normalization_stats = None


    # print(next_states[300][:-2])
    # pred = model.apply(dynamics_params, None, states[300], actions[300])
    # print(pred)
    

    # print(((next_states[300][:-2] - pred) ** 2))

    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data['dynamics_params']
            normalization_stats = saved_data.get('normalization_stats', None)
    else:
        print("No existing model found. Training a new model...")

        # Define a file path for the experience data
        

        # Check if experience data file exists
        if os.path.exists(experience_data_path):
            print(f"Loading existing experience data from {experience_data_path}...")
            with open(experience_data_path, 'rb') as f:
                saved_data = pickle.load(f)
                states = saved_data['states']
                actions = saved_data['actions']
                next_states = saved_data['next_states']
                rewards = saved_data['rewards']
        else:
            print("No existing experience data found. Collecting new experience data...")
            # Collect experience data (AtariWrapper handles frame stacking automatically)
            flattened_states, actions, _, rewards,_ = collect_experience_sequential(
                env,
                num_episodes=1,
                max_steps_per_episode=1000
            )
            next_states = flattened_states[1:]  # Next states are just the next frame in the sequence
            states = flattened_states[:-1]  # Current states are all but the last frame

            # render_trajectory(states, num_frames=1000, render_scale=2, delay=10)
            #I want to check whether the next_state is equal to the current state + 1
            print(states.shape)
            print(next_states.shape)

            if VERBOSE:
                print("Checking if next_state is equal to current state + 1...")
                for i in range(100):
                    if not jnp.allclose(states[i+1][:-2], next_states[i][:-2]):
                        print(f"Mismatch at index {i}: {states[i][:-2]} != {next_states[i][:-2]}")
                        print(states[i+1][:-2])
                        print(next_states[i][:-2])
                        print(states[i+1][:-2] - next_states[i][:-2])
                        sys.exit(1)
                print("All states match the expected transition.")

            
          
            # Save the collected experience data
            with open(experience_data_path, 'wb') as f:
                pickle.dump({
                    'states': states,
                    'actions': actions,
                    'next_states': next_states,
                    'rewards': rewards
                }, f)
            print(f"Experience data saved to {experience_data_path}")

        # Train world model
        dynamics_params, training_info = train_world_model(
            states,
            actions,
            next_states,
            rewards,
            batch_size=batch_size,
            num_epochs=10000,
        )
        normalization_stats = training_info.get('normalization_stats', None)

        # Save the model and scaling factor
        with open(save_path, 'wb') as f:
            pickle.dump({
                'dynamics_params': dynamics_params,
                'normalization_stats': training_info.get('normalization_stats', None)
            }, f)
        print(f"Model saved to {save_path}")

    states, actions = None, None
    if os.path.exists(experience_data_path):
        print(f"Loading existing experience data from {experience_data_path}...")
        with open(experience_data_path, 'rb') as f:
            saved_data = pickle.load(f)
            states = saved_data['states']
            actions = saved_data['actions']
            next_states = saved_data['next_states']
            rewards = saved_data['rewards']

    world_model = build_world_model()
    losses = []
    for i in range(len(states)):
        prediction = world_model.apply(
                dynamics_params, None, states[0+i], jnp.array([actions[0+i]]), normalization_stats
            )
        prediction
        
        loss = jnp.mean((prediction - next_states[0+i][:-2])**2)
        # print(loss)
        losses.append(loss)
        if i == 426:
            print('-----------------------------------------------------------------------------------------------------------------')

            print(f"Step {i}:")
            print(f"Loss : {loss}")
            print("Indexes where difference > 3:")
            for j in range(len(prediction[0])):
                if jnp.abs(prediction[0][j] - states[1+i][j]) > 3:
                    print(f"Index {j}: {prediction[0][j]} vs {states[1+i][j]}")
            print(f"Difference: {prediction - states[1+i][:-2]}")
            print(f"State {states[i]}")
            print("Negative values in state:")
            print(jnp.any(states[i][:-2] < -1))
            print(f"Prediction: {prediction}")
            print(f"Actual Next State {states[i+1]}")
            # print all indexes where the difference it greater than 10

            
        # if i == 30:
        #     # exit()
        #     break
        # print(f"Loss : {jnp.mean((prediction - states[1+i][:-2])**2)}")
    
    



    batch_size = batch_size
    num_batches = len(states) // batch_size
    total_loss = 0

    # Load normalization stats if available
    normalization_stats = None
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            normalization_stats = saved_data.get('normalization_stats', None)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        state_batch = states[start_idx:end_idx]
        action_batch = actions[start_idx:end_idx]
        next_state_batch = next_states[start_idx:end_idx]
        
        prediction = world_model.apply(
            dynamics_params, None, state_batch, jnp.array(action_batch), normalization_stats
        )
        
        loss = jnp.mean((prediction - next_state_batch[:,:-2])**2)
        total_loss += loss

    print(f"Average batch evaluation loss: {total_loss/num_batches}")

    exit()
    compare_real_vs_model(num_steps=5000, render_scale=2, states=states, actions=actions)
