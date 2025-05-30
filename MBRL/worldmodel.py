import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
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


def flatten_state(state, single_state: bool = False) -> Tuple[jnp.ndarray, Any]:
    """
    Flatten the state PyTree into a single array.
    This is useful for debugging and visualization.
    """
    #check whether it is a single state or a batch of states

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
    def forward(state, action):
        # print(state.shape)
        batch_size = action.shape[0] if len(action.shape) > 0 else 1

        if len(state.shape) == 1:
            feature_size = state.shape[0]
            flat_state_full = state.reshape(
                batch_size, feature_size // batch_size
            )
            
        else:
            flat_state_full = state
        flat_state = flat_state_full[..., :-2]




        action_one_hot = jax.nn.one_hot(action, num_classes=18)
        if len(state.shape) == 1:
            action_one_hot = action_one_hot.reshape(1,18)
        inputs = jnp.concatenate([flat_state, action_one_hot], axis=1)
        x = hk.Linear(512)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        if len(state.shape) == 1:
            flat_next_state = hk.Linear(state.shape[0]-2)(x)
        else:
            flat_next_state = hk.Linear(state.shape[1]-2)(x)
        # print(flat_next_state.shape)
        return flat_next_state

    return hk.transform(forward)


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
    
    model = build_world_model()
    lr_schedule = optax.exponential_decay(
        init_value=1e-5, transition_steps=500, decay_rate=0.9
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule)
    )
    SCALE_FACTOR = 1 / 1
    scaled_states = jax.tree.map(lambda x: x * SCALE_FACTOR, states)
    scaled_next_states = jax.tree.map(lambda x: x * SCALE_FACTOR, next_states)
    rng = jax.random.PRNGKey(42)
    dummy_state = jax.tree.map(lambda x: x[:1], scaled_states)
    dummy_action = actions[:1]
    params = model.init(rng, dummy_state, dummy_action)
    opt_state = optimizer.init(params)




    def loss_function(params, state_batch, action_batch, next_state_batch):




        pred_next_state = model.apply(params, None, state_batch, action_batch)
        flat_next_state_raw = jax.flatten_util.ravel_pytree(next_state_batch)[0]


        flat_next_state_raw = flat_next_state_raw.reshape(
            state_batch.shape[0], -1
        )  

        state_batch = state_batch[:, :-2]
        mse = jnp.mean((state_batch - pred_next_state)**2)

        return mse

        # batch_size = pred_next_state.shape[0]
        # pred_feature_size = pred_next_state.shape[1]
        # if hasattr(loss_function, "first_call") == False:
        #     print(f"Prediction shape: {pred_next_state.shape}")
        #     print(f"Target shape before reshape: {flat_next_state_raw.shape}")
        #     loss_function.first_call = True
        # if len(flat_next_state_raw.shape) == 1:
        #     total_size = flat_next_state_raw.shape[0]
        #     full_feature_size = total_size // batch_size
        #     flat_next_state_full = flat_next_state_raw.reshape(
        #         batch_size, full_feature_size
        #     )
        # else:
        #     flat_next_state_full = flat_next_state_raw
        # flat_next_state = flat_next_state_full[:, :-2]
        # single_frame_size = flat_next_state.shape[1] // 4
        # target_single_frame = flat_next_state[:, -single_frame_size:]
        # if target_single_frame.shape[1] != pred_feature_size:
        #     min_size = min(target_single_frame.shape[1], pred_feature_size)
        #     target_single_frame = target_single_frame[:, :min_size]
        #     pred_next_state = pred_next_state[:, :min_size]
        # mae = (jnp.abs(pred_next_state - target_single_frame))
        # print(
        #     f"MAE shape: {mae.shape}, first 5 values: {mae[:5]}"
        # )
        # print(
        #     f"MAE shape: {pred_next_state.shape}, first 5 values: {mae[:5]}"
        # )
        # print(
        #     f"MAE shape: {target_single_frame.shape}, first 5 values: {mae[:5]}"
        # )
        

    loss_function.first_call = False

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
        state_batch = jax.tree.map(lambda x: x[start_idx:end_idx], scaled_states)
        action_batch = actions[start_idx:end_idx]
        next_state_batch = jax.tree.map(
            lambda x: x[start_idx:end_idx], scaled_next_states
        )
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
        if VERBOSE and (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    return params, {"final_loss": epoch_loss, "scale_factor": SCALE_FACTOR}


def compare_real_vs_model(num_steps: int = 1000, render_scale: int = 2):
    """
    Compare the real environment with the world model predictions.
    Now works with 4 stacked frames from AtariWrapper.
    """
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
    world_model = build_world_model()
    import pygame
    import time

    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(
        "Real Environment vs World Model (AtariWrapper Frame Stack)"
    )
    rng = jax.random.PRNGKey(int(time.time()))
    rng, reset_key = jax.random.split(rng)
    real_obs, real_state = real_env.reset(reset_key)
    model_state = real_state
    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    def step_with_rng(rng, state, action):
        return real_env.step(rng, state, action)

    jitted_step = jax.jit(step_with_rng)

    running = True
    step_count = 0
    clock = pygame.time.Clock()
    while running and step_count < num_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        rng, action_key = jax.random.split(rng)
        action = jax.random.randint(action_key, shape=(), minval=0, maxval=18)
        if step_count % 4 == 0:
            action = 5
        rng, step_rng = jax.random.split(rng)
        real_obs, real_state, real_reward, real_done, real_info = jitted_step(
            step_rng, real_state, action
        )
        # model_state = jitted_predict(dynamics_params, model_state, action, scale_factor)
        flattened_state, unflattener = flatten_state(model_state.env_state, single_state=True)
        model_state_flattened = world_model.apply(
            dynamics_params, None, flattened_state, jnp.array([action])
        )
        #extend model_state_flattened by 2 zeros to match the shape of real_state.env_state
        model_state_flattened = jnp.concatenate(
            [model_state_flattened, jnp.zeros((model_state_flattened.shape[0], 2))],
            axis=-1,
        )
        print(model_state_flattened.shape)
        model_state_flattened_1d = model_state_flattened.reshape(-1)
        print(model_state_flattened_1d.shape)
        print(unflattener(model_state_flattened_1d))
        model_state = model_state.replace(env_state=unflattener(model_state_flattened_1d))
        
        model_state_structure = jax.tree_util.tree_structure(real_state.env_state)
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
        print(
            "---------------------------------------------------------------Real State-------------------------------------------------"
        )
        print(real_base_state)
        print(
            "---------------------------------------------------------------Model State------------------------------------------------"
        )
        print(model_state)
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
        if real_done:
            rng, reset_key = jax.random.split(rng)
            real_obs, real_state = real_env.reset(reset_key)
            model_state = real_state
        step_count += 1
        clock.tick(30)
    pygame.quit()
    print("Comparison completed")


if __name__ == "__main__":



    game = JaxSeaquest()
    env = AtariWrapper(
        game, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    env = FlattenObservationWrapper(env)

    save_path = "world_model.pkl"
    model = build_world_model()


    # print(next_states[300][:-2])
    # pred = model.apply(dynamics_params, None, states[300], actions[300])
    # print(pred)


    # print(((next_states[300][:-2] - pred) ** 2))

    if os.path.exists(save_path) and sys.argv[1] != 'retrain':
        print(f"Loading existing model from {save_path}...")
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data['dynamics_params']
    else:
        print("No existing model found. Training a new model...")

        # Define a file path for the experience data
        experience_data_path = "experience_data.pkl"

        # Check if experience data file exists
        if os.path.exists(experience_data_path) and sys.argv[1] != 'retrain':
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
            states, actions, next_states, rewards = collect_experience(
                env, num_episodes=1, max_steps_per_episode=1000, num_envs=1
            )

            states = flatten_state(states)[0]  # Flatten the state for training
            next_states = flatten_state(next_states)[0]  # Flatten the next states for training
            
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
            learning_rate=3e-4,
            batch_size=2,
            num_epochs=1000,
        )

        # Save the model and scaling factor
        with open(save_path, 'wb') as f:
            pickle.dump({
                'dynamics_params': dynamics_params,
                'scale_factor': training_info['scale_factor']
            }, f)
        print(f"Model saved to {save_path}")

    compare_real_vs_model(num_steps=5000, render_scale=6)
