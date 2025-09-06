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
from jaxatari.games.jax_pong import PongRenderer, JaxPong
from jaxatari.wrappers import LogWrapper, FlattenObservationWrapper, AtariWrapper
from jax import lax
import gc
from rtpt import RTPT
from obs_state_converter import pong_flat_observation_to_state

from model_architectures import *






def get_reward_from_observation_score(obs):
    """Extract reward from Pong observation - adjust index as needed for Pong"""
    if len(obs) < 100:
        raise ValueError(f"Observation must have sufficient elements, got {len(obs)}")

    return obs[-3] if len(obs) > 3 else 0





















MODEL_ARCHITECTURE = PongLSTM
model_scale_factor = 10


VERBOSE = True
model = None


action_map = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}


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
    renderer = PongRenderer()
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


def flatten_obs(
    state, single_state: bool = False, is_list=False
) -> Tuple[jnp.ndarray, Any]:
    """
    Flatten the state PyTree into a single array.
    This is useful for debugging and visualization.
    """

    if type(state) == list:
        flat_states = []

        for s in state:
            flat_state, _ = jax.flatten_util.ravel_pytree(s)
            flat_states.append(flat_state)
        flat_states = jnp.stack(flat_states, axis=0)
        print(flat_states.shape)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener
    batch_shape = (
        state.player_x.shape[0]
        if hasattr(state, "player_x")
        else state.paddle_y.shape[0]
    )

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)
    return flat_state, unflattener


def collect_experience_sequential(
    env,
    num_episodes: int = 1,
    max_steps_per_episode: int = 1000,
    episodic_life: bool = True,
    seed: int = 42,
    policy_params=None,
    network=None,
):
    """Collect experience data sequentially to ensure proper transitions."""
    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    boundaries = []

    dead = False
    total_steps = 0
    rng = jax.random.PRNGKey(seed)

    def pong_left_right_policy(rng):
        """Simple left-right movement policy for Pong"""
        rng, action_key = jax.random.split(rng)

        action_prob = jax.random.uniform(action_key)

        action = jax.random.randint(action_key, (), 0, 6)

        return action

    def pong_tracking_policy(rng, step_count):
        """Policy that alternates between tracking movements"""
        rng, action_key = jax.random.split(rng)

        cycle = step_count % 60

        if cycle < 30:
            if jax.random.uniform(action_key) < 0.7:
                action = 2
            else:
                action = jax.random.randint(action_key, (), 0, 6)
        else:
            if jax.random.uniform(action_key) < 0.7:
                action = 3
            else:
                action = jax.random.randint(action_key, (), 0, 6)

        return action

    def random_pong_policy(rng):
        """Completely random policy for exploration"""
        rng, action_key = jax.random.split(rng)
        action = jax.random.randint(action_key, (), 0, 6)
        return action

    def perfect_policy(obs, rng):

        rng, action_key = jax.random.split(rng)
        if jax.random.uniform(action_key) < 0.5:
            action = jax.random.randint(action_key, (), 0, 6)
            return action

        if obs.player.y[3] > obs.ball.y[3]:
            return 4
        if obs.player.y[3] < obs.ball.y[3]:
            return 3
        if obs.player.y[3] == obs.ball.y[3]:
            return 0

    for episode in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, state = env.reset(reset_key)

        for step in range(max_steps_per_episode):
            current_state = state
            current_obs = obs

            rng, action_key = jax.random.split(rng)

            if network and policy_params:

                flat_obs, _ = flatten_obs(obs, single_state=True)
                pi, _ = network.apply(policy_params, flat_obs)
                action = pi.sample(seed=action_key)
            else:
                action = perfect_policy(obs, rng)

            rng, step_key = jax.random.split(rng)
            next_obs, next_state, reward, done, _ = env.step(state, action)

            observations.append(current_obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            dones.append(done)

            if not episodic_life:

                if hasattr(current_state, "env_state") and hasattr(
                    current_state.env_state, "death_counter"
                ):
                    if current_state.env_state.death_counter > 0 and not dead:
                        dead = True
                    if not current_state.env_state.death_counter > 0 and dead:
                        dead = False
                        boundaries.append(total_steps)

            if done:
                print(f"Episode {episode+1} done after {step+1} steps")

                if episodic_life:
                    if len(boundaries) == 0:
                        boundaries.append(step)
                        print("No boundaries found, adding first step")
                    else:
                        boundaries.append(boundaries[-1] + step + 1)
                        print("Adding boundary at step", boundaries[-1])
                break

            state = next_state
            obs = next_obs
            total_steps += 1

    actions_array = jnp.array(actions)
    rewards_array = jnp.array(rewards)
    dones_array = jnp.array(dones)

    return (
        flatten_obs(observations, is_list=True),
        actions_array,
        rewards_array,
        dones_array,
        boundaries,
    )


def train_world_model(
    obs,
    actions,
    next_obs,
    rewards,
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=1000,
    sequence_length=32,
    episode_boundaries=None,
    frame_stack_size=4,
    model_scale_factor=1,
):

    gpu_batch_size = 250

    gpu_batch_size = gpu_batch_size // frame_stack_size

    state_mean = jnp.mean(obs, axis=0)
    state_std = jnp.std(obs, axis=0) + 1e-8

    state_mean = 0
    state_std = 1

    normalization_stats = {"mean": state_mean, "std": state_std}

    normalized_obs = (obs - state_mean) / state_std
    normalized_next_obs = (next_obs - state_mean) / state_std

    def create_sequential_batches(batch_size=32):

        sequences = []

        for i in range(len(episode_boundaries) - 1):
            if i == 0:
                start_idx = 0
                end_idx = episode_boundaries[0]
            else:
                start_idx = episode_boundaries[i - 1]
                end_idx = episode_boundaries[i]

            for j in range(
                0, end_idx - start_idx - sequence_length + 1, sequence_length // 4
            ):
                if start_idx + j + sequence_length > end_idx:

                    padding_length = start_idx + j + sequence_length - end_idx
                    padded_obs = jnp.concatenate(
                        [
                            normalized_obs[start_idx + j : end_idx],
                            jnp.tile(normalized_obs[end_idx - 1], (padding_length, 1)),
                        ],
                        axis=0,
                    )
                    padded_actions = jnp.concatenate(
                        [
                            actions[start_idx + j : end_idx],
                            jnp.tile(actions[end_idx - 1], (padding_length,)),
                        ],
                        axis=0,
                    )
                    padded_next_obs = jnp.concatenate(
                        [
                            normalized_next_obs[start_idx + j : end_idx],
                            jnp.tile(
                                normalized_next_obs[end_idx - 1], (padding_length, 1)
                            ),
                        ],
                        axis=0,
                    )

                    sequences.append((padded_obs, padded_actions, padded_next_obs))
                    continue

                sequences.append(
                    (
                        normalized_obs[start_idx + j : start_idx + j + sequence_length],
                        actions[start_idx + j : start_idx + j + sequence_length],
                        normalized_next_obs[
                            start_idx + j : start_idx + j + sequence_length
                        ],
                    )
                )

        return sequences

    batches = create_sequential_batches()
    print(f"Created {len(batches)} sequential batches of size {sequence_length}")

    total_batches = len(batches)
    train_size = int(0.8 * total_batches)

    rng_split = jax.random.PRNGKey(42)
    indices = jax.random.permutation(rng_split, total_batches)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_batches = [batches[i] for i in train_indices]
    val_batches = [batches[i] for i in val_indices]

    print(
        f"Training batches: {len(train_batches)}, Validation batches: {len(val_batches)}"
    )

    model = MODEL_ARCHITECTURE(model_scale_factor)

    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=num_epochs,
        alpha=0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8),
    )

    rng = jax.random.PRNGKey(42)
    dummy_state = normalized_obs[:1]
    dummy_action = actions[:1]
    params = model.init(rng, dummy_state, dummy_action, None)
    opt_state = optimizer.init(params)

    
    _, lstm_state_template = model.apply(params, rng, dummy_state, dummy_action, None)

    @jax.jit
    def single_sequence_loss(
        params,
        state_batch,
        action_batch,
        next_state_batch,
        lstm_template,
        epoch,
        max_epochs,
    ):

        def scan_fn(rssm_state, inputs):
            current_state, current_action, target_next_state = inputs

            
            pred_next_state, new_rssm_state = model.apply(
                params, rng, current_state[None, :], current_action[None], rssm_state
            )
            pred_next_state = pred_next_state.squeeze()
            loss = jnp.mean((target_next_state - pred_next_state) ** 2)

            return new_rssm_state, loss

        scan_inputs = (state_batch, action_batch, next_state_batch)
        _, step_losses = lax.scan(scan_fn, lstm_template, scan_inputs)

        return jnp.mean(step_losses)

    batched_loss_fn = jax.vmap(
        single_sequence_loss, in_axes=(None, 0, 0, 0, None, None, None)
    )

    @jax.jit
    def update_step_batched(
        params,
        opt_state,
        batch_states,
        batch_actions,
        batch_next_states,
        lstm_template,
        epoch=0,
        num_epochs=20000,
    ):

        def loss_fn(p):
            losses = batched_loss_fn(
                p,
                batch_states,
                batch_actions,
                batch_next_states,
                lstm_template,
                epoch,
                num_epochs,
            )
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def compute_validation_loss(
        params,
        batch_states,
        batch_actions,
        batch_next_states,
        lstm_template,
        epoch,
        num_epochs,
    ):
        """Compute validation loss without updating parameters"""
        losses = batched_loss_fn(
            params,
            batch_states,
            batch_actions,
            batch_next_states,
            lstm_template,
            epoch,
            num_epochs,
        )
        return jnp.mean(losses)

    train_batch_states = jnp.stack([batch[0] for batch in train_batches])
    train_batch_actions = jnp.stack([batch[1] for batch in train_batches])
    train_batch_next_states = jnp.stack([batch[2] for batch in train_batches])

    val_batch_states = jnp.stack([batch[0] for batch in val_batches])
    val_batch_actions = jnp.stack([batch[1] for batch in val_batches])
    val_batch_next_states = jnp.stack([batch[2] for batch in val_batches])

    rng_shuffle = jax.random.PRNGKey(123)

    best_loss = float("inf")
    patience = 50
    no_improve_count = 0

    for epoch in range(num_epochs):

        rng_shuffle, shuffle_key = jax.random.split(rng_shuffle)
        indices = jax.random.permutation(shuffle_key, len(train_batches))

        epoch_shuffle_key = jax.random.fold_in(rng_shuffle, epoch)
        epoch_indices = jax.random.permutation(epoch_shuffle_key, len(train_batches))

        shuffled_train_states = train_batch_states[epoch_indices]
        shuffled_train_actions = train_batch_actions[epoch_indices]
        shuffled_train_next_states = train_batch_next_states[epoch_indices]

        if shuffled_train_states.shape[0] > gpu_batch_size:
            shuffled_train_states = shuffled_train_states[:gpu_batch_size]
            shuffled_train_actions = shuffled_train_actions[:gpu_batch_size]
            shuffled_train_next_states = shuffled_train_next_states[:gpu_batch_size]

        params, opt_state, train_loss = update_step_batched(
            params,
            opt_state,
            shuffled_train_states,
            shuffled_train_actions,
            shuffled_train_next_states,
            lstm_state_template,
            epoch,
            num_epochs,
        )

        if train_loss < best_loss:
            best_loss = train_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if False and no_improve_count >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs"
            )
            val_loss = compute_validation_loss(
                params,
                val_batch_states,
                val_batch_actions,
                val_batch_next_states,
                lstm_state_template,
                epoch,
                num_epochs,
            )

            current_lr = lr_schedule(epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            )
            break

        if VERBOSE and (epoch + 1) % 10 == 0:
            val_loss = compute_validation_loss(
                params,
                val_batch_states,
                val_batch_actions,
                val_batch_next_states,
                lstm_state_template,
                epoch,
                num_epochs,
            )

            current_lr = lr_schedule(epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            )

    print("Training completed")
    return params, {
        "final_loss": train_loss,
        "normalization_stats": normalization_stats,
        "best_loss": best_loss,
    }


def compare_real_vs_model(
    num_steps: int = 150,
    render_scale: int = 2,
    obs=None,
    actions=None,
    normalization_stats=None,
    steps_into_future: int = 20,
    clock_speed=10,
    boundaries=None,
    env=None,
    starting_step: int = 0,
    render_debugging: bool = False,
    frame_stack_size: int = 4,
    model_scale_factor=4,
    model_path=None,
):

    rng = jax.random.PRNGKey(0)

    if len(obs) == 1:
        obs = obs.squeeze(0)

    def debug_obs(
        step,
        real_obs,
        pred_obs,
        action,
    ):
        error = jnp.mean((real_obs - pred_obs[0]) ** 2)
        print(
            f"Step {step}, Unnormalized Error: {error:.2f} | Action: {action_map.get(int(action), action)} Reward : {get_dense_pong_reward(real_obs, action, frame_stack_size=frame_stack_size):.2f}"
        )

        if error > 20 and render_debugging:
            print("-" * 100)
            print("Indexes where difference > 1:")
            for j in range(len(pred_obs[0])):
                if jnp.abs(pred_obs[0][j] - real_obs[j]) > 10:
                    print(
                        f"Index {j}: Predicted {pred_obs[0][j]:.2f} vs Real {real_obs[j]:.2f}"
                    )
            print("-" * 100)

    def check_lstm_state_health(lstm_state, step):
        if lstm_state is not None:

            lstm1_state, lstm2_state = lstm_state

            hidden1_norm = jnp.linalg.norm(lstm1_state.hidden)
            cell1_norm = jnp.linalg.norm(lstm1_state.cell)

            hidden2_norm = jnp.linalg.norm(lstm2_state.hidden)
            cell2_norm = jnp.linalg.norm(lstm2_state.cell)

            max_norm = max(hidden1_norm, cell1_norm, hidden2_norm, cell2_norm)
            min_norm = min(hidden1_norm, cell1_norm, hidden2_norm, cell2_norm)

            if max_norm > 5.0:
                print(
                    f"Step {step}: LSTM state explosion - Layer1 h:{hidden1_norm:.2f} c:{cell1_norm:.2f}, Layer2 h:{hidden2_norm:.2f} c:{cell2_norm:.2f}"
                )
            elif min_norm < 0.01:
                print(
                    f"Step {step}: LSTM state vanishing - Layer1 h:{hidden1_norm:.2f} c:{cell1_norm:.2f}, Layer2 h:{hidden2_norm:.2f} c:{cell2_norm:.2f}"
                )
            else:

                if step % 50 == 0:
                    print(
                        f"Step {step}: LSTM states healthy - Layer1 h:{hidden1_norm:.2f} c:{cell1_norm:.2f}, Layer2 h:{hidden2_norm:.2f} c:{cell2_norm:.2f}"
                    )

    if normalization_stats:
        state_mean = normalization_stats["mean"]
        state_std = normalization_stats["std"]
    else:
        state_mean = 0
        state_std = 1

    renderer = PongRenderer()
    if not model_path:
        if len(sys.argv) > 4 and sys.argv[4].startswith("check"):
            model_path = sys.argv[4]
        else:
            if os.path.exists(f"world_model_{MODEL_ARCHITECTURE.__name__}_pong.pkl"):
                model_path = f"world_model_{MODEL_ARCHITECTURE.__name__}_pong.pkl"
            else:
                model_path = "model_pong.pkl"

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        dynamics_params = model_data["dynamics_params"]
        normalization_stats = model_data.get("normalization_stats", None)
    world_model = MODEL_ARCHITECTURE(model_scale_factor)


    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs World Model (Pong)")

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    step_count = 0 + starting_step
    clock = pygame.time.Clock()

    game = JaxPong()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    dummy_obs, _ = env.reset(jax.random.PRNGKey(int(time.time())))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    real_obs = obs[0]
    model_obs = obs[0]

    lstm_state = None
    lstm_real_state = None

    model_base_state = None

    while step_count < min(num_steps, len(obs) - 1):

        action = actions[step_count]
        
        
        


        print(f"Reward : {get_dense_pong_reward(obs[step_count + 1], action, frame_stack_size=frame_stack_size):.2f}")

        

        next_real_obs = obs[step_count + 1]

        if steps_into_future > 0 and (
            step_count % steps_into_future == 0 or step_count in boundaries
        ):
            print("State reset")
            model_obs = obs[step_count]

        normalized_flattened_model_obs = (model_obs - state_mean) / state_std

        if steps_into_future > 0:

            normalized_model_prediction, lstm_state = world_model.apply(
                dynamics_params,
                rng,
                normalized_flattened_model_obs,
                jnp.array([action]),
                lstm_state,
            )
        else:
            normalized_model_prediction = normalized_flattened_model_obs

        unnormalized_model_prediction = jnp.round(
            normalized_model_prediction * state_std + state_mean
        )

        model_obs = unnormalized_model_prediction

        if steps_into_future > 0:
            debug_obs(step_count, next_real_obs, unnormalized_model_prediction, action)

        real_base_state = pong_flat_observation_to_state(
            real_obs, unflattener, frame_stack_size=frame_stack_size
        )
        model_base_state = pong_flat_observation_to_state(
            model_obs.squeeze(), unflattener, frame_stack_size=frame_stack_size
        )

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
        model_text = font.render("World Model (Pong)", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        pygame.display.flip()

        real_obs = obs[step_count]
        if steps_into_future > 0:
            normalized_real_obs = (real_obs - state_mean) / state_std
            _, lstm_real_state = world_model.apply(
                dynamics_params,
                rng,
                normalized_real_obs,
                jnp.array([action]),
                lstm_real_state,
            )

        if steps_into_future > 0 and (
            step_count % steps_into_future == 0 or step_count in boundaries
        ):
            lstm_state = lstm_real_state

        step_count += 1
        clock.tick(clock_speed)

    pygame.quit()
    print("Comparison completed")


def main():

    frame_stack_size = 4

    game = JaxPong()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    env = FlattenObservationWrapper(env)

    save_path = f"world_model_{MODEL_ARCHITECTURE.__name__}_pong.pkl"
    experience_data_path = "experience_data_LSTM_pong.pkl"
    model = MODEL_ARCHITECTURE(model_scale_factor)
    normalization_stats = None

    experience_its = 1

    if not os.path.exists("experience_data_LSTM_pong_0.pkl"):
        print("No existing experience data found. Collecting new experience data...")

        for i in range(0, experience_its):
            print(f"Collecting experience data (iteration {i+1}/{experience_its})...")
            obs, actions, rewards, _, boundaries = collect_experience_sequential(
                env, num_episodes=20, max_steps_per_episode=10000, seed=i
            )
            next_obs = obs[1:]
            obs = obs[:-1]

            experience_path = "experience_data_LSTM_pong" + "_" + str(i) + ".pkl"

            with open(experience_path, "wb") as f:
                pickle.dump(
                    {
                        "obs": obs,
                        "actions": actions,
                        "next_obs": next_obs,
                        "rewards": rewards,
                        "boundaries": boundaries,
                    },
                    f,
                )
            print(f"Experience data saved to {experience_path}")

            del obs, actions, rewards, boundaries, next_obs
            gc.collect()

    obs = []
    actions = []
    next_obs = []
    rewards = []
    boundaries = []

    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, "rb") as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data["dynamics_params"]
            normalization_stats = saved_data.get("normalization_stats", None)
    else:
        print("No existing model found. Training a new model...")

        for i in range(0, experience_its):
            experience_path = "experience_data_LSTM_pong" + "_" + str(i) + ".pkl"
            with open(experience_path, "rb") as f:
                saved_data = pickle.load(f)
                obs.extend(saved_data["obs"])
                actions.extend(saved_data["actions"])
                next_obs.extend(saved_data["next_obs"])
                rewards.extend(saved_data["rewards"])

                offset = boundaries[-1] if boundaries else 0

                adjusted_boundaries = [b + offset for b in saved_data["boundaries"]]
                boundaries.extend(adjusted_boundaries)

        obs_array = jnp.array(obs)
        actions_array = jnp.array(actions)
        next_obs_array = jnp.array(next_obs)
        rewards_array = jnp.array(rewards)

        dynamics_params, training_info = train_world_model(
            obs_array,
            actions_array,
            next_obs_array,
            rewards_array,
            episode_boundaries=boundaries,
            frame_stack_size=frame_stack_size,
            model_scale_factor=model_scale_factor,
        )
        normalization_stats = training_info.get("normalization_stats", None)

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

    gc.collect()

    with open(f"experience_data_LSTM_pong_{0}.pkl", "rb") as f:
        saved_data = pickle.load(f)
        obs = saved_data["obs"]
        actions = saved_data["actions"]
        next_obs = saved_data["next_obs"]
        rewards = saved_data["rewards"]
        boundaries = saved_data["boundaries"]

    if len(args := sys.argv) > 1 and args[1] == "render":
        compare_real_vs_model(
            num_steps=1000,
            render_scale=6,
            obs=obs,
            actions=actions,
            normalization_stats=normalization_stats,
            boundaries=boundaries,
            env=env,
            starting_step=0,
            steps_into_future=10,
            render_debugging=(args[3] == "verbose" if len(args) > 3 else False),
            frame_stack_size=frame_stack_size,
            model_scale_factor=model_scale_factor,
        )


if __name__ == "__main__":

    main()
