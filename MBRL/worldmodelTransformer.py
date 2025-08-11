import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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


class TransformerWorldModel(hk.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        max_seq_len: int = 32,
        name: str = "transformer_world_model",
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len

    def __call__(self, states, actions, training=True):
        """
        Simplified approach - treat state-action pairs as single tokens
        """
        batch_size, seq_len, state_dim = states.shape

        print(batch_size, states.shape, actions.shape)

        actions_expanded = jnp.expand_dims(actions, -1)

        combined = jnp.concatenate(
            [states, actions_expanded.astype(jnp.float32)], axis=-1
        )

        x = hk.Linear(self.d_model)(combined)

        positions = jnp.arange(seq_len)[None, :, None]
        pos_encoding = self._positional_encoding(positions, self.d_model)
        x += pos_encoding

        if training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        for _ in range(self.num_layers):
            x = self._transformer_block(x, training=training)

        predicted_next_states = hk.Linear(state_dim)(x)

        return predicted_next_states

    def _positional_encoding(self, positions, d_model):
        """Generate sinusoidal positional encodings."""

        def get_angles(pos, i, d_model):
            angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / jnp.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(positions, jnp.arange(d_model)[None, None, :], d_model)

        angle_rads = angle_rads.at[:, :, 0::2].set(jnp.sin(angle_rads[:, :, 0::2]))

        angle_rads = angle_rads.at[:, :, 1::2].set(jnp.cos(angle_rads[:, :, 1::2]))

        return angle_rads

    def _transformer_block(self, x, training=True):
        """Single transformer block with multi-head attention and feed-forward."""

        attn_output = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.d_model // self.num_heads,
            w_init_scale=1.0,
        )(x, x, x)

        if training:
            attn_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn_output)

        x1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            x + attn_output
        )

        ff_output = hk.Sequential(
            [hk.Linear(self.d_model * 2), jax.nn.relu, hk.Linear(self.d_model)]
        )(x1)

        if training:
            ff_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff_output)

        x2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(
            x1 + ff_output
        )

        return x2


class MLPWorldModel(hk.Module):
    def __init__(self, hidden_size: int = 512, num_layers: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def __call__(self, state, action, hidden_state=None):

        x = jnp.concatenate([state, jnp.array([action])], axis=-1)

        for _ in range(self.num_layers):
            x = hk.Linear(self.hidden_size)(x)
            x = jax.nn.relu(x)

        next_state = hk.Linear(state.shape[-1])(x)

        return next_state, None


def get_model_architecture():
    """Get the model architecture from command line or default to Transformer."""
    if len(sys.argv) > 1:
        model_name = sys.argv[1].upper()
        if model_name == "TRANSFORMER":
            return TransformerWorldModel
        elif model_name == "MLP":
            return MLPWorldModel
        else:
            print(f"Unknown model architecture: {model_name}, using Transformer")
            return TransformerWorldModel
    else:
        return TransformerWorldModel


MODEL_ARCHITECTURE = get_model_architecture()
VERBOSE = True


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
    """Render a trajectory of states in a single window."""
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
    """Flatten the state PyTree into a single array."""
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

    if hasattr(state, "player") and hasattr(state.player, "y"):
        batch_shape = state.player.y.shape[0]
    elif hasattr(state, "ball") and hasattr(state.ball, "x"):
        batch_shape = state.ball.x.shape[0]
    else:
        first_leaf = jax.tree_util.tree_leaves(state)[0]
        batch_shape = (
            first_leaf.shape[0]
            if hasattr(first_leaf, "shape") and len(first_leaf.shape) > 0
            else 1
        )

    flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
    flat_state = flat_state.reshape(batch_shape, -1)
    return flat_state, unflattener


def train_world_model(
    obs,
    actions,
    next_obs,
    rewards,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100,
    sequence_length=32,
    episode_boundaries=None,
    frame_stack_size=4,
):
    """Train the Transformer world model."""

    state_mean = jnp.mean(obs, axis=0)
    state_std = jnp.std(obs, axis=0) + 1e-8
    normalization_stats = {"mean": state_mean, "std": state_std}

    normalized_obs = (obs - state_mean) / state_std
    normalized_next_obs = (next_obs - state_mean) / state_std

    def create_sequential_batches():
        """Create batches of sequential data for transformer training."""
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
                if start_idx + j + sequence_length <= end_idx:
                    sequences.append(
                        (
                            normalized_obs[
                                start_idx + j : start_idx + j + sequence_length
                            ],
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

    def model_fn(states, actions):
        model = MODEL_ARCHITECTURE()
        return model(states, actions)

    model = hk.transform(model_fn)

    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=num_epochs, alpha=0.1
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule)
    )

    rng = jax.random.PRNGKey(42)
    dummy_states = jnp.ones((1, sequence_length, normalized_obs.shape[-1]))
    dummy_actions = jnp.ones((1, sequence_length), dtype=jnp.int32)

    params = model.init(rng, dummy_states, dummy_actions)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, rng, states_batch, actions_batch, targets_batch):
        predictions = model.apply(params, rng, states_batch, actions_batch)
        loss = jnp.mean((predictions - targets_batch) ** 2)
        return loss

    @jax.jit
    def update_step(params, opt_state, rng, states_batch, actions_batch, targets_batch):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            params, rng, states_batch, actions_batch, targets_batch
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    gpu_batch_size = 10

    gpu_batch_size = gpu_batch_size // frame_stack_size

    train_states = jnp.stack([batch[0] for batch in train_batches])
    train_actions = jnp.stack([batch[1] for batch in train_batches])
    train_targets = jnp.stack([batch[2] for batch in train_batches])

    val_states = jnp.stack([batch[0] for batch in val_batches])
    val_actions = jnp.stack([batch[1] for batch in val_batches])
    val_targets = jnp.stack([batch[2] for batch in val_batches])

    best_loss = float("inf")

    for epoch in range(num_epochs):

        rng, shuffle_key = jax.random.split(rng)
        indices = jax.random.permutation(shuffle_key, len(train_batches))

        max_batches_per_epoch = min(gpu_batch_size, len(train_batches))
        selected_indices = indices[:max_batches_per_epoch]

        epoch_states = train_states[selected_indices]
        epoch_actions = train_actions[selected_indices]
        epoch_targets = train_targets[selected_indices]

        rng, epoch_rng = jax.random.split(rng)

        params, opt_state, train_loss = update_step(
            params, opt_state, epoch_rng, epoch_states, epoch_actions, epoch_targets
        )

        if train_loss < best_loss:
            best_loss = train_loss

        if VERBOSE and (epoch + 1) % 10 == 0:

            rng, val_rng = jax.random.split(rng)
            val_loss = loss_fn(params, val_rng, val_states, val_actions, val_targets)
            current_lr = lr_schedule(epoch)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            )

    print("Training completed")
    return params, {
        "final_loss": train_loss,
        "normalization_stats": normalization_stats,
        "best_loss": best_loss,
        "model_fn": model,
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
    model_params=None,
    model_fn=None,
):
    """Compare real environment vs model predictions with rendering."""

    if len(obs) == 1:
        obs = obs.squeeze(0)

    def debug_obs(step, real_obs, pred_obs, action):
        print(real_obs)
        error = jnp.mean((real_obs - pred_obs[0]) ** 2)
        print(
            f"Step {step}, Unnormalized Error: {error:.2f} | Action: {action_map.get(int(action), 'UNKNOWN')}"
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

    state_mean = normalization_stats["mean"]
    state_std = normalization_stats["std"]

    renderer = PongRenderer()

    # Use the parameters passed to the function
    dynamics_params = model_params

    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs Transformer World Model (Pong)")

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    step_count = starting_step
    clock = pygame.time.Clock()

    game = JaxPong()
    env_wrapper = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    dummy_obs, _ = env_wrapper.reset(jax.random.PRNGKey(int(time.time())))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    # Initialize model and real observations
    real_obs = obs[0]
    model_obs = obs[0]
    sequence_buffer = []
    real_sequence_buffer = []
    
    # Initialize RNG for model inference
    rng = jax.random.PRNGKey(42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Total observations: {len(obs)}")

    while step_count < min(num_steps, len(obs) - 1):
        action = actions[step_count]#
        real_obs = obs[step_count]
        next_real_obs = obs[step_count + 1]

        # Model reset logic - reset at specified intervals or boundaries
        if steps_into_future > 0 and (
            step_count % steps_into_future == 0 or step_count in boundaries
        ):
            print("Model Reset!")
            model_obs = obs[step_count]
            sequence_buffer = []

        # Normalize observations
        normalized_model_obs = (model_obs - state_mean) / state_std

        if steps_into_future > 0:
            # Build sequence buffer for transformer
            sequence_buffer.append((normalized_model_obs, action))
            
            # Keep only the last 10 steps for sequence modeling
            if len(sequence_buffer) > 10:
                sequence_buffer = sequence_buffer[-10:]

            # Only predict if we have enough context
            if len(sequence_buffer) >= 2:
                # Pad sequence to required length if needed
                seq_len = 10
                if len(sequence_buffer) < seq_len:
                    # Pad with the first observation
                    padding_needed = seq_len - len(sequence_buffer)
                    padded_buffer = [(sequence_buffer[0][0], sequence_buffer[0][1])] * padding_needed + sequence_buffer
                else:
                    padded_buffer = sequence_buffer

                seq_states = jnp.stack([s[0] for s in padded_buffer])
                seq_actions = jnp.stack([s[1] for s in padded_buffer])

                seq_states = seq_states[None, ...]  # Add batch dimension
                seq_actions = seq_actions[None, ...]  # Add batch dimension

                # Split RNG key for model inference
                rng, inference_rng = jax.random.split(rng)
                
                predicted_states = model_fn.apply(
                    dynamics_params, inference_rng, seq_states, seq_actions, training=False
                )
                predicted_next_state = predicted_states[0, -1]

                model_obs = predicted_next_state * state_std + state_mean
            else:
                # Use real observation if not enough context
                model_obs = next_real_obs
        else:
            model_obs = normalized_model_obs * state_std + state_mean

        if steps_into_future > 0:
            debug_obs(step_count, next_real_obs, [model_obs], action)


        # Render real environment
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
        model_text = font.render("Transformer Model", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        pygame.display.flip()


        step_count += 1


def main():
    frame_stack_size = 1

    game = JaxPong()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    env = FlattenObservationWrapper(env)

    save_path = f"transformer_pong.pkl"

    normalization_stats = None
    model_fn = None

    obs = []
    actions = []
    next_obs = []
    rewards = []
    boundaries = []

    experience_path = "experience_data_LSTM_pong_0.pkl"
    with open(experience_path, "rb") as f:
        saved_data = pickle.load(f)
        obs.extend(saved_data["obs"])
        actions.extend(saved_data["actions"])
        next_obs.extend(saved_data["next_obs"])
        rewards.extend(saved_data["rewards"])

        offset = boundaries[-1] if boundaries else 0

        adjusted_boundaries = [b + offset for b in saved_data["boundaries"]]
        boundaries.extend(adjusted_boundaries)

    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}...")
        with open(save_path, "rb") as f:
            saved_data = pickle.load(f)
            dynamics_params = saved_data["dynamics_params"]
            normalization_stats = saved_data.get("normalization_stats", None)
            
        # Create model_fn for loaded model
        def model_fn_inner(states, actions, training=False):
            model = MODEL_ARCHITECTURE()
            return model(states, actions, training=training)
        model_fn = hk.transform(model_fn_inner)
    else:
        print("No existing model found. Training a new model...")

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
        )
        normalization_stats = training_info.get("normalization_stats", None)
        model_fn = training_info["model_fn"]

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

    if len(args := sys.argv) > 2 and args[2] == "render":
        compare_real_vs_model(
            num_steps=1000,
            render_scale=6,
            obs=obs,
            actions=actions,
            normalization_stats=normalization_stats,
            boundaries=boundaries,
            env=env,
            starting_step=0,
            steps_into_future=100,
            render_debugging=(args[3] == "verbose" if len(args) > 3 else False),
            frame_stack_size=frame_stack_size,
            model_params=dynamics_params,
            model_fn=model_fn,
        )


if __name__ == "__main__":

    main()
