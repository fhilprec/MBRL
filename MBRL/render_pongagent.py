import jax
import jax.numpy as jnp
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
import argparse
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from typing import Any, Tuple

from model_architectures import get_simple_dense_reward
from jaxatari.games.jax_pong import JaxPong
from jaxatari.wrappers import FlattenObservationWrapper, LogWrapper, AtariWrapper
from jaxatari.games.jax_pong import PongRenderer, JaxPong

from pong_agent import create_dreamerv2_actor, create_dreamerv2_critic


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


def load_model(path):
    """Load a previously saved model with proper parameter structure handling."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Raw loaded data type: {type(data)}")
    print(
        f"Raw loaded data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
    )

    # Handle different save formats
    if isinstance(data, dict):
        if "params" in data:
            params = data["params"]
            print(f"Params type: {type(params)}")
            print(
                f"Params keys: {list(params.keys()) if isinstance(params, dict) else 'Not a dict'}"
            )

            # Check if there's an extra "params" layer
            if isinstance(params, dict) and "params" in params:
                print("Removing extra params layer...")
                final_params = params["params"]
            else:
                final_params = params
        else:
            # Data is already the params dict
            final_params = data
    else:
        # Data is already the params dict
        final_params = data

    print(f"Final params type: {type(final_params)}")
    print(
        f"Final params keys: {list(final_params.keys()) if isinstance(final_params, dict) else 'Not a dict'}"
    )

    # Check if final_params has the expected structure for Flax
    if isinstance(final_params, dict):
        # Look for Dense layer parameters
        for key, value in final_params.items():
            print(f"Key: {key}, Type: {type(value)}")
            if isinstance(value, dict):
                print(f"  Subkeys: {list(value.keys())}")

    return final_params


def reset_environment():
    """Set up and reset the Pong environment."""
    # With this (add the missing wrappers):
    env = JaxPong()
    env = AtariWrapper(
        env, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )
    env = FlattenObservationWrapper(env)  # This is missing!
    env = LogWrapper(env)  # This too!

    rng = jax.random.PRNGKey(int(time.time()))
    obs, state = env.reset(rng)
    return env, obs, state


def render_agent(
    actor_model_path,
    critic_model_path=None,
    num_episodes=5,
    fps=60,
    record=False,
    output_path=None,
):
    """
    Render the DreamerV2 agent playing Pong.
    """
    # Initialize environment
    env, obs, state = reset_environment()

    print(f"Observation type: {type(obs)}")
    print(f"State type: {type(state)}")
    print(f"State attributes: {dir(state)}")

    # Debug the state structure
    if hasattr(state, "atari_state"):
        print(f"Atari state type: {type(state.atari_state)}")
        print(f"Atari state attributes: {dir(state.atari_state)}")
        if hasattr(state.atari_state, "env_state"):
            print(f"Env state type: {type(state.atari_state.env_state)}")
            print(f"Env state attributes: {dir(state.atari_state.env_state)}")

    flattened_obs, _ = flatten_obs(obs, single_state=True)
    flattened_obs = np.array(flattened_obs)
    print(f"Flattened observation shape: {flattened_obs.shape}")
    print(f"Observation values: {flattened_obs}")

    # Create networks
    action_dim = 6
    actor_network = create_dreamerv2_actor(action_dim)
    critic_network = create_dreamerv2_critic()

    with open("actor_params.pkl", "rb") as f:
        saved_data = pickle.load(f)
        # Check if it's the old format (direct params) or new format (with "params" key)
        if isinstance(saved_data, dict) and "params" in saved_data:
            loaded_actor_params = saved_data["params"]
        else:
            loaded_actor_params = saved_data  # Old format
        print("Loaded existing actor parameters")

    if critic_model_path:
        print(f"Loading critic model from: {critic_model_path}")
        loaded_critic_params = load_model(critic_model_path)
    else:
        loaded_critic_params = None

    print(f"Actor parameters loaded: {type(loaded_actor_params)}")

    # Test the networks
    print("\n=== TESTING NETWORKS ===")
    try:
        # Ensure flattened_obs has the right shape for testing
        print(flattened_obs.shape)
        if len(flattened_obs.shape) > 1:
            flattened_obs = flattened_obs.squeeze()

        # Convert to JAX array
        test_obs = jnp.array(flattened_obs)

        # Test actor
        pi = actor_network.apply(loaded_actor_params, test_obs)
        print(f"Actor test successful - logits shape: {pi.logits.shape}")
        action = pi.sample(seed=jax.random.PRNGKey(0))
        print(f"Sample action: {action}")

        # Test critic only if we have loaded parameters
        if critic_model_path and loaded_critic_params:
            value = critic_network.apply(loaded_critic_params, test_obs)
            print(f"Critic test successful - value: {value}")
        else:
            print("No critic model provided or critic disabled")

    except Exception as e:
        print(f"Network test failed: {e}")
        if "critic" in str(e).lower() or "Dense_0" in str(e):
            print("Critic test failed, but continuing with actor only...")
            loaded_critic_params = None
        else:
            return

    # Initialize pygame for Pong rendering

    # Jit the policy function with proper error handling
    @jax.jit
    def get_action(params, obs):
        # obs is already flattened due to FlattenObservationWrapper
        flattened_obs, _ = flatten_obs(obs, single_state=True)
        if len(flattened_obs.shape) > 1:
            flattened_obs = flattened_obs.squeeze()
        pi = actor_network.apply(params, flattened_obs)

        return pi.sample(seed=jax.random.PRNGKey(0))

    def get_value(params, obs):
        if params is None:
            return 0.0
        try:
            flattened_obs, _ = flatten_obs(obs, single_state=True)
            flattened_obs = jnp.array(flattened_obs)
            if len(flattened_obs.shape) > 1:
                flattened_obs = flattened_obs.squeeze()
            value = critic_network.apply(params, flattened_obs)
            return value
        except Exception as e:
            print(f"Critic evaluation failed: {e}")
            return 0.0

    # Compile the get_value function only if we have critic params
    if loaded_critic_params is not None:
        get_value = jax.jit(get_value)

    renderer = PongRenderer()

    render_scale = 4

    pygame.init()
    WIDTH = 160
    HEIGHT = 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real Environment vs World Model (Pong)")

    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))

    step_count = 0
    clock = pygame.time.Clock()

    # With this:
    env = JaxPong()
    env = AtariWrapper(
        env, sticky_actions=False, episodic_life=False, frame_stack_size=4
    )

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    obs, state = env.reset(reset_key)
    obs, _ = flatten_obs(obs, single_state=True)

    while True:

        real_raster = renderer.render(state.env_state)
        real_img = np.array(real_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(real_surface, real_img)

        model_raster = renderer.render(state.env_state)
        model_img = np.array(model_raster * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(model_surface, model_img)

        action = get_action(loaded_actor_params, obs)
        reward = get_simple_dense_reward(obs, action, frame_stack_size=4)
        # print("action :", action)
        obs, state, reward, done, _ = env.step(state, action)
        obs, _ = flatten_obs(obs, single_state=True)

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

        if done:
            break

        step_count += 1
        clock.tick(30)

    pygame.quit()
    print("Comparison completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a trained DreamerV2 Pong agent"
    )
    parser.add_argument(
        "--actor",
        type=str,
        required=True,
        help="Path to the actor model checkpoint (actor_params.pkl)",
    )
    parser.add_argument(
        "--critic",
        type=str,
        default=None,
        help="Path to the critic model checkpoint (critic_params.pkl)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run"
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Frames per second for rendering"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record a video of the agent playing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pong_dreamerv2_agent.mp4",
        help="Output path for the recorded video",
    )

    args = parser.parse_args()

    render_agent(
        actor_model_path=args.actor,
        critic_model_path=args.critic,
        num_episodes=args.episodes,
        fps=args.fps,
    )
