import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Any, Callable, Tuple, Dict
import numpy as np

frame_stack_size = 1


action_map = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}


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


class Encoder(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs):
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        return nn.Dense(self.hidden_dim)(x)


class RSSM(nn.Module):
    deter_dim: int = 200
    stoch_dim: int = 30
    action_dim: int = 6

    @nn.compact
    def __call__(self, embed, action, prev_state):
        h_prev, z_prev = prev_state
        x = jnp.concatenate([z_prev, action], axis=-1)
        h = nn.GRUCell(self.deter_dim)(h_prev, x)[0]

        prior_mean = nn.Dense(self.stoch_dim)(h)
        prior_std = nn.softplus(nn.Dense(self.stoch_dim)(h)) + 1e-3
        prior = (prior_mean, prior_std)

        x_post = jnp.concatenate([h, embed], axis=-1)
        post_mean = nn.Dense(self.stoch_dim)(x_post)
        post_std = nn.softplus(nn.Dense(self.stoch_dim)(x_post)) + 1e-3
        posterior = (post_mean, post_std)

        return h, prior, posterior

    def sample(self, mean, std, key):
        return mean + std * jax.random.normal(key, mean.shape)


class Decoder(nn.Module):
    output_dim: int = 14
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, h, z):
        x = jnp.concatenate([h, z], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)


class RewardPredictor(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, h, z):
        x = jnp.concatenate([h, z], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(1)(x)


class WorldModel(nn.Module):
    deter_dim: int = 200
    stoch_dim: int = 30
    action_dim: int = 6
    obs_dim: int = 14 * frame_stack_size

    def setup(self):
        self.encoder = Encoder()
        self.rssm = RSSM(self.deter_dim, self.stoch_dim, self.action_dim)
        self.decoder = Decoder(self.obs_dim)
        self.reward_predictor = RewardPredictor()

    def __call__(self, obs_seq, action_seq, key, open_loop_start: int = 1):
        """
        obs_seq: (B, T, obs_dim)
        action_seq: (B, T)
        open_loop_start: first time-step index (t) where we stop conditioning on the
                         posterior (i.e. use prior-only sampling). Typical: 1..(T-1).

        Behavior:
         - For t < open_loop_start: perform the usual posterior update using embed(t+1)
         - For t >= open_loop_start: use prior-only (imagination / open-loop) updates

        Returns reconstructions and diagnostics similar to before.
        """
        B, T, _ = obs_seq.shape
        embed_seq = jax.vmap(jax.vmap(self.encoder))(obs_seq)

        h = jnp.zeros((B, self.deter_dim))
        z = jnp.zeros((B, self.stoch_dim))

        prior_means, prior_stds = [], []
        post_means, post_stds = [], []
        h_seq, z_seq = [], []

        for t in range(T - 1):
            a = jax.nn.one_hot(action_seq[:, t], self.action_dim)

            x = jnp.concatenate([z, a], axis=-1)

            h_candidate = (
                nn.GRUCell(self.rssm.deter_dim).apply({"params": {}}, h, x)[0]
                if False
                else None
            )

            h_next, prior, posterior = self.rssm(embed_seq[:, t + 1], a, (h, z))

            key, subkey = jax.random.split(key)

            use_posterior = t < open_loop_start

            if use_posterior:
                z_next = self.rssm.sample(posterior[0], posterior[1], subkey)
            else:
                z_next = self.rssm.sample(prior[0], prior[1], subkey)

            prior_means.append(prior[0])
            prior_stds.append(prior[1])
            post_means.append(posterior[0])
            post_stds.append(posterior[1])

            h_seq.append(h_next)
            z_seq.append(z_next)

            h = h_next
            z = z_next

        h_seq = jnp.stack(h_seq, axis=1)
        z_seq = jnp.stack(z_seq, axis=1)

        recon_obs = jax.vmap(jax.vmap(self.decoder))(h_seq, z_seq)
        pred_rewards = jax.vmap(jax.vmap(self.reward_predictor))(h_seq, z_seq)

        return {
            "recon_obs": recon_obs,
            "pred_rewards": pred_rewards.squeeze(-1),
            "prior_mean": jnp.stack(prior_means, 1),
            "prior_std": jnp.stack(prior_stds, 1),
            "post_mean": jnp.stack(post_means, 1),
            "post_std": jnp.stack(post_stds, 1),
        }


def kl_divergence(mean1, std1, mean2, std2):
    return 0.5 * jnp.sum(
        (std1**2 + (mean1 - mean2) ** 2) / (std2**2 + 1e-8)
        - 1
        + 2 * (jnp.log(std2 + 1e-8) - jnp.log(std1 + 1e-8)),
        axis=-1,
    )


def compute_loss(model_outputs, obs_target, reward_target):
    if reward_target.ndim == 2:
        reward_loss = jnp.mean(
            (model_outputs["pred_rewards"] - reward_target[:, 1:]) ** 2
        )
    elif reward_target.ndim == 1:
        reward_loss = jnp.mean((model_outputs["pred_rewards"] - reward_target[1:]) ** 2)
    else:
        raise ValueError(f"Unexpected reward_target shape: {reward_target.shape}")
    recon_loss = jnp.mean((model_outputs["recon_obs"] - obs_target[:, 1:]) ** 2)
    kl_loss = jnp.mean(
        kl_divergence(
            model_outputs["post_mean"],
            model_outputs["post_std"],
            model_outputs["prior_mean"],
            model_outputs["prior_std"],
        )
    )
    return recon_loss + reward_loss + 1.0 * kl_loss, {
        "recon_loss": recon_loss,
        "reward_loss": reward_loss,
        "kl_loss": kl_loss,
    }


def train_step(
    params, model, optimizer, opt_state, batch, key, open_loop_start: int = 1
):
    def loss_fn(p):
        outputs = model.apply(p, batch["obs"], batch["actions"], key, open_loop_start)
        loss, metrics = compute_loss(outputs, batch["obs"], batch["rewards"])
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, metrics


train_step_jit = jax.jit(
    train_step, static_argnames=["model", "optimizer", "open_loop_start"]
)


def compare_dreamer_vs_real(
    obs,
    actions,
    params,
    model,
    num_steps=1000,
    render_scale=3,
    steps_into_future=100,
):

    from jaxatari.games.jax_pong import JaxPong
    from jaxatari.wrappers import AtariWrapper

    def debug_obs(
        step,
        real_obs,
        pred_obs,
        action,
    ):
        error = jnp.mean((real_obs - pred_obs) ** 2)
        print(
            f"Step {step}, Unnormalized Error: {error:.2f} | Action: {action_map.get(int(action), action)}"
        )

    game = JaxPong()
    env = AtariWrapper(
        game,
        sticky_actions=False,
        episodic_life=False,
        frame_stack_size=frame_stack_size,
    )
    dummy_obs, _ = env.reset(jax.random.PRNGKey(0))
    _, unflattener = flatten_obs(dummy_obs, single_state=True)

    import jax.random as rnd
    from obs_state_converter import pong_flat_observation_to_state

    key = rnd.PRNGKey(0)
    print('params["params"] keys:', list(params["params"].keys()))
    import pygame
    from jaxatari.games.jax_pong import PongRenderer

    pygame.init()
    renderer = PongRenderer()
    WIDTH, HEIGHT = 160, 210
    WINDOW_WIDTH = WIDTH * render_scale * 2 + 20
    WINDOW_HEIGHT = HEIGHT * render_scale
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real vs Dreamer Model (Pong)")
    real_surface = pygame.Surface((WIDTH, HEIGHT))
    model_surface = pygame.Surface((WIDTH, HEIGHT))
    step_count = 0
    clock = pygame.time.Clock()

    model_obs = obs[0]

    h = jnp.zeros((1, model.deter_dim))
    z = jnp.zeros((1, model.stoch_dim))
    while step_count < num_steps:

        action = actions[step_count]
        next_real_obs = obs[step_count + 1]
        normalized_model_obs = model_obs

        embed = Encoder(hidden_dim=128).apply(
            {"params": params["params"]["encoder"]}, normalized_model_obs[None, :]
        )
        a = jax.nn.one_hot(jnp.array([action]), model.action_dim).reshape(
            1, model.action_dim
        )

        key, subkey = jax.random.split(key)
        h, prior, post = RSSM(deter_dim=200, stoch_dim=30, action_dim=6).apply(
            {"params": params["params"]["rssm"]}, embed, a, (h, z)
        )
        z = RSSM(deter_dim=200, stoch_dim=30, action_dim=6).sample(
            post[0], post[1], subkey
        )

        if step_count % 100 == 0:

            print("Model Reset!")
            model_obs = obs[step_count]
            h = jnp.zeros((1, model.deter_dim))
            z = jnp.zeros((1, model.stoch_dim))

        pred_obs = Decoder(output_dim=model.obs_dim, hidden_dim=128).apply(
            {"params": params["params"]["decoder"]}, h, z
        )[0]
        pred_obs = jnp.round(pred_obs)
        model_obs = pred_obs
        debug_obs(step_count, next_real_obs, model_obs, action)

        real_state = pong_flat_observation_to_state(
            next_real_obs, unflattener, frame_stack_size=frame_stack_size
        )
        real_img = np.array(renderer.render(real_state) * 255, dtype=np.uint8)
        pygame.surfarray.blit_array(real_surface, real_img)
        model_state = pong_flat_observation_to_state(
            model_obs, unflattener, frame_stack_size=frame_stack_size
        )
        model_img = np.array(renderer.render(model_state) * 255, dtype=np.uint8)
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
        real_text = font.render("Real Env", True, (255, 255, 255))
        model_text = font.render("Dreamer Model", True, (255, 255, 255))
        screen.blit(real_text, (20, 10))
        screen.blit(model_text, (WIDTH * render_scale + 40, 10))
        pygame.display.flip()
        step_count += 1
        clock.tick(10)
    pygame.quit()
    print("Comparison completed")


if __name__ == "__main__":
    import jax.random as rnd

    epochs = 2000

    key = rnd.PRNGKey(0)
    batch_size = 32
    seq_len = 32
    obs_dim = 14 * frame_stack_size
    action_dim = 6

    model = WorldModel()
    import pickle

    with open("experience_data_LSTM_pong_0.pkl", "rb") as f:
        saved_data = pickle.load(f)

    all_obs = jnp.asarray(saved_data["obs"])
    all_actions = jnp.asarray(saved_data["actions"])
    all_rewards = jnp.asarray(saved_data["rewards"])
    boundaries = saved_data["boundaries"]

    def make_batches(obs, actions, rewards, batch_size):
        B = obs.shape[0]
        for i in range(0, B, batch_size):
            batch_obs = obs[i : i + batch_size]
            batch_actions = actions[i : i + batch_size]
            batch_rewards = rewards[i : i + batch_size]
            if batch_obs.shape[0] == batch_size:
                yield {
                    "obs": batch_obs,
                    "actions": batch_actions,
                    "rewards": batch_rewards,
                }

    T = 32
    sequential_obs = all_obs.copy()
    sequential_actions = all_actions.copy()
    N = all_obs.shape[0]
    B = N // T

    all_obs = all_obs[: B * T].reshape(B, T, model.obs_dim)
    all_actions = all_actions[: B * T].reshape(B, T)

    params = model.init(key, all_obs, all_actions, key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    save_path = f"dreamer_model_pong.pkl"
    import os

    if not os.path.exists(save_path):

        for step in range(epochs):
            key, subkey = rnd.split(key)
            for batch in make_batches(all_obs, all_actions, all_rewards, batch_size):
                open_loop_start = int((step / epochs) * 15)
                params, opt_state, loss, metrics = train_step_jit(
                    params, model, optimizer, opt_state, batch, subkey, open_loop_start
                )
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss}, Metrics: {metrics}")

        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "params": params,
                },
                f,
            )
        print(f"Model saved to {save_path}")

    with open(save_path, "rb") as f:
        saved = pickle.load(f)
        params = saved["params"]
    compare_dreamer_vs_real(
        sequential_obs, sequential_actions, params, model, steps_into_future=100
    )
