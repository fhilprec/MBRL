import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Any, Callable, Tuple, Dict
import numpy as np


def flatten_obs(
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
            flat_state, _ = jax.flatten_util.ravel_pytree(s)  # Extract only the flattened array
            flat_states.append(flat_state)
        flat_states = jnp.stack(flat_states, axis=0)
        print(flat_states.shape)
        return flat_states

    if single_state:
        flat_state, unflattener = jax.flatten_util.ravel_pytree(state)
        return flat_state, unflattener
    batch_shape = state.player_x.shape[0] if hasattr(state, 'player_x') else state.paddle_y.shape[0]

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
    obs_dim: int = 56

    def setup(self):
        self.encoder = Encoder()
        self.rssm = RSSM(self.deter_dim, self.stoch_dim, self.action_dim)
        self.decoder = Decoder(self.obs_dim)
        self.reward_predictor = RewardPredictor()

    def __call__(self, obs_seq, action_seq, key):
        print(obs_seq.shape, action_seq.shape)
        B, T, _ = obs_seq.shape
        embed_seq = jax.vmap(jax.vmap(self.encoder))(obs_seq)

        h = jnp.zeros((B, self.deter_dim))
        z = jnp.zeros((B, self.stoch_dim))

        prior_means, prior_stds = [], []
        post_means, post_stds = [], []
        h_seq, z_seq = [], []

        for t in range(T - 1):
            a = jax.nn.one_hot(action_seq[:, t], self.action_dim)
            embed = embed_seq[:, t + 1]  # t+1 due to next obs
            key, subkey = jax.random.split(key)
            h, prior, post = self.rssm(embed, a, (h, z))
            z = self.rssm.sample(post[0], post[1], subkey)

            prior_means.append(prior[0])
            prior_stds.append(prior[1])
            post_means.append(post[0])
            post_stds.append(post[1])
            h_seq.append(h)
            z_seq.append(z)

        h_seq = jnp.stack(h_seq, axis=1)
        z_seq = jnp.stack(z_seq, axis=1)

        recon_obs = jax.vmap(jax.vmap(self.decoder))(h_seq, z_seq)
        pred_rewards = jax.vmap(jax.vmap(self.reward_predictor))(h_seq, z_seq)

        return {
            'recon_obs': recon_obs,                  # (B, T-1, obs_dim)
            'pred_rewards': pred_rewards.squeeze(-1),# (B, T-1)
            'prior_mean': jnp.stack(prior_means, 1),
            'prior_std': jnp.stack(prior_stds, 1),
            'post_mean': jnp.stack(post_means, 1),
            'post_std': jnp.stack(post_stds, 1),
        }


def kl_divergence(mean1, std1, mean2, std2):
    return 0.5 * jnp.sum(
        (std1**2 + (mean1 - mean2)**2) / (std2**2 + 1e-8) - 1 + 2*(jnp.log(std2 + 1e-8) - jnp.log(std1 + 1e-8)),
        axis=-1
    )


def compute_loss(model_outputs, obs_target, reward_target):
    print(f"reward_target.shape: {reward_target.shape}")
    if reward_target.ndim == 2:
        reward_loss = jnp.mean((model_outputs['pred_rewards'] - reward_target[:, 1:]) ** 2)
    elif reward_target.ndim == 1:
        reward_loss = jnp.mean((model_outputs['pred_rewards'] - reward_target[1:]) ** 2)
    else:
        raise ValueError(f"Unexpected reward_target shape: {reward_target.shape}")
    recon_loss = jnp.mean((model_outputs['recon_obs'] - obs_target[:, 1:]) ** 2)
    kl_loss = jnp.mean(kl_divergence(
        model_outputs['post_mean'], model_outputs['post_std'],
        model_outputs['prior_mean'], model_outputs['prior_std']
    ))
    return recon_loss + reward_loss + 1.0 * kl_loss, {
        'recon_loss': recon_loss,
        'reward_loss': reward_loss,
        'kl_loss': kl_loss
    }


def train_step(params, model, optimizer, opt_state, batch, key):
    def loss_fn(p):
        outputs = model.apply(p, batch['obs'], batch['actions'], key)
        loss, metrics = compute_loss(outputs, batch['obs'], batch['rewards'])
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, metrics

# JIT after definition
train_step_jit = jax.jit(train_step, static_argnames=['model', 'optimizer'])


# Example usage:
if __name__ == "__main__":
    import jax.random as rnd

    key = rnd.PRNGKey(0)
    batch_size = 32
    seq_len = 32
    obs_dim = 14
    action_dim = 6

    model = WorldModel()
    import pickle

    with open("experience_data_LSTM_pong_0.pkl", "rb") as f:
        saved_data = pickle.load(f)

    all_obs = jnp.asarray(saved_data["obs"])            # shape (N, obs_dim)
    all_actions = jnp.asarray(saved_data["actions"])    # shape (N,)
    all_rewards = jnp.asarray(saved_data["rewards"])    # shape (N,)
    boundaries = saved_data["boundaries"]               # list of episode boundaries


    def make_batches(obs, actions, rewards, batch_size):
        B = obs.shape[0]
        for i in range(0, B, batch_size):
            batch_obs = obs[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            batch_rewards = rewards[i:i+batch_size]
            if batch_obs.shape[0] == batch_size:
                yield {
                    "obs": batch_obs,
                    "actions": batch_actions,
                    "rewards": batch_rewards,
                }

    

    T = 32  # or any sequence length you want
    N = all_obs.shape[0]
    B = N // T

    all_obs = all_obs[:B*T].reshape(B, T, model.obs_dim)
    all_actions = all_actions[:B*T].reshape(B, T)

    params = model.init(key, all_obs, all_actions, key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    save_path = f"dreamer_model_pong.pkl"
    import os
    if not os.path.exists(save_path):
        # Training loop
        for step in range(1000):
            key, subkey = rnd.split(key)
            for batch in make_batches(all_obs, all_actions, all_rewards, batch_size):
                params, opt_state, loss, metrics = train_step_jit(params, model, optimizer, opt_state, batch, subkey)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss}, Metrics: {metrics}")

        # Save model and normalization stats
        
       
        with open(save_path, "wb") as f:
            pickle.dump({
                "params": params,
            }, f)
        print(f"Model saved to {save_path}")

    # Compare function
    def compare_dreamer_vs_real(obs, actions, params, model, num_steps=1000, render_scale=3):

        # code to get the unflattener
        from jaxatari.games.jax_pong import JaxPong
        from jaxatari.wrappers import AtariWrapper
        

        game = JaxPong()
        env = AtariWrapper(
            game,
            sticky_actions=False,
            episodic_life=False,
            frame_stack_size=4,
        )
        dummy_obs, _ = env.reset(jax.random.PRNGKey(0))
        _, unflattener = flatten_obs(dummy_obs, single_state=True)




        import jax.random as rnd
        from obs_state_converter import pong_flat_observation_to_state
        key = rnd.PRNGKey(0)
        print('params["params"] keys:', list(params['params'].keys()))
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
        # Use only the current observation for the first batch
        model_obs = obs[0, 0]
        # Initialize RSSM state
        h = jnp.zeros((1, model.deter_dim))
        z = jnp.zeros((1, model.stoch_dim))
        while step_count < min(num_steps, obs.shape[1] - 1):

            action = actions[0, step_count]
            next_real_obs = obs[0, step_count + 1]
            normalized_model_obs = model_obs

            # Encode observation using encoder submodule (direct call)
            embed = Encoder(hidden_dim=128).apply({'params': params['params']['encoder']}, normalized_model_obs[None, :])
            # One-hot action, ensure shape is (1, action_dim)
            a = jax.nn.one_hot(jnp.array([action]), model.action_dim).reshape(1, model.action_dim)
            # Step RSSM using rssm submodule (direct call)
            key, subkey = jax.random.split(key)
            h, prior, post = RSSM(deter_dim=200, stoch_dim=30, action_dim=6).apply({'params': params['params']['rssm']}, embed, a, (h, z))
            z = RSSM(deter_dim=200, stoch_dim=30, action_dim=6).sample(post[0], post[1], subkey)
            # Decode predicted observation using decoder submodule (direct call)
            pred_obs = Decoder(output_dim=model.obs_dim, hidden_dim=128).apply({'params': params['params']['decoder']}, h, z)[0]
            pred_obs = jnp.round(pred_obs)
            model_obs = pred_obs
            print(next_real_obs)
            print(model_obs)
            # Render
            real_state = pong_flat_observation_to_state(next_real_obs, unflattener, frame_stack_size=4)
            real_img = np.array(renderer.render(real_state) * 255, dtype=np.uint8)
            pygame.surfarray.blit_array(real_surface, real_img)
            model_state = pong_flat_observation_to_state(model_obs, unflattener, frame_stack_size=4)
            model_img = np.array(renderer.render(model_state) * 255, dtype=np.uint8)
            pygame.surfarray.blit_array(model_surface, model_img)
            screen.fill((0, 0, 0))
            scaled_real = pygame.transform.scale(real_surface, (WIDTH * render_scale, HEIGHT * render_scale))
            screen.blit(scaled_real, (0, 0))
            scaled_model = pygame.transform.scale(model_surface, (WIDTH * render_scale, HEIGHT * render_scale))
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

    # Load and compare
    with open(save_path, "rb") as f:
        saved = pickle.load(f)
        params = saved["params"]
    compare_dreamer_vs_real(all_obs, all_actions, params, model)