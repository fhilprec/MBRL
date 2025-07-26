import os
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Must be set before importing JAX

import pygame
import time
import jax
import jax.numpy as jnp
from jax import lax, random
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
import gc
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import distrax
from rtpt import RTPT


from worldmodel import collect_experience_sequential, train_world_model, get_reward_from_observation, compare_real_vs_model
from ppo_with_rollouts import train_actor_critic, create_actor_critic_network, generate_imagined_rollouts








def main():

    iterations = 3
    frame_stack_size = 1
    
    # Initialize policy parameters and network
    policy_params = None
    network = None
    action_dim = 18
    
    # Hyperparameters for policy training
    rollout_length = 50  # Length of imagined rollouts
    num_rollouts = 10  # Number of rollouts per iteration
    policy_epochs = 20   # Number of policy training epochs
    learning_rate = 3e-4
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # Collect real experience
        game = JaxSeaquest()
        env = AtariWrapper(
            game, sticky_actions=False, episodic_life=False, frame_stack_size=frame_stack_size
        )
        env = FlattenObservationWrapper(env)

        # obs, actions, rewards, _, boundaries = collect_experience_sequential(
        #     env, num_episodes=2, max_steps_per_episode=500, seed=i, policy_params=policy_params, network=network
        # )

        # next_obs = obs[1:] 
        # obs = obs[:-1]  

        # # Train world model
        # print("Training world model...")
        # dynamics_params, training_info = train_world_model(
        #     obs,
        #     actions,
        #     next_obs,
        #     rewards,
        #     episode_boundaries=boundaries,
        #     frame_stack_size=frame_stack_size,
        #     num_epochs=10,
        # )
        # normalization_stats = training_info.get("normalization_stats", None)


        #  #for debugging part ------------------------------------------------------------------------------------------------------------------------------------------
        # with open("model.pkl", "wb") as f:
        #     pickle.dump(
        #         {
        #             "dynamics_params": dynamics_params,
        #             "normalization_stats": training_info.get(
        #                 "normalization_stats", None
        #             ),
        #         },
        #         f,
        #     )
        # with open("experience.pkl", "wb") as f:
        #     pickle.dump(
        #         {
        #             "obs": obs,
        #             "actions": actions,
        #             "next_obs": next_obs,
        #             "rewards": rewards,
        #             "boundaries": boundaries,
        #         },
        #         f,
        #     )

        

        if os.path.exists("model.pkl"):    
            with open("model.pkl", "rb") as f:
                saved_data = pickle.load(f)
                dynamics_params = saved_data["dynamics_params"]
                normalization_stats = saved_data.get("normalization_stats", None)
            with open("experience.pkl", "rb") as f:
                saved_data = pickle.load(f)
                obs = (saved_data["obs"])
                actions = (saved_data["actions"])
                next_obs = (saved_data["next_obs"])
                rewards = (saved_data["rewards"])
        #for debugging part ------------------------------------------------------------------------------------------------------------------------------------------


        # Train policy using imagined rollouts exclusively from the world model
        print("Training policy with imagined rollouts...")
        
        # Create or update actor-critic network
        if network is None:
            obs_shape = obs.shape[1:]  # Get observation shape
            network = create_actor_critic_network(obs_shape, action_dim)
            
            # Initialize policy parameters if first iteration
            key = jax.random.PRNGKey(42)
            dummy_obs = jnp.zeros((1,) + obs_shape)
            policy_params = network.init(key, dummy_obs)


       
        
        
        # Generate imagined rollouts using the world model
        print("Generating imagined rollouts...")
        imagined_obs, imagined_actions, imagined_rewards, imagined_values, imagined_log_probs = generate_imagined_rollouts(
            dynamics_params=dynamics_params,
            policy_params=policy_params,
            network=network,
            initial_observations=obs[:num_rollouts],  # Use real observations as starting points
            rollout_length=rollout_length,
            normalization_stats=normalization_stats,
            key=jax.random.PRNGKey(i * 1000)
        )

        print(imagined_obs.shape)

        # #only take one rollout for comparison
        # single_imagined_obs = imagined_obs[0:1]
        # single_imagined_actions = imagined_actions[0:1]
        # single_imagined_rewards = imagined_rewards[0:1]
        # single_imagined_values = imagined_values[0:1]
        # single_imagined_log_probs = imagined_log_probs[0:1]
        

        # # compare_real_vs_model(
        # #     num_steps = 1000,
        # #     render_scale=6,
        # #     obs=single_imagined_obs,
        # #     actions=single_imagined_actions,
        # #     normalization_stats=normalization_stats,
        # #     boundaries=None,
        # #     env=env,
        # #     starting_step=0,
        # #     steps_into_future=0,
        # #     render_debugging = True,
        # #     frame_stack_size=1
        # # )

        
        # Train actor-critic on imagined rollouts
        print("Training actor-critic...")
        policy_params, training_metrics = train_actor_critic(
            params=policy_params,
            network=network,
            observations=imagined_obs,
            actions=imagined_actions,
            rewards=imagined_rewards,
            values=imagined_values,
            log_probs=imagined_log_probs,
            num_epochs=policy_epochs,
            learning_rate=learning_rate,
            key=jax.random.PRNGKey(i * 2000)
        )
        
        # Print training progress
        if training_metrics:
            print(f"Policy loss: {training_metrics.get('policy_loss', 'N/A'):.4f}")
            print(f"Value loss: {training_metrics.get('value_loss', 'N/A'):.4f}")
            print(f"Mean reward: {jnp.mean(imagined_rewards):.4f}")
        
        # Optional: Save checkpoints
        if i % 1 == 0:
            checkpoint = {
                'policy_params': policy_params,
                'dynamics_params': dynamics_params,
                'normalization_stats': normalization_stats,
                'iteration': i
            }
            with open(f'checkpoint_iter_{i}.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
        
        # Clean up memory
        gc.collect()
        
        print(f"Completed iteration {i+1}")
        print("-" * 50)






if __name__ == "__main__":
        # Create RTPT object
    rtpt = RTPT(name_initials='FH', experiment_name='TestingIterateAgent', max_iterations=3)

    # Start the RTPT tracking
    rtpt.start()    
    main()