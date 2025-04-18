"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    # num_of_action = 10
    # action_range = [-10, 10]  
    # learning_rate = 1e-3
    # hidden_dim = 128
    # n_episodes = 10000
    # initial_epsilon = 0.00
    # epsilon_decay = 0.9999
    # final_epsilon = 0.05
    # discount = 0.99
    # buffer_size = 50000
    # batch_size = 64

    num_of_action = 4            
    action_range = [-5, 5]  
    learning_rate = 0.01 # 0.01
    hidden_dim =  128 # 256
    n_episodes = 2000              # Usually enough for convergence
    initial_epsilon = 0.00
    epsilon_decay = 0.9995         # Faster decay is better here
    final_epsilon = 0.01
    discount = 0.95                # Slightly higher for better long-term return
    buffer_size = 10000            # Smaller is fine for CartPole
    batch_size = 128
    max_episodes = 20
    
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    agent = DQN(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        initial_epsilon = initial_epsilon,
        epsilon_decay = epsilon_decay,
        final_epsilon = final_epsilon,
        discount_factor = discount,
        buffer_size = buffer_size,
        batch_size = batch_size,
    )

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "DQN"  
    Algorithm_ex = "DQN_EX3"
    episode_selected = 1900
    q_value_file = f"{Algorithm_name}_{episode_selected}_{num_of_action}_{action_range[1]}.pth"
    full_path = os.path.join(f"weight/{task_name}", Algorithm_ex)
    agent.load_w(full_path, q_value_file)

    csv_file_path = "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/play_experiment_3_epi_1900.csv"
    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode

        episode_rewards = []

        with torch.inference_mode():

            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0

                while not done:
                    # agent stepping
                    action_tensor, scaled_action = agent.select_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
                    reward_value = reward.item()
                    cumulative_reward += reward_value

                    done = terminated or truncated
                    obs = next_obs

                episode_rewards.append(cumulative_reward)
                print(f"cumulative_reward : {cumulative_reward}")
                if episode == max_episodes - 1:
                    with open(csv_file_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        # Write header
                        writer.writerow(["Episode", "Reward"])
                        # Write data
                        for i in range(max_episodes):
                            writer.writerow([
                                i,
                                episode_rewards[i],
                            ])

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()