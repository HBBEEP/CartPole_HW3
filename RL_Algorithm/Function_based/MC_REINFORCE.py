from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import csv
import random
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_actions)
        self.dropout = nn.Dropout(p=dropout)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        x = x.to(device)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        out = self.out(x)
        action_prob = F.softmax(out, dim=-1)
        return action_prob
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0

        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    
    def calculate_stepwise_returns(self, rewards):
        """
        Compute stepwise returns for the trajectory.

        Args:
            rewards (list): List of rewards obtained in the episode.
        
        Returns:
            Tensor: Normalized stepwise returns.
        """
        # ========= put your code here ========= #
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + R * self.discount_factor
            returns.insert(0, R)
        returns = torch.tensor(returns)
        normalized_returns = (returns - returns.mean()) / returns.std()

        if returns.numel() > 1:
            normalized_returns = (returns - returns.mean()) / returns.std()
        else:
            normalized_returns = returns
        # print(f"::: {returns}")
        return normalized_returns

        # ====================================== #

    def select_action(self, state):
        state = state['policy']
        action_prob = self.policy_net(state)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        action = action.item()
        scaled_action = self.scale_action(action)
        return scaled_action
    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple: (episode_return, stepwise_returns, log_prob_actions, trajectory)
        """
        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Store state-action-reward history (list)  ????
        # Store log probabilities of actions (list)  ????
        # Store rewards at each step (list)
        # Track total episode return (float) 
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        log_prob_actions = []
        rewards = []
        trajectory = []
        done = False
        timestep = 0
        episode_return = 0
        # ====================================== #
        state, _ = env.reset()
        self.policy_net.train()

        # ===== Collect trajectory through agent-environment interaction ===== #
        while not done:
            
            # Predict action from the policy network
            # ========= put your code here ========= #
            state = state['policy']
            action_prob = self.policy_net(state)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            action = action.item()
            scaled_action = self.scale_action(action)
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated
            # ====================================== #

            # Store action log probability reward and trajectory history
            # ========= put your code here ========= #
            log_prob_actions.append(log_prob_action)
            rewards.append(reward)
            episode_return += reward
            # after env.step and reward:
            trajectory.append((state, action, reward, next_state, done))

            # ====================================== #
            
            # Update state
            state = next_state

            timestep += 1
            if done:
                self.plot_durations(timestep)
                break



        # ===== Stack log_prob_actions &  stepwise_returns ===== #
        # ========= put your code here ========= #
        log_prob_actions = torch.cat(log_prob_actions)
        stepwise_returns = self.calculate_stepwise_returns(rewards)

        return episode_return, stepwise_returns, log_prob_actions, trajectory
        # ====================================== #
    
    def calculate_loss(self, stepwise_returns, log_prob_actions):
        """
        Compute the loss for policy optimization.

        Args:
            stepwise_returns (Tensor): Stepwise returns for the trajectory.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        # print("stepwise_returns device:", stepwise_returns.device)
        # print("log_prob_actions device:", log_prob_actions.device)
        stepwise_returns = stepwise_returns.to(device)
        loss = -(stepwise_returns * log_prob_actions).sum()
        return loss
        # ====================================== #

    def update_policy(self, stepwise_returns, log_prob_actions):
        """
        Update the policy using the calculated loss.

        Args:
            stepwise_returns (Tensor): Stepwise returns.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        stepwise_returns = stepwise_returns.detach()
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        # ====================================== #
    
    def learn(self, env, episode, max_episodes = None):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        # ========= put your code here ========= #
        if episode == 0:
            self.episode_rewards = []
            self.episode_losses = []

        self.policy_net.train()
        episode_return, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return_value = episode_return.item()
        # After episode ends, record reward and loss
        self.episode_rewards.append(return_value)
        self.episode_losses.append(loss)
        
        if (episode ) % 100 == 0:

            avg_reward = sum(self.episode_rewards[-100:]) / 100
            avg_loss = sum(self.episode_losses[-100:]) / 100
            print(f"Episode {episode + 1}: Mean Reward (last 100 eps): {avg_reward:.2f}, Mean Loss: {avg_loss:.4f}")

        # Save logs at the last episode
        if max_episodes:
            if episode == max_episodes - 1:
                with open(self.csv_file_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(["Episode", "Reward", "Loss", "Duration"])
                    # Write data
                    for i in range(max_episodes):
                        writer.writerow([
                            i,
                            self.episode_rewards[i],
                            self.episode_losses[i],
                            self.episode_durations[i]
                        ])

        return episode_return, loss, trajectory
        # ====================================== #

    def set_csv_file_path(self, file_path):
        self.csv_file_path = file_path

    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #
