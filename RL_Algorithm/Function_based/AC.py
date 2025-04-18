import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import torch.distributions as distributions
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm, Transition
import matplotlib
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):

        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        # ========= put your code here ========= #
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        # self.dropout = nn.Dropout(p=dropout)

        self.init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(state))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)
    
        # ====================================== #

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for Q-value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        # ========= put your code here ========= #
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        # self.dropout = nn.Dropout(p=dropout)

        self.init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state , action): #
        """
        Forward pass for Q-value estimation.

        Args:
            state (Tensor): Current state of the environment.
            action (Tensor): Action taken by the agent.

        Returns:
            Tensor: Estimated Q-value.
        """
        # ========= put your code here ========= #
        x = torch.cat([state, action], dim=1) 
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)
        # ====================================== #

class Actor_Critic(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                action_dim: int = 4,
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        # self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, action_dim, hidden_dim, learning_rate).to(device) 
        # self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)

        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor

        self.update_target_networks(tau=1)  # initialize target networks

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.
        self.episode_durations = []

        pass
        # ====================================== #

        super(Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    def select_action(self, state, noise=0.0):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
                - clipped_action: The action before scaling but after noise adjustment.
        """
        # ========= put your code here ========= #
        state = state['policy']
        action_prob = self.actor(state)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        # Execute action in the environment and observe next state and reward
        action = action.item()
        scaled_action = self.scale_action(action)
        return scaled_action
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        batch = self.memory.sample()
        if batch is None:
            return 
        batch = Transition(*zip(*batch))
        # ====================================== #
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        
        state_batch = torch.cat([s['policy'] for s in batch.state])
        next_state_batch = torch.cat([s['policy'] for s in batch.next_state])

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # ========= put your code here ========= #
        # Update Critic
        # print(actions)
        value = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_value = self.critic(next_states, next_actions)

            td_target = rewards + self.discount_factor * (1 - dones.float()) * next_value
        
        advantage = td_target - value

        # Gradient clipping for critic
        critic_loss = F.mse_loss(value, td_target.detach())
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        # Update Actor
        dist = Categorical(actions)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        actor_loss = -log_prob * advantage.detach()
        
        # Gradient clipping for actor

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)


        return critic_loss, actor_loss
        # ====================================== #

    def update_policy(self, states, actions, rewards, next_states, dones):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        # sample = self.generate_sample(self.batch_size)
        # if sample is None:
        #     print("none")
        #     return
        
        # states, actions, rewards, next_states, dones = sample

        # # Normalize rewards (optional but often helpful)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Compute critic and actor loss
        critic_loss, actor_loss = self.calculate_loss(states, actions, rewards, next_states, dones)
        
        # Backpropagate and update critic network parameters
        self.critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic.optimizer.step()

        # Backpropagate and update actor network parameters
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        return critic_loss, actor_loss

        # ====================================== #


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def learn(self, env, max_steps, num_agents, episode, max_episodes=None, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        if episode == 0:
            self.episode_rewards = []
            self.episode_actor_losses = []
            self.episode_critic_losses = []

        state, _ = env.reset()
        cumulative_reward = 0.0
        done = False
        timestep = 0

        actor_losses = []
        critic_losses = []

        for step in range(max_steps):
            # Predict action from the policy network
            state = state['policy']
            action_prob = self.actor(state)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            # Execute action in the environment and observe next state and reward
            action = action.item()
            scaled_action = self.scale_action(action)
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            reward_value = reward.item()
            cumulative_reward += reward_value

            # Store the transition in memory (not implemented here)
            if num_agents > 1:
                pass
            else:
                pass

            # Decay the noise (not used yet in code)
            done = terminated or truncated

            # Perform one step of the optimization (on the policy network)
            critic_loss, actor_loss = self.update_policy(state, action_prob, reward_value, next_state["policy"], done)

            # Store losses
            actor_losses.append(actor_loss.mean().item())
            critic_losses.append(critic_loss.item())

            # Update state
            state = next_state
            timestep += 1

            if done:
                self.plot_durations(timestep)
                break

        self.episode_rewards.append(cumulative_reward)
        self.episode_actor_losses.append(actor_losses)
        self.episode_critic_losses.append(critic_losses)

        # Every 100 episodes, print summary statistics
        if (episode + 1) % 100 == 0:
            avg_reward = sum(self.episode_rewards[-100:]) / 100
            avg_actor_loss = sum([sum(losses) / len(losses) for losses in self.episode_actor_losses[-100:]]) / 100
            avg_critic_loss = sum([sum(losses) / len(losses) for losses in self.episode_critic_losses[-100:]]) / 100

            print(f"Episode {episode + 1}:")
            print(f"  Avg Reward (last 100): {avg_reward:.3f}")
            print(f"  Avg Actor Loss (last 100): {avg_actor_loss:.5f}")
            print(f"  Avg Critic Loss (last 100): {avg_critic_loss:.5f}")

        # Save logs at the last episode
        if max_episodes:
            if episode == max_episodes - 1:
                with open(self.csv_file_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(["Episode", "Reward", "Actor_Loss", "Critic_Loss", "Duration"])
                    # Write data
                    for i in range(max_episodes):
                        writer.writerow([
                            i,
                            self.episode_rewards[i],
                            self.episode_actor_losses[i],
                            self.episode_critic_losses[i],
                            self.episode_durations[i]
                        ])

    def save_actor_weight(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        
        # Construct the full filepath
        full_path = os.path.join(path, filename if filename.endswith('.pth') else f"{filename}.pth")
        
        # Save the model's state_dict
        torch.save(self.actor.state_dict(), full_path)
        # ====================================== #

    def save_critic_weight(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        
        # Construct the full filepath
        full_path = os.path.join(path, filename if filename.endswith('.pth') else f"{filename}.pth")
        
        # Save the model's state_dict
        torch.save(self.critic.state_dict(), full_path)
        # ====================================== #
        
    def load_actor_weight(self, path, filename):
        """
        Load actor weight parameters.
        """
        # Construct the full filepath
        full_path = os.path.join(path, filename if filename.endswith('.pth') else f"{filename}.pth")

        # Load the model's state_dict
        self.actor.load_state_dict(torch.load(full_path))
        self.actor.eval()  # Set the model to evaluation mode (optional but common)

    def load_critic_weight(self, path, filename):
        """
        Load critic weight parameters.
        """
        # Construct the full filepath
        full_path = os.path.join(path, filename if filename.endswith('.pth') else f"{filename}.pth")

        # Load the model's state_dict
        self.critic.load_state_dict(torch.load(full_path))
        self.critic.eval()  # Set the model to evaluation mode (optional but common)

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