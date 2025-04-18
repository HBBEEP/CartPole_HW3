from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm, Transition

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import csv
import random
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
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
            x (Tensor): Input state tensor.
        
        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        x = x.to(device)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)

        # ====================================== #

class DQN(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 5,
            action_range: list = [-5, 5],
            n_observations: int = 4,
            hidden_dim: int = 128,
            dropout: float = 0.05,
            learning_rate: float = 0.0001,
            tau: float = 0.005,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.9995,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
            buffer_size: int = 10000,
            batch_size: int = 128,
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
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())


        self.device = device
        self.steps_done = 0
        self.num_of_action = num_of_action
        self.tau = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.episode_durations = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,  
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        prob = np.random.rand()
        # print(f"prob : {prob} | self.epsilon : {self.epsilon}")

        if prob < self.epsilon:
            # Random discrete action
            action = np.random.choice(self.num_of_action)
            scaled_action = self.scale_action(action)

            action_tensor = torch.tensor([[action]], device=self.device, dtype=torch.long)
            return action_tensor, scaled_action
        else:
            state_tensor = state['policy']
            with torch.no_grad():
                action_tensor = self.policy_net(state_tensor).max(1)[1].view(1, 1)

            action = action_tensor.item()
            scaled_action = self.scale_action(action)

            return action_tensor, scaled_action
        
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            non_final_mask (Tensor): Mask indicating which states are non-final.
            non_final_next_states (Tensor): The next states that are not terminal.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of actions taken.
            reward_batch (Tensor): Batch of received rewards.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size , device=device)

        # Compute next Q values for non-terminal states using the target network
        if len(non_final_next_states) > 0:
            next_q_values = self.target_net(non_final_next_states).max(1)[0].detach().to(device)
            next_state_values[non_final_mask] = next_q_values

        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach().to(device)
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss

        # ====================================== #

    def generate_sample(self, ): # batch_size
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - non_final_mask (Tensor): A boolean mask indicating which states are non-final.
                - non_final_next_states (Tensor): The next states that are not terminal.
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        batch = self.memory.sample()
        if batch is None:
            return 
        batch = Transition(*zip(*batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s['policy'] for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat([s['policy'] for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch
        # ====================================== #

    # Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
    
    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # Generate a sample batch
        sample = self.generate_sample() # self.batch_size
        if sample is None:
            return
        
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        
        # Compute loss
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # Perform gradient descent step
        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        # ====================================== #

    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #
        policy_state_dict = self.policy_net.state_dict()
        target_state_dict = self.target_net.state_dict()
        # ====================================== #
        
        # Apply the soft update rule to each parameter in the target network
        # ========= put your code here ========= #
        for key in policy_state_dict:
            target_state_dict[key] = (
                (1.0 - self.tau) * target_state_dict[key] + self.tau * policy_state_dict[key]
            )
        # ====================================== #
        
        # Load the updated weights into the target network
        # ========= put your code here ========= #
        # self.target_net.load_state_dict(target_state_dict)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.policy_net.train()
        # ====================================== #

    def dict_tensor_equal(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        return all(torch.equal(dict1[k], dict2[k]) for k in dict1)

    def same_move(self, state, next_state, last_memory):
        return self.dict_tensor_equal(state, last_memory.state) and self.dict_tensor_equal(next_state, last_memory.next_state)

    def learn(self, env, episode, max_episodes = None):
        """
        Train the agent on a single step.
        Args:
            env: The environment to train in.
        """
        # Initialize tracking lists (only once at start)
        if episode == 0:
            self.episode_rewards = []
            self.episode_losses = []

        state, _ = env.reset()
        done = False
        cumulative_reward = 0
        timestep = 0
        episode_loss_values = []

        while not done:
            action_tensor, scaled_action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)

            reward_value = reward.item()
            cumulative_reward += reward_value

            if next_state == None or len(self.memory) == 0 or not self.same_move(state, next_state, self.memory.memory[-1]):
                self.memory.add(state, action_tensor, reward, next_state, done)

            done = terminated or truncated
            state = next_state

            loss = self.update_policy()
            if loss is not None:
                loss_value = loss.item()
                episode_loss_values.append(loss_value)

            # if episode % 100 == 0:

            timestep += 1

            if done:
                self.plot_durations(timestep)
                break

        # After episode ends, record reward and loss
        self.episode_rewards.append(cumulative_reward)
        self.episode_losses.append(
            sum(episode_loss_values) / len(episode_loss_values) if episode_loss_values else 0.0
        )
    
        # Print every 100 episodes
        if (episode ) % 100 == 0:
            self.update_target_networks()

            avg_reward = sum(self.episode_rewards[-100:]) / 100
            avg_loss = sum(self.episode_losses[-100:]) / 100
            print(f"Episode {episode}: Mean Reward (last 100 eps): {avg_reward:.2f}, Mean Loss: {avg_loss:.4f}")

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