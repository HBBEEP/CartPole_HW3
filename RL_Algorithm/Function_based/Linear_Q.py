from __future__ import annotations
import torch
import numpy as np
import csv
from RL_Algorithm.RL_base_function import BaseAlgorithm
import matplotlib
import matplotlib.pyplot as plt

class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
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
        self.episode_durations = []

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action, # : int,
        terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #
        # phi_current = obs['policy'].cpu().numpy()
        # phi_next = next_obs['policy']

        # q_sa = phi_current @ self.w[:, action]

        # # Compute TD target
        # if terminated:
        #     target = reward
        # else:
        #     q_next = self.q(next_obs)
        #     target = reward + self.discount_factor * np.max(q_next)

        # # TD error
        # td_error = target - q_sa.item()
        
        # # Update weights
        # self.w[:, action] += self.lr * td_error * phi_current.reshape(-1)

        # self.training_error.append(td_error)
        # phi_current = obs['policy'].squeeze().cpu().numpy()
        # q_sa = float(phi_current @ self.w[:, action])

        # # Update weights
        # self.w[:, action] += self.lr * td_error * phi_current
        

        phi_current = obs['policy'].cpu().numpy()
        q_sa = float(phi_current @ self.w[:, action])

        next_action_tensor, _ = self.select_action(next_obs)
        reward = np.clip(reward, -1, 1)
        if terminated:
            target = reward
        else:
            q_next = self.q(next_obs).flatten() 
            target = reward + self.discount_factor * q_next[next_action_tensor]

        td_error = target - q_sa
        # print(q_sa)
        self.w[:, action] += self.lr * td_error * phi_current.reshape(-1)
        self.training_error.append(td_error)


        # ====================================== #

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
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_of_action)
            scaled_action = self.scale_action(action)

            return  action, scaled_action
        
        # state = state['policy']
        q_values = self.q(state)
        action = int(np.argmax(q_values))
        scaled_action = self.scale_action(action)

        return  action, scaled_action
        # ====================================== #

    def learn(self, env, max_steps, episode, max_episodes = None):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        # pass

        if episode == 0:
            self.episode_rewards = []

        state, _ = env.reset()
        timestep = 0
        done = False
        cumulative_reward = 0
        while not done:

            action_tensor, scaled_action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)

            reward_value = reward.item()
            cumulative_reward += reward_value
            self.update(state, action_tensor, reward_value, next_state, None, terminated )
            done = terminated or truncated
            state = next_state

            timestep += 1

            if done:
                self.plot_durations(timestep)
                break

        self.episode_rewards.append(cumulative_reward)

        if (episode ) % 100 == 0:
            # self.update_target_networks()

            avg_reward = sum(self.episode_rewards[-100:]) / 100
            avg_error = sum(self.training_error[-100:]) / 100
            print(f"Episode {episode + 1}: Mean Reward (last 100 eps): {avg_reward:.2f}")
            print(f"Episode {episode + 1}: Mean Error (last 100 eps): {avg_error:.2f}")

        # ====================================== #
    
                # Save logs at the last episode
        if max_episodes:
            if episode == max_episodes - 1:
                with open(self.csv_file_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(["Episode", "Reward", "Training_error", "Duration"])
                    # Write data
                    for i in range(max_episodes):
                        writer.writerow([
                            i,
                            self.episode_rewards[i],
                            self.training_error[i],
                            self.episode_durations[i]
                        ])
    def set_csv_file_path(self, file_path):
        self.csv_file_path = file_path

    def save_weights(self, filename="weights.npy"):
        """Save the weight matrix to a file."""
        np.save(filename, self.w)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename="weights.npy"):
        """Load the weight matrix from a file."""
        try:
            self.w = np.load(filename)
            print(f"Weights loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. Returning None.")
    
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