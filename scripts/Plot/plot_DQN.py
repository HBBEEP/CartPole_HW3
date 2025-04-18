import pandas as pd
import matplotlib.pyplot as plt

# File paths for each experiment
file_paths = {
    "Experiment 1": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/experiment_1.csv",
    "Experiment 2": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/experiment_2.csv",
    "Experiment 3": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/experiment_3.csv",
}

# Settings
metrics = ["Reward", "Loss", "Duration"]
colors = ["tab:blue", "tab:green", "tab:red"]
window_size = 10  # Smoothing window

# Loop over each metric and create a separate plot
for metric_index, metric in enumerate(metrics):
    plt.figure(figsize=(10, 5))

    for j, (label, path) in enumerate(file_paths.items()):
        df = pd.read_csv(path)
        episodes = df["Episode"]
        values = df[metric]

        # Compute moving average and std deviation
        moving_avg = values.rolling(window=window_size).mean()
        moving_std = values.rolling(window=window_size).std()

        # Plot with error band
        plt.plot(episodes, moving_avg, label=label, color=colors[j])
        plt.fill_between(
            episodes,
            moving_avg - moving_std,
            moving_avg + moving_std,
            color=colors[j],
            alpha=0.2
        )

    plt.title(f"DQN : {metric} Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
