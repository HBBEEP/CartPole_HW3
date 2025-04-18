import csv

file_paths = {
    "Linear_Q_1": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/Linear_Q/play_experiment_1_epi_2000.csv",
    "Linear_Q_2": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/Linear_Q/play_experiment_2_epi_2000.csv",
    "Linear_Q_3": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/Linear_Q/play_experiment_3_epi_2000.csv",
    "DQN_1": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/play_experiment_1_epi_1900.csv",
    "DQN_2": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/play_experiment_2_epi_1900.csv",
    "DQN_3": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/play_experiment_3_epi_1900.csv",
    "DQN_B_1": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/play_experiment_1_epi_1200.csv",
    "DQN_B_2": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/play_experiment_2_epi_1200.csv",
    "DQN_B_3": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/DQN/play_experiment_3_epi_1200.csv",
    "REINFORCE_1": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/MC_REINFORCE/play_experiment_1_epi_2000.csv",
    "REINFORCE_2": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/MC_REINFORCE/play_experiment_2_epi_2000.csv",
    "REINFORCE_3": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/MC_REINFORCE/play_experiment_3_epi_2000.csv",
    "AC_1": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/AC/play_experiment_1_epi_2000.csv",
    "AC_2": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/AC/play_experiment_2_epi_2000.csv",
    "AC_3": "/home/hbbeep-p/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.5.0/result/Stabilize/AC/play_experiment_3_epi_2000.csv",
}

all_rewards = []

for name, path in file_paths.items():
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rewards = [float(row['Reward']) for row in reader]
            all_rewards.extend(rewards)
            print(f"{name}: {len(rewards)} rewards, Mean = {sum(rewards) / len(rewards):.6f}")
    except Exception as e:
        print(f"Error reading {name} ({path}): {e}")
