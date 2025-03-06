import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import pandas as pd
import seaborn as sns
import gym
import gym_donkeycar
from IPython.display import clear_output
from stable_baselines3.common.callbacks import BaseCallback

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

class TrainingStats:
    """
    Simple class to track and visualize training statistics
    """
    def __init__(self, verbose=1, live_plot=False):  # Default live_plot to False
        self.verbose = verbose
        self.live_plot = live_plot
        self.training_start = time.time()
        self.timestamp = int(time.time())
        self.csv_path = f"logs/donkey_stats_{self.timestamp}.csv"
        
        # Stats tracking
        self.timesteps = 0
        self.episodes = 0
        self.timesteps_list = []
        self.rewards_list = []
        self.lengths_list = []
        self.reward_buffer = []
        self.length_buffer = []
        
        # Keep buffer size reasonable
        self.buffer_size = 100
        
        # Create CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timesteps', 'episodes', 'time_elapsed',
                'mean_reward', 'mean_episode_length', 'fps'
            ])
    
    def log_episode(self, episode_reward, episode_length):
        """Record stats for a completed episode"""
        print(f"Logging episode {self.episodes+1} with reward={episode_reward:.2f}, length={episode_length}")
        
        self.episodes += 1
        self.timesteps += episode_length
        
        # Update buffers
        self.reward_buffer.append(episode_reward)
        self.length_buffer.append(episode_length)
        
        # Keep buffer to a reasonable size
        if len(self.reward_buffer) > self.buffer_size:
            self.reward_buffer = self.reward_buffer[-self.buffer_size:]
            self.length_buffer = self.length_buffer[-self.buffer_size:]
        
        # Calculate stats
        time_elapsed = time.time() - self.training_start
        mean_reward = np.mean(self.reward_buffer)
        mean_length = np.mean(self.length_buffer)
        fps = self.timesteps / time_elapsed if time_elapsed > 0 else 0
        
        # Save to CSV
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.timesteps,
                    self.episodes,
                    time_elapsed,
                    mean_reward,
                    mean_length,
                    fps
                ])
            print(f"Successfully wrote to CSV: {self.csv_path}")
        except Exception as e:
            print(f"Error writing to CSV: {e}")
        
        # Store for live plotting
        self.timesteps_list.append(self.timesteps)
        self.rewards_list.append(mean_reward)
        self.lengths_list.append(mean_length)
        
        # Print stats
        if self.verbose > 0:
            print(f"Episode {self.episodes} | "
                  f"Steps: {self.timesteps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Mean reward: {mean_reward:.2f} | "
                  f"Length: {episode_length}")
        
        # Update live plot if enabled
        if self.live_plot and len(self.timesteps_list) > 1:
            self.plot_live_progress()
    
    def plot_live_progress(self):
        """Create live training progress plot in the notebook"""
        clear_output(wait=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        ax1.plot(self.timesteps_list, self.rewards_list, 'b-', label='Mean Reward')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Training Progress')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(self.timesteps_list, self.lengths_list, 'r-', label='Mean Episode Length')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Mean Episode Length')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print current status
        print(f"Episodes: {self.episodes} | Total steps: {self.timesteps}")
        print(f"Training time: {(time.time() - self.training_start)/60:.1f} minutes")

class StatsCallback(BaseCallback):
    """
    Callback for saving statistics during training
    """
    def __init__(self, stats_tracker, check_freq=1, verbose=0):
        super(StatsCallback, self).__init__(verbose)
        self.stats = stats_tracker
        self.check_freq = check_freq
        self.episode_reward = 0
        self.episode_length = 0
        self.total_episodes = 0
        
    def _on_step(self):
        # Update episode stats
        if "rewards" in self.locals and len(self.locals["rewards"]) > 0:
            self.episode_reward += self.locals["rewards"][0]
            self.episode_length += 1
        
        # If episode is done, log it
        if "dones" in self.locals and len(self.locals["dones"]) > 0 and self.locals["dones"][0]:
            self.stats.log_episode(self.episode_reward, self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0
            self.total_episodes += 1
            
            # Save checkpoint periodically
            if self.total_episodes % 5 == 0:
                algo_name = self.model.__class__.__name__.lower()
                checkpoint_path = f"models/{algo_name}_donkey_checkpoint_{self.stats.timestamp}_{self.total_episodes}.zip"
                self.model.save(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        
        # Periodic logging if episode is long
        if self.episode_length > 0 and self.episode_length % 500 == 0:
            print(f"Long episode in progress: {self.episode_length} steps, current reward: {self.episode_reward:.2f}")
            
        return True

def plot_training_stats(csv_path=None):
    """
    Create visualizations from the training statistics CSV file
    """
    # Select the most recent CSV file if none provided
    if csv_path is None:
        log_files = [f for f in os.listdir("logs") if f.startswith("donkey_stats_") and f.endswith(".csv")]
        if not log_files:
            print("No training log files found!")
            return
        log_files.sort(reverse=True)  # Most recent first
        csv_path = os.path.join("logs", log_files[0])
    
    # Load the data
    try:
        data = pd.read_csv(csv_path)
        print(f"Loaded training data with {len(data)} entries from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Set up the style
    sns.set(style="darkgrid")
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Mean reward over time
    plt.subplot(3, 2, 1)
    plt.plot(data['timesteps'], data['mean_reward'], 'b-')
    plt.title('Mean Reward vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    
    # Plot 2: Mean episode length over time
    plt.subplot(3, 2, 2)
    plt.plot(data['timesteps'], data['mean_episode_length'], 'r-')
    plt.title('Mean Episode Length vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Length')
    plt.grid(True)
    
    # Plot 3: FPS over time
    plt.subplot(3, 2, 3)
    plt.plot(data['timesteps'], data['fps'], 'g-')
    plt.title('Training Speed (FPS) vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Frames Per Second')
    plt.grid(True)
    
    # Plot 4: Mean reward vs episode length (scatter)
    plt.subplot(3, 2, 4)
    plt.scatter(data['mean_episode_length'], data['mean_reward'], alpha=0.7)
    plt.title('Mean Reward vs Episode Length')
    plt.xlabel('Mean Episode Length')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    
    # Plot 5: Smoothed reward
    def smooth_curve(y, window=3):
        """Apply moving average smoothing to a curve."""
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    if len(data) > 2:
        plt.subplot(3, 2, 5)
        plt.plot(data['timesteps'], data['mean_reward'], 'b-', alpha=0.4, label='Raw')
        plt.plot(data['timesteps'], smooth_curve(data['mean_reward']), 'r-', 
                linewidth=2, label='Smoothed')
        plt.title('Smoothed Mean Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.legend()
        plt.grid(True)
    
    # Plot 6: Reward improvement rate (derivative)
    if len(data) > 2:
        plt.subplot(3, 2, 6)
        reward_changes = np.diff(data['mean_reward'])
        plt.bar(data['timesteps'][1:], reward_changes, alpha=0.7, width=data['timesteps'].iloc[1] * 0.8)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Reward Improvement Rate')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward Change')
        plt.grid(True)
    
    # Add some overall info as text
    if len(data) > 0:
        total_time = data['time_elapsed'].iloc[-1] / 60  # minutes
        final_reward = data['mean_reward'].iloc[-1]
        max_reward = data['mean_reward'].max()
        
        info_text = (
            f"Training Summary:\n"
            f"Total time: {total_time:.1f} minutes\n"
            f"Final reward: {final_reward:.1f}\n"
            f"Max reward: {max_reward:.1f}\n"
            f"Total timesteps: {data['timesteps'].iloc[-1]}"
        )
        
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12, 
                   bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
    
    # Save and show the figure
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust for the text at the bottom
    output_path = csv_path.replace('.csv', '_plots.png')
    plt.savefig(output_path)
    print(f"Visualizations saved to {output_path}")
    plt.show()

def parse_ppo_logs(log_text):
    """
    Parse PPO training logs from the copied text output
    """
    lines = log_text.strip().split('\n')
    timesteps = []
    rewards = []
    ep_lengths = []
    
    current_timestep = 0
    for line in lines:
        if "|" in line and "rollout" in line:
            # Start of a new log block
            continue
        elif "ep_len_mean" in line:
            value = float(line.split('|')[2].strip())
            ep_lengths.append(value)
        elif "ep_rew_mean" in line:
            value = float(line.split('|')[2].strip())
            rewards.append(value)
        elif "total_timesteps" in line:
            value = int(line.split('|')[2].strip())
            timesteps.append(value)
    
    # Create a CSV from the parsed data
    timestamp = int(time.time())
    csv_path = f"logs/parsed_ppo_stats_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timesteps', 'episodes', 'time_elapsed',
            'mean_reward', 'mean_episode_length', 'fps'
        ])
        
        for i in range(len(timesteps)):
            writer.writerow([
                timesteps[i],
                i+1,
                i*15,  # Placeholder for time elapsed
                rewards[i],
                ep_lengths[i],
                15  # Placeholder for fps
            ])
    
    print(f"Parsed PPO logs saved to {csv_path}")
    return csv_path

def train_donkey_car_with_tqc():
    """
    Main function to train the Donkey Car with TQC
    """
    try:
        # First, try to import TQC
        from sb3_contrib import TQC
        use_tqc = True
    except ImportError:
        print("Could not import TQC, falling back to SAC")
        from stable_baselines3 import SAC
        use_tqc = False
    
    # Create environment with custom reward and max_cte
    env_config = {
        "max_cte": 8.0,  # Larger cross-track error threshold
        "reward_fn": None  # Use default reward
    }
    
    env = gym.make("donkey-mountain-track-v0", conf=env_config)
    
    # Create stats tracker
    stats = TrainingStats(verbose=1, live_plot=False)
    
    # Create stats callback
    stats_callback = StatsCallback(stats_tracker=stats, check_freq=1, verbose=1)
    
    # Create model with adjusted parameters
    if use_tqc:
        model = TQC(
            "CnnPolicy", 
            env, 
            learning_rate=0.0003,
            buffer_size=50000,
            batch_size=256,
            ent_coef="auto_0.1",
            gamma=0.99,
            tau=0.02,
            train_freq=1,
            gradient_steps=1,
            top_quantiles_to_drop_per_net=2,
            verbose=1,
            tensorboard_log="./tqc_donkey_tensorboard/"
        )
        print("Using TQC algorithm")
    else:
        model = SAC(
            "CnnPolicy", 
            env, 
            learning_rate=0.0003,
            buffer_size=50000,
            batch_size=256,
            ent_coef="auto",
            gamma=0.99,
            tau=0.02,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log="./sac_donkey_tensorboard/"
        )
        print("Using SAC algorithm (TQC not available)")
    
    # Training settings
    total_timesteps = 100000  # Longer training for TQC/SAC
    
    # Train the model using built-in learn method with callback
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=stats_callback,
            log_interval=10,
            tb_log_name="TQC_donkey",
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
    
    # Save final model
    algorithm_name = "tqc" if use_tqc else "sac"
    model_path = f"models/{algorithm_name}_donkey_{stats.timestamp}_final.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Close environment
    env.close()
    
    # Plot final results
    plot_training_stats(stats.csv_path)
    
    return stats.csv_path

# Run training
if __name__ == "__main__":
    train_donkey_car_with_tqc()