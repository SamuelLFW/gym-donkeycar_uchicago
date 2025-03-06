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
from stable_baselines3 import PPO  # Import PPO directly

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

# ... [rest of your code remains the same until train_donkey_car() function] ...

def train_donkey_car():
    """
    Main function to train the Donkey Car with PPO
    """
    # Create environment with custom reward and max_cte
    env_config = {
        "max_cte": 6.0,  # Larger cross-track error threshold
        "reward_fn": None  # Use default reward
    }
    
    env = gym.make("donkey-mountain-track-v0", conf=env_config)
    
    # Create stats tracker
    stats = TrainingStats(verbose=1, live_plot=False)
    
    # Create PPO model with adjusted parameters
    model = PPO(
        "CnnPolicy", 
        env, 
        n_steps=256,
        learning_rate=0.0005,  # Higher learning rate
        gamma=0.95,            # Slightly lower discount factor
        ent_coef=0.05,         # Higher entropy for more exploration
        verbose=1
    )
    
    # Training settings
    total_timesteps = 50000  # Increase training time
    
    # Training loop
    episode = 0
    steps_since_log = 0
    log_frequency = 500  # Log every 500 steps regardless of episode completion
    
    while stats.timesteps < total_timesteps:
        episode += 1
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"Starting episode {episode}")
        
        # Run episode
        while not done and stats.timesteps < total_timesteps:
            # Get action with exploration noise in early training
            action, _ = model.predict(obs, deterministic=False)
            
            # Add exploration noise
            if stats.episodes < 10:  # First few episodes
                action += np.random.normal(0, 0.4, size=action.shape)  # Add significant noise
                action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            obs, reward, done, _ = env.step(action)
            
            # Update episode stats
            episode_reward += reward
            episode_length += 1
            steps_since_log += 1
            
            # Break if episode gets too long
            if episode_length > 500:
                print("Episode truncated at 500 steps")
                done = True
            
            # Periodic logging even if episode doesn't end
            if steps_since_log >= log_frequency:
                print(f"Periodic log at {steps_since_log} steps")
                stats.log_episode(episode_reward, episode_length)
                steps_since_log = 0
        
        # Episode completed
        if steps_since_log > 0:  # Only log if we haven't just logged
            stats.log_episode(episode_reward, episode_length)
            steps_since_log = 0
        
        # Print progress
        print(f"Completed episode {episode} with reward {episode_reward:.2f}")
        
        # Optionally save checkpoint every 5 episodes
        if episode % 5 == 0:
            checkpoint_path = f"models/ppo_donkey_checkpoint_{stats.timestamp}_{episode}.zip"
            model.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    model_path = f"models/ppo_donkey_{stats.timestamp}_final.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Close environment
    env.close()
    
    # Plot final results
    plot_training_stats(stats.csv_path)
    
    return stats.csv_path

# Run training
if __name__ == "__main__":
    train_donkey_car()