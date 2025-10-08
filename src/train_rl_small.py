from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from affordance_rl_env_small import AffordanceRLSmallEnv
import signal
import os
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from threading import Lock

def generate_reward_plot(log_file="training_log.csv", save_path="reward_curve.png"):
    """Generate reward curve plot from training log"""
    try:
        if not os.path.exists(log_file):
            print(f"⚠️ Log file {log_file} not found, skipping plot generation")
            return
        
        # Read CSV data
        df = pd.read_csv(log_file)
        
        if len(df) < 2:
            print("⚠️ Not enough data for plotting, skipping")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Reward per Episode
        ax1.plot(df['episode_id'], df['total_reward'], 'b-', linewidth=2, alpha=0.7)
        ax1.set_title('Episode Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success Rate (rolling average)
        ax2.plot(df['episode_id'], df['success_rate'], 'g-', linewidth=2, alpha=0.7)
        ax2.set_title('Success Rate (Rolling Avg)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Affordance Probability
        ax3.plot(df['episode_id'], df['avg_afford_prob'], 'r-', linewidth=2, alpha=0.7)
        ax3.set_title('Average Affordance Probability')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Avg Affordance')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Minimum Distance
        ax4.plot(df['episode_id'], df['min_distance'], 'm-', linewidth=2, alpha=0.7)
        ax4.set_title('Minimum Distance to Objects')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Min Distance (m)')
        ax4.grid(True, alpha=0.3)
        
        # Add training statistics as text
        latest_data = df.iloc[-1]
        stats_text = f"""Latest Episode: {int(latest_data['episode_id'])}
Total Steps: {int(latest_data['total_steps'])}
Success Rate: {latest_data['success_rate']:.3f}
Avg Reward: {df['total_reward'].tail(10).mean():.3f}
Max Reward: {df['total_reward'].max():.3f}"""
        
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Reward curve plot saved to {save_path}")
        
    except Exception as e:
        print(f"⚠️ Error generating plot: {e}")

class TrainingLogger:
    def __init__(self, episode_log_file="training_log_episode.csv", step_log_file="training_log_step.csv"):
        self.episode_log_file = episode_log_file
        self.step_log_file = step_log_file
        self.episode_rewards = []
        self.success_window = []
        self.lock = Lock()  # Thread lock for safe concurrent access
        
        # Initialize episode CSV file with headers
        if not os.path.exists(episode_log_file):
            with self.lock:
                with open(episode_log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'episode_id', 'total_steps', 'total_reward', 'success_rate', 
                        'avg_afford_prob', 'min_distance', 'timestamp'
                    ])
        
        # Initialize step CSV file with headers
        if not os.path.exists(step_log_file):
            with self.lock:
                with open(step_log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'episode_id', 'step_in_episode', 'total_steps', 'r1', 'r2', 'r3', 'penalty', 'penalty_scale', 'shaping', 'total_reward', 'success', 'hard_penalty', 'soft_penalty', 'reward_scale', 'r1_weight', 'timestamp'
                    ])
    
    def log_step(self, episode_id, step_in_episode, total_steps, r1, r2, r3, penalty, penalty_scale, shaping, total_reward, success, hard_penalty=0.0, soft_penalty=0.0, reward_scale=1.0, r1_weight=1.0):
        """Log detailed step information"""
        with self.lock:
            with open(self.step_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_id, step_in_episode, total_steps, r1, r2, r3, penalty, penalty_scale, shaping, total_reward, int(success), hard_penalty, soft_penalty, reward_scale, r1_weight, datetime.now().isoformat()
                ])
    
    def log_episode(self, episode_id, total_steps, total_reward, success, 
                   avg_afford_prob, min_distance):
        # Update success window
        self.success_window.append(int(success))
        if len(self.success_window) > 50:  # Keep last 50 episodes
            self.success_window.pop(0)
        
        success_rate = np.mean(self.success_window) if self.success_window else 0.0
        
        # Log to CSV
        with self.lock:
            with open(self.episode_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_id, total_steps, total_reward, success_rate,
                    avg_afford_prob, min_distance, datetime.now().isoformat()
                ])
        
        print(f"📊 Logged episode {episode_id}: reward={total_reward:.3f}, success_rate={success_rate:.3f}")

class CheckpointCallback(BaseCallback):
    def __init__(self, save_path, logger, verbose=0, total_timesteps=20000, save_freq=2000):
        super().__init__(verbose)
        self.save_path = save_path
        self.training_logger = logger  # Use different name to avoid conflict
        self.best_reward = -float('inf')
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.episode_count = 0
        self.episode_reward = 0
        self.episode_afford_probs = []
        self.episode_min_distances = []
        self.last_episode_count = 0  # Track episodes via ep_info_buffer

    def _on_step(self) -> bool:
        # Accumulate episode data
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_reward = ep_info['r']
            # Note: We can't easily track afford_prob and min_distance here
            # They would need to be passed from the environment
        
        current_steps = self.model.num_timesteps
        remaining = max(0, self.total_timesteps - current_steps)
        progress = (current_steps / self.total_timesteps) * 100
        print(f"📊 Training Progress: {current_steps}/{self.total_timesteps} steps ({progress:.1f}%) - {remaining} steps remaining")
        
        # Check for episode end
        if len(self.model.ep_info_buffer) > self.last_episode_count:
            self._log_episode_end()
            self.last_episode_count = len(self.model.ep_info_buffer)
        
        # Periodic saving
        if current_steps % self.save_freq == 0 and current_steps > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            periodic_save_path = f"models/ppo_step_{current_steps}_{timestamp}.zip"
            self.model.save(periodic_save_path)
            # Generate reward curve plot with matching timestamp
            plot_path = f"models/ppo_step_{current_steps}_{timestamp}_reward_curve.png"
            generate_reward_plot(self.training_logger.episode_log_file, plot_path)
            print(f"💾 Periodic save: {periodic_save_path}")
            print(f"📊 Reward plot: {plot_path}")
        
        # Best reward saving
        if len(self.model.ep_info_buffer) > 0:
            current_reward = self.model.ep_info_buffer[-1]['r']
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                best_save_path = f"models/ppo_best_{timestamp}.zip"
                self.model.save(best_save_path)
                # Generate reward curve plot with matching timestamp
                plot_path = f"models/ppo_best_{timestamp}_reward_curve.png"
                generate_reward_plot(self.training_logger.episode_log_file, plot_path)
                if self.verbose > 0:
                    print(f"🏆 New best reward: {current_reward:.2f}, saved to {best_save_path}")
                    print(f"📊 Reward plot: {plot_path}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout (episode) - keeping for compatibility but not using for logging"""
        pass

    def _log_episode_end(self):
        """Log episode end when detected via ep_info_buffer"""
        # Get episode summary BEFORE it gets reset
        try:
            underlying_env = self.model.env.envs[0].env
            if hasattr(underlying_env, 'get_episode_summary'):
                episode_data = underlying_env.get_episode_summary()
                avg_afford_prob = episode_data['avg_afford_prob']
                min_distance = episode_data['min_distance'] 
                total_reward = episode_data['total_reward']
                success = episode_data['success']
            else:
                raise AttributeError("get_episode_summary not found")
        except Exception as e:
            print(f"⚠️ Could not access episode summary: {e}")
            # Fallback: use data from ep_info_buffer
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                total_reward = ep_info['r']
                success = total_reward > 0.5  # Better heuristic for success
                avg_afford_prob = 0.5
                min_distance = 0.1
            else:
                total_reward = 0.0
                success = False
                avg_afford_prob = 0.0
                min_distance = 0.0
        
        # Update success window for rolling success rate
        self.training_logger.success_window.append(int(success))
        if len(self.training_logger.success_window) > 50:
            self.training_logger.success_window.pop(0)
        
        success_rate = np.mean(self.training_logger.success_window) if self.training_logger.success_window else 0.0
        
        # Log episode data
        self.episode_count += 1
        with self.training_logger.lock:
            with open(self.training_logger.episode_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_count, self.model.num_timesteps, total_reward, success_rate,
                    avg_afford_prob, min_distance, datetime.now().isoformat()
                ])

def save_model(signum, frame):
    global model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f"models/ppo_manual_{timestamp}.zip"
    model.save(save_path)
    # Generate reward curve plot with matching timestamp
    plot_path = f"models/ppo_manual_{timestamp}_reward_curve.png"
    generate_reward_plot("training_log_episode.csv", plot_path)
    print(f"✅ Manual save: {save_path} on signal {signum}")
    print(f"📊 Reward plot: {plot_path}")

def main():
    # Clean up any existing training log to start fresh BEFORE creating logger
    if os.path.exists("training_log_episode.csv"):
        os.remove("training_log_episode.csv")
        print("🧹 Removed existing episode training log for fresh start")
    if os.path.exists("training_log_step.csv"):
        os.remove("training_log_step.csv")
        print("🧹 Removed existing step training log for fresh start")
    
    # Initialize training logger first (will create files with headers)
    logger = TrainingLogger(episode_log_file="training_log_episode.csv", step_log_file="training_log_step.csv")
    
    env = AffordanceRLSmallEnv(
    model_path="models/affordance_model_best (copy).pth",
    k_candidates=10,
    gui=False,
    logger=logger
)

    global model
    # Always start fresh training (no loading existing model)
    print("🆕 Starting fresh training...")
    
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        verbose=2,
        device="cpu",
    )
    
    # Set total training timesteps
    total_timesteps = 20000  # Total steps for fresh training
    checkpoint_callback = CheckpointCallback(
        save_path="models/ppo_checkpoint.zip", 
        logger=logger,
        verbose=1, 
        total_timesteps=total_timesteps,
        save_freq=2000
    )
    
    # Set up signal handler for manual save
    signal.signal(signal.SIGUSR1, save_model)
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("⏹️ Training interrupted! Saving current model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        interrupted_save_path = f"models/ppo_interrupted_{timestamp}.zip"
        model.save(interrupted_save_path)
        # Generate reward curve plot with matching timestamp
        plot_path = f"models/ppo_interrupted_{timestamp}_reward_curve.png"
        generate_reward_plot("training_log_episode.csv", plot_path)
        print(f"✅ Model saved to {interrupted_save_path}")
        print(f"📊 Reward plot: {plot_path}")
        raise
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_save_path = f"models/ppo_final_{timestamp}.zip"
    model.save(final_save_path)
    # Generate reward curve plot with matching timestamp
    plot_path = f"models/ppo_final_{timestamp}_reward_curve.png"
    generate_reward_plot("training_log_episode.csv", plot_path)
    print(f"✅ Fresh training complete and saved to {final_save_path}")
    print(f"📊 Final reward plot: {plot_path}")

if __name__ == "__main__":
    main()
