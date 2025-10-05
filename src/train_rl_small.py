from stable_baselines3 import PPO
from affordance_rl_env_small import AffordanceRLSmallEnv

def main():
    env = AffordanceRLSmallEnv(
    model_path="/media/irisliu/Elements/UbuntuWorkspace/Projects/AffordancePicking/models/affordance_model_best (copy).pth",
    k_candidates=5,
    gui=True
)

    model = PPO("MlpPolicy", env, verbose=1, gamma=0.9, learning_rate=3e-4)
    model.learn(total_timesteps=200)
    model.save("models/ppo_small_rl.zip")
    print("✅ RL训练完成并保存")

if __name__ == "__main__":
    main()
