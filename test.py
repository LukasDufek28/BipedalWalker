import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy

# Parameters
benchmark = "BipedalWalkerHardcore-v3"
model_path = "TD3_BipedalWalkerHardcore-v3_hardcore"
max_steps = 1000
seed = 42

# Load model
model = TD3.load(model_path)

# Create environment with seed
env = gym.make(benchmark, render_mode="human", max_episode_steps=max_steps)
env.reset(seed=seed)

# Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")