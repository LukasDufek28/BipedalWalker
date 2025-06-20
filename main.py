# %% [markdown]
# **Importy**

# %%
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os

# %%
# Parameters
benchmark = "BipedalWalker-v3"
model_ = TD3
max_stepov_na_epizodu = 1000
num_envs = 6

# Custom environment wrapper (optional reward shaping)
class CustomBipedalWalker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

# Function to create monitored env
def make_env():
    env = gym.make(benchmark, max_episode_steps=max_stepov_na_epizodu)
    env = CustomBipedalWalker(env)
    env = Monitor(env)  # Required for episode reward tracking
    return env

# Vectorized environments with monitoring
vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
vec_env = VecMonitor(vec_env)

# Action noise for TD3
action_noise = NormalActionNoise(
    mean=np.zeros(vec_env.action_space.shape),
    sigma=0.1 * np.ones(vec_env.action_space.shape)
)

# Custom callback for logging average reward
class AvgRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    reward = info["episode"]["r"]
                    self.episode_rewards.append(reward)
                    if len(self.episode_rewards) >= 100:
                        avg_reward = sum(self.episode_rewards[-100:]) / 100
                        self.logger.record("custom/avg_reward_100ep", avg_reward)
        return True

# Eval environment
eval_env = Monitor(gym.make(benchmark, max_episode_steps=max_stepov_na_epizodu))

def make_eval_env():
    env = gym.make(benchmark, max_episode_steps=max_stepov_na_epizodu)
    env = CustomBipedalWalker(env)
    env = Monitor(env)
    return env

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecMonitor(eval_env)


# Eval callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./log/",
    eval_freq=5000,
    deterministic=True,
    render=False
)



# %%
# Define model
model = model_(
    'MlpPolicy',
    vec_env,
    verbose=1,
    device="cuda",
    action_noise=action_noise,
    tensorboard_log="./log/" + model_.__name__ + "_" + benchmark,
    batch_size=256,
    learning_rate=0.0003,
    buffer_size = 1_000_000,
    gamma=0.99,
    learning_starts=10000,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
)

# %%
# Train with callbacks
model.learn(
    total_timesteps=2_000_000,
    callback=[eval_callback, AvgRewardCallback()],
    progress_bar=True,
)

# %%
model.save(model_.__name__ + "_" + benchmark)

# %%
# Parameters
benchmark = "BipedalWalkerHardcore-v3"
model_ = TD3
max_stepov_na_epizodu = 1000
num_envs = 6

# Custom environment wrapper (optional reward shaping)
class CustomBipedalWalker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

# Function to create monitored env
def make_env():
    env = gym.make(benchmark, max_episode_steps=max_stepov_na_epizodu)
    env = CustomBipedalWalker(env)
    env = Monitor(env)  # Required for episode reward tracking
    return env

# Vectorized environments with monitoring
vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
vec_env = VecMonitor(vec_env)

# Action noise for TD3
action_noise = NormalActionNoise(
    mean=np.zeros(vec_env.action_space.shape),
    sigma=0.1 * np.ones(vec_env.action_space.shape)
)

# Custom callback for logging average reward
class AvgRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    reward = info["episode"]["r"]
                    self.episode_rewards.append(reward)
                    if len(self.episode_rewards) >= 100:
                        avg_reward = sum(self.episode_rewards[-100:]) / 100
                        self.logger.record("custom/avg_reward_100ep", avg_reward)
        return True

# Eval environment
eval_env = Monitor(gym.make(benchmark, max_episode_steps=max_stepov_na_epizodu))

def make_eval_env():
    env = gym.make(benchmark, max_episode_steps=max_stepov_na_epizodu)
    env = CustomBipedalWalker(env)
    env = Monitor(env)
    return env

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecMonitor(eval_env)


# Eval callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./log/",
    eval_freq=5000,
    deterministic=True,
    render=False
)



model = TD3.load("TD3_BipedalWalker-v3", env=vec_env)

# Train with callbacks
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, AvgRewardCallback()],
    progress_bar=True,
)

# %%
model.save(model_.__name__ + "_" + benchmark + "_hardcore")

# %%
# model = model_.load(model_.__name__ + "_" + benchmark + "_hardcore") # Načítanie modelu
model = model_.load("TD3_BipedalWalkerHardcore-v3_hardcore") # Načítanie modelu
env = gym.make(benchmark, render_mode="human", max_episode_steps=max_stepov_na_epizodu)

# Set seed for reproducibility
env.reset(seed=42)

# Spustenie evaluacie
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(mean_reward, std_reward)