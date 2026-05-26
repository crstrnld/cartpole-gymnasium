import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# pilih game 
env = gym.make("CartPole-v1")

# pilih agent sesuai import
model = PPO("MlpPolicy", env, verbose=1)

# latihan (semakin banyak latihan semakin jagoo)
timesteps = 15000
reward_history = []

# catatan dari latihan
def reward_callback(_locals, _globals):
    if "infos" in _locals and len(_locals["infos"]) > 0:
        rewards = [info.get("episode", {}).get("r", None) for info in _locals["infos"]]
        rewards = [r for r in rewards if r is not None]
        reward_history.extend(rewards)
    return True

model.learn(total_timesteps=timesteps, callback=reward_callback)

# visual jika sulit memahami table dari output
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Performa Agent PPO di CartPole")
plt.show()

# hasil latihan/ujian bagi agen (episode=banyak ujian yang akan dijalani)
def evaluate_agent(env, model, episodes=100):
    rewards = []
    for _ in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

mean_reward, std_reward = evaluate_agent(env, model)
print(f"Rata-rata reward: {mean_reward:.2f} ± {std_reward:.2f}")
