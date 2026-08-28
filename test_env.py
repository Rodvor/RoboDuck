from env.duck_env import DuckEnv
import numpy as np


env = DuckEnv()

observation, info = env.reset()

print("================================")
print("Duck RL environment test")
print("================================")

print("Observation shape:")
print(observation.shape)

print()

print("Action space:")
print(env.action_space)

print()

print("Observation space:")
print(env.observation_space)

print()

for i in range(100):

    action = np.zeros(14)

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    print(
        f"step={i:3d} "
        f"reward={reward:.3f} "
        f"height={env.data.qpos[2]:.3f}"
    )

    if terminated or truncated:

        print("Duck fell!")

        break

env.close()