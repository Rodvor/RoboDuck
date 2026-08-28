from env.microduck_env import MicroDuckEnv

import numpy as np


env = MicroDuckEnv()

observation, info = env.reset()

print("======================================")
print("       MICRODUCK RL ENVIRONMENT")
print("======================================")

print()

print("Observation shape:")
print(observation.shape)

print()

print("Action space:")
print(env.action_space)

print()

print("Observation space:")
print(env.observation_space)

print()

print("Running 500 simulation steps...")
print()


for i in range(500):

    # Zero action = neutral servo targets
    action = np.zeros(
        14,
        dtype=np.float32
    )

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    if i % 50 == 0:

        print(
            f"step={i:3d} "
            f"height={env.data.qpos[2]:.3f} "
            f"reward={reward:.3f}"
        )

    if terminated:

        print()
        print("Duck fell!")
        break

    if truncated:

        print()
        print("Episode finished.")
        break


env.close()

print()
print("Environment test finished.")
