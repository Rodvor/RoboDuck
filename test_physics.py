import time
import numpy as np
from env.microduck_env import MicroDuckEnv

env = MicroDuckEnv(render_mode="human")

obs, info = env.reset()

print("Starting physics test...")
print("Initial height:", env.data.qpos[2])

for i in range(500):

    # ZERO MOTOR COMMAND
    action = np.zeros(14, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    if i % 20 == 0:
        print(
            f"step={i:3d} "
            f"height={env.data.qpos[2]:.4f} "
            f"vel_z={env.data.qvel[2]:.4f}"
        )

    time.sleep(0.01)

    if terminated or truncated:
        print("DUCK FELL")

        print("height:", env.data.qpos[2])
        print("quat:", env.data.qpos[3:7])
        print("quat[0]^2:", env.data.qpos[3] ** 2)

        break
env.close()