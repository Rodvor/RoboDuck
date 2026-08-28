import time

from env.microduck_env import MicroDuckEnv
from stable_baselines3 import PPO


# Load environment
env = MicroDuckEnv(render_mode="human")

# Load trained model
model = PPO.load(
    "microduck_stand",
    env=env
)

print("Watching trained MicroDuck...")
print("Close the MuJoCo window to stop.")

obs, info = env.reset()

while True:

    action, _states = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.002)

    if terminated or truncated:

        print(
            f"Episode ended | reward={reward:.2f}"
        )

        obs, info = env.reset()