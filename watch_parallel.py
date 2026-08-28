import time
import numpy as np

from stable_baselines3 import PPO

from env.microduck_env import MicroDuckEnv


# Load trained model
model = PPO.load(
    "microduck_stand_parallel"
)

# Create ONE visual environment
env = MicroDuckEnv(
    render_mode="human"
)

obs, info = env.reset()

print("Watching trained MicroDuck...")
print("Close the MuJoCo window to stop.")

while True:

    # Ask the trained policy for an action
    action, _ = model.predict(
        obs,
        deterministic=True
    )

    # Run simulation
    obs, reward, terminated, truncated, info = env.step(
        action
    )

    # Display
    env.render()

    time.sleep(0.01)

    # Restart when duck falls
    if terminated or truncated:

        print(
            f"Episode ended | reward={reward:.2f}"
        )

        obs, info = env.reset()


env.close()