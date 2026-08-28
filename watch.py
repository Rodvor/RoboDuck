import time

from stable_baselines3 import PPO

from env.duck_env import DuckEnv


env = DuckEnv(
    render_mode="human"
)

model = PPO.load(
    "duck_standing"
)

observation, info = env.reset()

while True:

    action, _states = model.predict(
        observation,
        deterministic=True
    )

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    env.render()

    time.sleep(0.002)

    if terminated or truncated:

        print(
            "Episode finished. "
            "Reward:",
            reward
        )

        observation, info = env.reset()