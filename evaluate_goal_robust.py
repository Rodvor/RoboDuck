import argparse
import math
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

import numpy as np
from stable_baselines3 import PPO

from env.microduck_goal_robust_env import MicroDuckGoalRobustEnv


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate goal navigation and fall recovery headlessly."
    )
    parser.add_argument("--model", default="microduck_goal_robust")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--speed", type=float, default=0.15)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be greater than zero")
    if not 0.0 < args.speed <= 0.20:
        parser.error("speed must be greater than zero and at most 0.20 m/s")

    model = PPO.load(args.model, device="cpu")
    env = MicroDuckGoalRobustEnv(evaluation=True)
    goals_per_episode = []
    final_distances = []
    lengths = []
    tilts = []
    recovery_attempts = 0
    successful_recoveries = 0

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=20_000 + episode)
        env.target_speed = args.speed
        observation = env._get_observation()
        minimum_up = 1.0
        recovery_active = False

        for step in range(env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            quaternion = env.data.qpos[3:7]
            up_z = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
            minimum_up = min(minimum_up, up_z)

            if info["recovery_steps"] > 0 and not recovery_active:
                recovery_attempts += 1
                recovery_active = True
            elif info["recovery_steps"] == 0 and recovery_active:
                successful_recoveries += 1
                recovery_active = False

            if terminated or truncated:
                break

        goals_per_episode.append(info["goals_reached"])
        final_distances.append(info["goal_distance"])
        lengths.append(step + 1)
        tilts.append(math.degrees(math.acos(np.clip(minimum_up, -1.0, 1.0))))

    env.close()
    goals = np.asarray(goals_per_episode)
    print(f"episodes:          {args.episodes}")
    print(f"goal success:      {100.0 * np.mean(goals > 0):.1f}%")
    print(f"mean goals:        {np.mean(goals):.2f} per episode")
    print(f"mean final error:  {np.mean(final_distances):.3f} m")
    print(f"full episode rate: {100.0 * np.mean(np.asarray(lengths) == 1000):.1f}%")
    print(f"mean max tilt:     {np.mean(tilts):.1f} degrees")
    if recovery_attempts:
        print(
            f"recovery success:  "
            f"{100.0 * successful_recoveries / recovery_attempts:.1f}% "
            f"({successful_recoveries}/{recovery_attempts})"
        )
    else:
        print("recovery success:  no recovery events observed")


if __name__ == "__main__":
    main()
