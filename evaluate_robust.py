import argparse

import numpy as np
from stable_baselines3 import PPO

from env.microduck_robust_env import MicroDuckRobustEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate robust locomotion headlessly.")
    parser.add_argument("--model", default="microduck_robust_parallel")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--speed", type=float, default=0.15)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = MicroDuckRobustEnv(evaluation=True)
    distances, speeds, lengths, tilts = [], [], [], []
    dt = env.frame_skip * env.model.opt.timestep

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=10_000 + episode)
        env.target_speed = args.speed
        observation = env._get_observation()
        start_x = float(env.data.qpos[0])
        minimum_up = 1.0
        for step in range(env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            quat = env.data.qpos[3:7]
            minimum_up = min(minimum_up, 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
            if terminated or truncated:
                break
        distance = float(env.data.qpos[0]) - start_x
        seconds = (step + 1) * dt
        distances.append(distance)
        speeds.append(distance / seconds)
        lengths.append(step + 1)
        tilts.append(math.degrees(math.acos(np.clip(minimum_up, -1.0, 1.0))))

    env.close()
    print(f"episodes:       {args.episodes}")
    print(f"success rate:   {100 * np.mean(np.asarray(lengths) == 1000):.1f}%")
    print(f"mean distance:  {np.mean(distances):.3f} m")
    print(f"mean speed:     {np.mean(speeds):.3f} m/s")
    print(f"speed error:    {np.mean(np.abs(np.asarray(speeds) - args.speed)):.3f} m/s")
    print(f"mean max tilt:  {np.mean(tilts):.1f} degrees")


if __name__ == "__main__":
    import math
    main()

