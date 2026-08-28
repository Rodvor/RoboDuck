import argparse
import os
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

from stable_baselines3 import PPO

from env.microduck_robust_env import MicroDuckRobustEnv


def main():
    parser = argparse.ArgumentParser(description="Watch robust commanded walking.")
    parser.add_argument("--model", default="microduck_robust_parallel")
    parser.add_argument("--speed", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.0 <= args.speed <= 0.20:
        parser.error("--speed must be between 0.0 and 0.20 m/s")

    model = PPO.load(args.model, device="cpu")
    env = MicroDuckRobustEnv(render_mode="human", evaluation=True)
    observation, _ = env.reset()
    env.target_speed = args.speed
    observation = env._get_observation()
    start_x = float(env.data.qpos[0])
    reward_total = 0.0
    control_timestep = env.frame_skip * env.model.opt.timestep

    print(
        f"Commanded speed: {args.speed:.2f} m/s. "
        "Close the window or press Ctrl+C to stop."
    )
    try:
        while env.viewer is None or env.viewer.is_running():
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward_total += reward
            env.render()
            time.sleep(control_timestep)
            if terminated or truncated:
                distance = float(env.data.qpos[0]) - start_x
                seconds = env.step_count * control_timestep
                print(
                    f"distance={distance:.2f} m | speed={distance / seconds:.3f} m/s "
                    f"| reward={reward_total:.1f}",
                    flush=True,
                )
                observation, _ = env.reset()
                env.target_speed = args.speed
                observation = env._get_observation()
                start_x = float(env.data.qpos[0])
                reward_total = 0.0
    except KeyboardInterrupt:
        print("\nPlayback stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

