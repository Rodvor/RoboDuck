import argparse
import os
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

from stable_baselines3 import PPO

from env.microduck_walk_env import MicroDuckWalkEnv


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained MicroDuck walking policy."
    )
    parser.add_argument(
        "--model",
        default="microduck_walk_parallel",
        help="Walking checkpoint (default: microduck_walk_parallel)",
    )
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = MicroDuckWalkEnv(render_mode="human")
    observation, _ = env.reset()

    episode = 1
    episode_reward = 0.0
    episode_start_x = float(env.data.qpos[0])
    control_timestep = env.frame_skip * env.model.opt.timestep

    print(f"Watching {args.model}. Close the window or press Ctrl+C to stop.")

    try:
        while env.viewer is None or env.viewer.is_running():
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(control_timestep)

            if terminated or truncated:
                distance = float(env.data.qpos[0]) - episode_start_x
                seconds = env.step_count * control_timestep
                speed = distance / seconds if seconds > 0.0 else 0.0
                reason = "fallen" if terminated else "time limit"
                print(
                    f"Episode {episode}: {reason} | distance={distance:.2f} m | "
                    f"speed={speed:.3f} m/s | reward={episode_reward:.2f}",
                    flush=True,
                )
                observation, _ = env.reset()
                episode += 1
                episode_reward = 0.0
                episode_start_x = float(env.data.qpos[0])
    except KeyboardInterrupt:
        print("\nPlayback stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
