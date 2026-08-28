import argparse
import os
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

from env.microduck_env import MicroDuckEnv
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained MicroDuck standing policy."
    )
    parser.add_argument(
        "--model",
        default="microduck_stand_parallel",
        help="Path to the saved PPO model (default: microduck_stand_parallel)",
    )
    args = parser.parse_args()

    env = MicroDuckEnv(render_mode="human")
    model = PPO.load(args.model, env=env, device="cpu")
    observation, _ = env.reset()

    episode = 1
    episode_steps = 0
    episode_reward = 0.0
    control_timestep = env.model.opt.timestep * env.frame_skip

    print(f"Watching {args.model}...")
    print("Close the MuJoCo window or press Ctrl+C to stop.")

    try:
        while env.viewer is None or env.viewer.is_running():
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1
            episode_reward += reward

            env.render()
            time.sleep(control_timestep)

            if terminated or truncated:
                reason = "fallen" if terminated else "time limit"
                print(
                    f"Episode {episode} ended: {reason} | "
                    f"steps={episode_steps} | reward={episode_reward:.2f}",
                    flush=True,
                )
                observation, _ = env.reset()
                episode += 1
                episode_steps = 0
                episode_reward = 0.0
    except KeyboardInterrupt:
        print("\nPlayback stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
