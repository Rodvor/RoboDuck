import argparse
import os
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

from stable_baselines3 import PPO

from env.microduck_goal_robust_env import MicroDuckGoalRobustEnv


def main():
    parser = argparse.ArgumentParser(
        description="Watch robust MicroDuck navigate between random waypoints."
    )
    parser.add_argument("--model", default="microduck_goal_robust")
    parser.add_argument("--speed", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.0 < args.speed <= 0.20:
        parser.error("speed must be greater than zero and at most 0.20 m/s")

    model = PPO.load(args.model, device="cpu")
    env = MicroDuckGoalRobustEnv(render_mode="human", evaluation=True)
    observation, info = env.reset()
    env.target_speed = args.speed
    observation = env._get_observation()
    control_timestep = env.frame_skip * env.model.opt.timestep

    print(
        f"Navigating at {args.speed:.2f} m/s toward random waypoints. "
        "Close the MuJoCo window or press Ctrl+C to stop."
    )
    print(f"Initial goal is {info['goal_distance']:.2f} m away.")
    try:
        while env.viewer is None or env.viewer.is_running():
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(control_timestep)

            if info["reached_goal"]:
                print(
                    f"Reached goal {info['goals_reached']}; next goal is "
                    f"{info['goal_distance']:.2f} m away.",
                    flush=True,
                )
            if terminated or truncated:
                reason = "recovery failed" if terminated else "time limit"
                print(
                    f"Episode ended: {reason}; reached "
                    f"{info['goals_reached']} goals.",
                    flush=True,
                )
                observation, info = env.reset()
                env.target_speed = args.speed
                observation = env._get_observation()
    except KeyboardInterrupt:
        print("\nPlayback stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
