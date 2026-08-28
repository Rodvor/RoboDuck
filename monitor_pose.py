import argparse
import math
import os
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

import mujoco
import numpy as np
from stable_baselines3 import PPO

from env.microduck_env import MicroDuckEnv


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained MicroDuck policy and report its pose."
    )
    parser.add_argument(
        "--model",
        default="microduck_stand_parallel",
        help="Path to the saved PPO model (default: microduck_stand_parallel)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between pose reports (default: 2.0)",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    env = MicroDuckEnv(render_mode="human")
    model = PPO.load(args.model, env=env)
    observation, _ = env.reset()

    neck_joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch"
    )
    head_joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch"
    )
    neck_qpos_address = env.model.jnt_qposadr[neck_joint_id]
    head_qpos_address = env.model.jnt_qposadr[head_joint_id]

    next_report = time.monotonic()
    episode = 1

    print("Monitoring MicroDuck. Close the MuJoCo window or press Ctrl+C to stop.")

    try:
        while env.viewer is None or env.viewer.is_running():
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            env.render()

            now = time.monotonic()
            if now >= next_report:
                body_height = float(env.data.qpos[2])
                head_height = float(env.data.xpos[env.head_body_id, 2])
                displacement = float(np.linalg.norm(
                    env.data.qpos[:2] - env.initial_xy
                ))
                horizontal_speed = float(np.linalg.norm(env.data.qvel[:2]))
                com_error = float(np.linalg.norm(env._get_com_support_error()))
                w, x, y, z = (float(value) for value in env.data.qpos[3:7])
                neck_pitch = float(env.data.qpos[neck_qpos_address])
                head_pitch = float(env.data.qpos[head_qpos_address])
                print(
                    f"episode={episode} body_height={body_height:.4f} m "
                    f"head_height={head_height:.4f} m "
                    f"displacement={displacement:.4f} m "
                    f"speed={horizontal_speed:.4f} m/s "
                    f"com_error={com_error:.4f} m "
                    f"quaternion=(w={w:+.5f}, x={x:+.5f}, "
                    f"y={y:+.5f}, z={z:+.5f}) "
                    f"neck_pitch={math.degrees(neck_pitch):+.1f} deg "
                    f"head_pitch={math.degrees(head_pitch):+.1f} deg",
                    flush=True,
                )
                next_report = now + args.interval

            if terminated or truncated:
                reason = "fallen" if terminated else "time limit"
                print(f"Episode {episode} ended: {reason}. Resetting.", flush=True)
                observation, _ = env.reset()
                episode += 1

            # Match the environment's simulated control timestep.
            time.sleep(env.model.opt.timestep * env.frame_skip)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
