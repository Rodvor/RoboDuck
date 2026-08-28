import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.microduck_walk_env import MicroDuckWalkEnv


def make_env():
    def _init():
        return Monitor(MicroDuckWalkEnv())

    return _init


def create_vector_env(number_of_envs):
    factories = [make_env() for _ in range(number_of_envs)]
    if number_of_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def checkpoint_exists(model_name):
    path = Path(model_name)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    return path.exists()


def main():
    available_cpus = os.cpu_count() or 2
    default_envs = max(1, min(8, available_cpus - 1))

    parser = argparse.ArgumentParser(
        description="Train the MicroDuck forward-walking policy."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Additional training timesteps (default: 5000000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=default_envs,
        help=f"Parallel environments (default on this machine: {default_envs})",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Rollout steps per environment (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="PPO minibatch size (default: 512)",
    )
    parser.add_argument(
        "--start-model",
        default="microduck_stand_parallel",
        help="Standing checkpoint used to initialize walking",
    )
    parser.add_argument(
        "--output",
        default="microduck_walk_parallel",
        help="Output checkpoint name (default: microduck_walk_parallel)",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="PyTorch learner threads (default: 1)",
    )
    args = parser.parse_args()

    if args.timesteps <= 0 or args.n_envs <= 0 or args.n_steps <= 0:
        parser.error("timesteps, n-envs, and n-steps must be greater than zero")

    rollout_size = args.n_envs * args.n_steps
    if args.batch_size <= 0 or rollout_size % args.batch_size != 0:
        parser.error("batch-size must be positive and divide n-envs * n-steps")

    torch.set_num_threads(args.torch_threads)
    env = create_vector_env(args.n_envs)

    try:
        if checkpoint_exists(args.start_model):
            print(f"Starting from standing policy: {args.start_model}")
            model = PPO.load(
                args.start_model,
                env=env,
                device="cpu",
                n_steps=args.n_steps,
                batch_size=args.batch_size,
            )
            # Restore exploration after the standing policy has converged.
            model.policy.log_std.data.fill_(-1.0)
        else:
            print("Standing checkpoint not found; starting walking from scratch.")
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs={"log_std_init": -1.0},
                learning_rate=3e-4,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                verbose=1,
                device="cpu",
            )

        print(
            f"Training with {args.n_envs} environments, "
            f"{args.n_steps} steps per environment, and "
            f"{rollout_size} samples per rollout."
        )
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
        )
        model.save(args.output)
        print(f"Walking model saved as {args.output}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
