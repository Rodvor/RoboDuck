import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.microduck_env import MicroDuckEnv


def make_env():
    def _init():
        return Monitor(MicroDuckEnv())

    return _init


def create_vector_env(number_of_envs):
    factories = [make_env() for _ in range(number_of_envs)]

    if number_of_envs == 1:
        return DummyVecEnv(factories)

    return SubprocVecEnv(factories, start_method="spawn")


def main():
    parser = argparse.ArgumentParser(
        description="Continue training an existing MicroDuck PPO checkpoint."
    )
    parser.add_argument(
        "--model",
        default="microduck_stand_parallel",
        help="Checkpoint to continue from (default: microduck_stand_parallel)",
    )
    parser.add_argument(
        "--output",
        default="microduck_stand_parallel_continued",
        help="Continued checkpoint name (default: microduck_stand_parallel_continued)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Additional training timesteps (default: 1000000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=8,
        help="Parallel MuJoCo environments (default: 8)",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="PyTorch CPU threads in the learner process (default: 1)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="Rollout steps collected by each worker (default: 1024)",
    )
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be greater than zero")
    if args.n_steps <= 0:
        parser.error("--n-steps must be greater than zero")

    torch.set_num_threads(args.torch_threads)
    rollout_size = args.n_steps * args.n_envs
    if rollout_size % 512 != 0:
        parser.error("--n-steps multiplied by --n-envs must be divisible by 512")
    env = create_vector_env(args.n_envs)

    try:
        checkpoint = PPO.load(args.model, device="cpu")
        checkpoint_size = checkpoint.observation_space.shape[0]
        environment_size = env.observation_space.shape[0]
        if checkpoint_size != environment_size:
            raise ValueError(
                f"Checkpoint has {checkpoint_size} observations but the successful "
                f"standing environment uses {environment_size}. Train a fresh model "
                f"with 'python3 train_stand.py' before continuing it."
            )

        model = PPO.load(
            args.model,
            env=env,
            device="cpu",
            n_steps=args.n_steps,
            batch_size=512,
        )

        print(
            f"Continuing {args.model} for {args.timesteps:,} additional timesteps "
            f"with {args.n_envs} parallel environments and "
            f"{rollout_size} samples per PPO rollout using "
            f"{args.torch_threads} PyTorch thread(s)..."
        )

        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
        )

        model.save(args.output)
        print(f"Continued model saved as {args.output}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
