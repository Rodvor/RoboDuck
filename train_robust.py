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

from env.microduck_robust_env import MicroDuckRobustEnv


def make_env(curriculum_horizon):
    def _init():
        return Monitor(MicroDuckRobustEnv(curriculum_horizon=curriculum_horizon))
    return _init


def create_vector_env(number_of_envs, curriculum_horizon):
    factories = [make_env(curriculum_horizon) for _ in range(number_of_envs)]
    if number_of_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def transfer_policy(source_name, destination):
    """Transfer a 38D walking policy into the new 39D commanded policy."""
    source = PPO.load(source_name, device="cpu")
    source_state = source.policy.state_dict()
    destination_state = destination.policy.state_dict()
    copied = 0

    with torch.no_grad():
        for name, target in destination_state.items():
            original = source_state.get(name)
            if original is None:
                continue
            if original.shape == target.shape:
                target.copy_(original)
                copied += 1
            elif (
                original.ndim == 2
                and target.ndim == 2
                and original.shape[0] == target.shape[0]
                and original.shape[1] + 1 == target.shape[1]
            ):
                target.zero_()
                target[:, :original.shape[1]].copy_(original)
                copied += 1

    destination.policy.load_state_dict(destination_state)
    destination.policy.log_std.data.fill_(-1.0)
    print(f"Transferred {copied} policy tensors from {source_name}.")


def main():
    parser = argparse.ArgumentParser(
        description="Train commanded, push-resistant MicroDuck locomotion."
    )
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--n-envs", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--start-model", default="microduck_walk_parallel")
    parser.add_argument("--output", default="microduck_robust_parallel")
    args = parser.parse_args()

    if min(args.timesteps, args.n_envs, args.n_steps, args.batch_size) <= 0:
        parser.error("training values must be greater than zero")
    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        parser.error("batch-size must divide n-envs * n-steps")

    torch.set_num_threads(args.torch_threads)
    per_worker_horizon = max(1, args.timesteps // args.n_envs)
    env = create_vector_env(args.n_envs, per_worker_horizon)

    try:
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
        if os.path.exists(args.start_model) or os.path.exists(args.start_model + ".zip"):
            transfer_policy(args.start_model, model)
        else:
            print(f"Start model {args.start_model!r} not found; training from scratch.")

        print(
            f"Training {args.n_envs} environments for {args.timesteps:,} samples. "
            "Curriculum: commands -> pushes -> friction/motors -> rough terrain."
        )
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=True)
        model.save(args.output)
        print(f"Robust model saved as {args.output}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()

