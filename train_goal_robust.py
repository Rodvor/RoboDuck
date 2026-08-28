import argparse
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "roboduck-matplotlib"),
)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.microduck_goal_robust_env import MicroDuckGoalRobustEnv


def model_path(model_name):
    path = Path(model_name)
    if path.exists():
        return path
    zipped = Path(str(path) + ".zip")
    return zipped if zipped.exists() else None


def make_env(curriculum_horizon, curriculum_start_fraction):
    def _init():
        return Monitor(MicroDuckGoalRobustEnv(
            curriculum_horizon=curriculum_horizon,
            curriculum_start_fraction=curriculum_start_fraction,
        ))
    return _init


def create_vector_env(number_of_envs, curriculum_horizon, start_fraction):
    factories = [
        make_env(curriculum_horizon, start_fraction)
        for _ in range(number_of_envs)
    ]
    if number_of_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def checkpoint_step(path, prefix):
    match = re.fullmatch(
        re.escape(prefix) + r"_(\d+)_steps\.zip",
        path.name,
    )
    return int(match.group(1)) if match else -1


def inspect_goal_model(path):
    model = PPO.load(str(path), device="cpu")
    if model.observation_space.shape != (41,):
        raise ValueError(
            f"{path} has observation shape {model.observation_space.shape}, not (41,)"
        )
    return int(model.num_timesteps)


def find_resume_checkpoint(output, checkpoint_directory):
    output_path = model_path(output)
    prefix = Path(output).name
    candidates = []
    if output_path is not None:
        candidates.append((None, output_path))
    if checkpoint_directory.exists():
        for path in checkpoint_directory.glob(f"{prefix}_*_steps.zip"):
            step = checkpoint_step(path, prefix)
            if step >= 0:
                candidates.append((step, path))

    inspected = []
    for recorded_step, path in candidates:
        try:
            actual_step = inspect_goal_model(path)
        except Exception as error:
            print(f"Ignoring unreadable goal checkpoint {path}: {error}")
            continue
        if recorded_step is not None and actual_step != recorded_step:
            print(
                f"Checkpoint {path} says {recorded_step:,} in its filename "
                f"but contains {actual_step:,}; using the contained value."
            )
        inspected.append((actual_step, path))

    return max(inspected, default=(0, None), key=lambda item: item[0])


def transfer_policy(source_name, destination):
    """Transfer a walking/robust policy and zero-initialize new goal inputs."""
    source = PPO.load(str(source_name), device="cpu")
    if source.action_space.shape != destination.action_space.shape:
        raise ValueError(
            f"Cannot transfer action space {source.action_space} to "
            f"{destination.action_space}."
        )

    source_state = source.policy.state_dict()
    destination_state = destination.policy.state_dict()
    copied = 0
    expanded = 0
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
                and original.shape[1] < target.shape[1]
            ):
                target.zero_()
                target[:, :original.shape[1]].copy_(original)
                expanded += 1

    destination.policy.load_state_dict(destination_state)
    print(
        f"Transferred {copied} policy tensors and expanded {expanded} input "
        f"tensors from {source_name}."
    )


def linear_learning_rate(progress_remaining):
    return 3e-5 + 2.7e-4 * progress_remaining


def build_new_model(env, arguments):
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"log_std_init": -1.0},
        learning_rate=linear_learning_rate,
        n_steps=arguments.n_steps,
        batch_size=arguments.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        device="cpu",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically train goal-directed, push-resistant MicroDuck "
            "locomotion through a staged curriculum."
        )
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000_000)
    parser.add_argument("--n-envs", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5_000_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output", default="microduck_goal_robust")
    parser.add_argument("--run-dir", default="goal_training")
    parser.add_argument("--robust-model", default="microduck_robust_parallel")
    parser.add_argument("--walk-model", default="microduck_walk_parallel")
    args = parser.parse_args()

    positive_values = (
        args.total_timesteps,
        args.n_envs,
        args.n_steps,
        args.batch_size,
        args.checkpoint_every,
    )
    if min(positive_values) <= 0:
        parser.error("training and checkpoint values must be greater than zero")
    if args.eval_episodes < 0:
        parser.error("eval-episodes cannot be negative")
    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size != 0:
        parser.error("batch-size must divide n-envs * n-steps")

    torch.set_num_threads(args.torch_threads)
    run_directory = Path(args.run_dir)
    checkpoint_directory = run_directory / "checkpoints"
    best_directory = run_directory / "best"
    evaluation_directory = run_directory / "evaluations"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    best_directory.mkdir(parents=True, exist_ok=True)
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    completed_samples, resume_path = find_resume_checkpoint(
        args.output,
        checkpoint_directory,
    )
    if completed_samples >= args.total_timesteps:
        print(
            f"Goal training is already complete at {completed_samples:,} samples "
            f"(target {args.total_timesteps:,})."
        )
        if resume_path != model_path(args.output):
            PPO.load(str(resume_path), device="cpu").save(args.output)
            print(f"Consolidated the latest checkpoint into {args.output}.zip")
        return

    remaining_samples = args.total_timesteps - completed_samples
    start_fraction = completed_samples / args.total_timesteps
    per_worker_horizon = max(1, args.total_timesteps // args.n_envs)
    env = create_vector_env(args.n_envs, per_worker_horizon, start_fraction)
    eval_env = None
    model = None

    try:
        if resume_path is not None:
            print(
                f"Resuming goal checkpoint {resume_path} at {completed_samples:,} "
                f"of {args.total_timesteps:,} samples "
                f"({100.0 * start_fraction:.1f}% curriculum)."
            )
            model = PPO.load(
                str(resume_path),
                env=env,
                device="cpu",
                n_steps=args.n_steps,
                batch_size=args.batch_size,
            )
            reset_num_timesteps = False
        else:
            model = build_new_model(env, args)
            robust_path = model_path(args.robust_model)
            walk_path = model_path(args.walk_model)
            if robust_path is not None:
                print(f"Starting goal curriculum from robust model {robust_path}.")
                transfer_policy(robust_path, model)
            elif walk_path is not None:
                print(
                    f"Robust model not found; starting goal curriculum from "
                    f"walking model {walk_path}."
                )
                transfer_policy(walk_path, model)
            else:
                print("No robust or walking checkpoint found; training from scratch.")
            reset_num_timesteps = True

        callback_step_scale = args.n_envs
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // callback_step_scale),
            save_path=str(checkpoint_directory),
            name_prefix=Path(args.output).name,
            verbose=1,
        )
        callbacks = [checkpoint_callback]

        if args.eval_episodes > 0:
            eval_factory = [
                lambda: Monitor(MicroDuckGoalRobustEnv(evaluation=True))
            ]
            if args.n_envs == 1:
                eval_env = DummyVecEnv(eval_factory)
            else:
                # Match the training VecEnv type so Stable-Baselines can
                # compare the environments without a spurious warning.
                eval_env = SubprocVecEnv(eval_factory, start_method="spawn")
            eval_env.seed(10_000)
            callbacks.append(EvalCallback(
                eval_env,
                n_eval_episodes=args.eval_episodes,
                eval_freq=max(1, args.checkpoint_every // callback_step_scale),
                log_path=str(evaluation_directory),
                best_model_save_path=str(best_directory),
                deterministic=True,
                render=False,
                verbose=1,
                warn=False,
            ))

        print(
            f"Training automatically from {completed_samples:,} to approximately "
            f"{args.total_timesteps:,} total goal samples across {args.n_envs} "
            f"environments. Checkpoints and evaluation occur every "
            f"{args.checkpoint_every:,} samples."
        )
        model.learn(
            total_timesteps=remaining_samples,
            reset_num_timesteps=reset_num_timesteps,
            callback=CallbackList(callbacks),
        )
        model.save(args.output)
        print(f"Completed goal model saved as {args.output}.zip")
    except KeyboardInterrupt:
        if model is not None:
            model.save(args.output)
            print(f"\nInterrupted safely; resumable model saved as {args.output}.zip")
    finally:
        if eval_env is not None:
            eval_env.close()
        env.close()


if __name__ == "__main__":
    main()
