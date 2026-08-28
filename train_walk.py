import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from env.microduck_walk_env import MicroDuckWalkEnv


NUM_ENVS = int(os.getenv("NUM_ENVS", "16"))
# Walking needs substantially more policy updates than static balance. The
# shorter 400k validation run discovered forward gait in only some seeds;
# five million samples gives PPO room to consolidate it.
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "5000000"))
N_STEPS = int(os.getenv("N_STEPS", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
MODEL_NAME = os.getenv("MODEL_NAME", "microduck_walk_parallel")
START_MODEL = os.getenv("START_MODEL", "microduck_stand_parallel")


def make_env():
    def _init():
        return MicroDuckWalkEnv(render_mode=None)
    return _init


if __name__ == "__main__":
    print("======================================")
    print("        PARALLEL WALK TRAINING")
    print("======================================")
    print(f"Environments: {NUM_ENVS}")
    print(f"Timesteps:    {TOTAL_TIMESTEPS}")

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    start_path = Path(START_MODEL).with_suffix(".zip")
    if start_path.exists():
        print(f"Starting from standing policy: {start_path}")
        model = PPO.load(start_path, env=env, device="cpu")
        # A converged standing controller explores very little. Restore enough
        # variance to discover alternating steps in the new task.
        model.policy.log_std.data.fill_(-1.0)
    else:
        print("Standing policy not found; starting walking policy from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={"log_std_init": -1.0},
            learning_rate=3e-4,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            device="cpu",
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
    model.save(MODEL_NAME)
    env.close()
    print(f"Training finished! Saved as {MODEL_NAME}.zip")
