import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from env.microduck_env import MicroDuckEnv


# ==========================================
# SETTINGS
# ==========================================

# Use 16 parallel physics workers by default. NUM_ENVS remains available for
# easy machine-specific overrides.
DEFAULT_NUM_ENVS = 16
NUM_ENVS = int(os.getenv("NUM_ENVS", str(DEFAULT_NUM_ENVS)))
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "1000000"))
N_STEPS = int(os.getenv("N_STEPS", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
MODEL_NAME = os.getenv("MODEL_NAME", "microduck_stand_parallel")


# ==========================================
# CREATE ONE ENVIRONMENT
# ==========================================

def make_env():

    def _init():

        env = MicroDuckEnv(
            render_mode=None
        )

        return env

    return _init


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("======================================")
    print("       PARALLEL MICRODUCK TRAINING")
    print("======================================")

    print(f"Environments: {NUM_ENVS}")
    print(f"Timesteps:    {TOTAL_TIMESTEPS}")

    # Create parallel environments
    env = SubprocVecEnv(
        [make_env() for _ in range(NUM_ENVS)]
    )

    # PPO
    model = PPO(
        "MlpPolicy",
        env,
        # Full-scale unit Gaussian exploration is too violent for a balancing
        # controller with 14 servos. exp(-1) ~= 0.37 gives useful perturbations
        # while allowing early episodes to remain upright long enough to learn.
        policy_kwargs={"log_std_init": -1.0},
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # Standing is a precision task; persistent entropy at 0.01 keeps
        # injecting unnecessarily large servo offsets after a balance is found.
        ent_coef=0.0,
        verbose=1,
        device="cpu",
    )

    # ==========================================
    # TRAIN
    # ==========================================

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS
    )

    # ==========================================
    # SAVE
    # ==========================================

    model.save(MODEL_NAME)

    env.close()

    print()
    print("Training finished!")
    print("Saved as:")
    print(f"{MODEL_NAME}.zip")
