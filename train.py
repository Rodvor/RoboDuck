from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.duck_env import DuckEnv


# ============================================
# Create environment
# ============================================

env = DuckEnv()


# ============================================
# Check Gymnasium compatibility
# ============================================

print("Checking environment...")

check_env(env)

print("Environment OK!")


# ============================================
# Create PPO
# ============================================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=64,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    ent_coef=0.01,

    device="cpu"
)


# ============================================
# Train
# ============================================

print("Starting training...")

model.learn(
    total_timesteps=500_000
)


# ============================================
# Save
# ============================================

model.save(
    "duck_standing"
)

print("Training finished!")