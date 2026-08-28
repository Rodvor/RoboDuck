from env.microduck_env import MicroDuckEnv

from stable_baselines3 import PPO


env = MicroDuckEnv()


model = PPO(
    "MlpPolicy",
    env,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=256,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    verbose=1,

    device="cpu",
)


model.learn(
    total_timesteps=100_000
)


model.save(
    "microduck_stand"
)

env.close()