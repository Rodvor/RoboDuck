import time

from stable_baselines3 import PPO

from env.microduck_walk_env import MicroDuckWalkEnv


model = PPO.load("microduck_walk_parallel", device="cpu")
env = MicroDuckWalkEnv(render_mode="human")
observation, _ = env.reset()
episode_start_x = env.data.qpos[0]

print("Watching MicroDuck walk. Close the MuJoCo window to stop.")

while True:
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.01)

    if terminated or truncated:
        distance = env.data.qpos[0] - episode_start_x
        seconds = env.step_count * env.frame_skip * env.model.opt.timestep
        print(f"Episode: {distance:.2f} m in {seconds:.1f} s")
        observation, _ = env.reset()
        episode_start_x = env.data.qpos[0]

