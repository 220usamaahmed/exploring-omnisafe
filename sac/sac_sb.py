import gymnasium as gym
from stable_baselines3 import SAC

# Create environment
env = gym.make("MountainCarContinuous-v0")

# Create the SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("sac_mountaincar")

# Load the trained model
model = SAC.load("sac_mountaincar", env=env)

# Run the trained model
env = gym.make("MountainCarContinuous-v0", render_mode="human")
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)

    print(action)

    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
