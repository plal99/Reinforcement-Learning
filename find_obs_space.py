import gym
env = gym.make("MountainCar-v0")
obs = env.reset()
print(env.observation_space.low)
print(env.observation_space.high)
