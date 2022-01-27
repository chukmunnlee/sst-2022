import numpy as np
import gym, time
import flappy_bird_gym

env = flappy_bird_gym.make("FlappyBird-v0")

state = env.reset()

print(state)

print(f'action = ' + str(env.action_space.sample()))

env.render()
time.sleep(30)