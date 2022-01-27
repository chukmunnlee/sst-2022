# import packages
import time, gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(8888)

GAMMA = .9
ALPHA = .1
MAX_PROB = .9
MIN_PROB = .02
MIN_ALPHA = 0.001
MAX_ALPHA = 0.1
DECAY = .001

EPISODES = 5000

CARTPOLE_ENV = 'CartPole-v0'

exploration = lambda ep: MIN_PROB + (MAX_PROB - MIN_PROB) * np.exp(-DECAY * ep)
learn_rate = lambda ep: MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * np.exp(-DECAY * ep)

def policy(st, eps):
    #dot = np.dot(st, w)
    if np.random.rand() > eps:
        dot = np.dot(st, w.T)
        return np.argmax(dot)

    return np.random.randint(0, 2)

def q_max_value(st):
    dot = np.dot(st, w.T)
    return np.amax(dot)

def q_value(st, act):
    dot = np.dot(st, w[act].T)
    return dot

steps_per_episode = []

w = np.ones(shape=(2, 4))
w = np.random.random((2, 4));
#w = np.random.random((4, 2));

# Create the environment
# CartPole-v0 - 200 steps
# CartPole-v1 - 500 steps
env = gym.make(CARTPOLE_ENV)

for e in tqdm(range(EPISODES)):
    # Reset
    epsilon = exploration(e)
    alpha = learn_rate(e)
    alpha = 0.1
    #epsilon = .2

    state = env.reset()

    done = False
    steps = 0

    while not done:

        #env.render()
        # take an action
        steps += 1
        action = policy(state, epsilon)
        new_state, reward, done, _ = env.step(action)

        if done:
            td_target = reward 
        else:
            td_target = reward + (GAMMA * q_max_value(new_state))

        td_error = td_target - q_value(state, action)

        # element-wise opration
        w[action] += alpha * td_error * state

        state = new_state

    steps_per_episode.append(steps)

plt.hlines(200, xmin=0, xmax=EPISODES, colors='r')
plt.plot(steps_per_episode, label='steps per episode')
plt.legend()
plt.title(f'Q-learning. Number or episodes: {EPISODES}')
plt.show()