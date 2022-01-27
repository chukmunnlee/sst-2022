# import packages
import time, gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(8080)

GAMMA = .9
ALPHA = .1
MAX_PROB = .9
MIN_PROB = .02
MIN_ALPHA = 0.001
MAX_ALPHA = 0.1
DECAY = .001

EPISODES = 100000

CARTPOLE_ENV = 'CartPole-v0'

exploration = lambda ep: MIN_PROB + (MAX_PROB - MIN_PROB) * np.exp(-DECAY * ep)
learn_rate = lambda ep: MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * np.exp(-DECAY * ep)

def policy(st, eps):
    #dot = np.dot(st, w)
    if np.random.random() > eps:
        dot = np.dot(st, w.T)
        return np.argmax(dot)

    return np.random.randint(0, 2)

def q_value(st, act):
    dot = np.dot(st, w[act].T)
    return dot

steps_per_episode = []

w = np.random.random((2, 4));
w = np.ones(shape=(2, 4))
#w = np.random.random((4, 2));

# Create the environment
# CartPole-v0 - 200 steps
# CartPole-v1 - 500 steps
env = gym.make(CARTPOLE_ENV)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.ion()

for e in tqdm(range(EPISODES)):
    # Reset
    epsilon = exploration(e)
    alpha = learn_rate(e)

    state = env.reset()
    action = policy(state, epsilon)

    done = False
    steps = 0

    while not done:

        #env.render()
        # take an action
        steps += 1
        new_state, reward, done, _ = env.step(action)

        if done:
            td_target = reward
        else:
            next_action = policy(new_state, epsilon)
            td_target = reward + (GAMMA * q_value(new_state, next_action))

        td_error = td_target - q_value(state, action)

        # element-wise opration
        #print(f'before: action = {action}, w = {w[action]}')
        #w[action] = w[action] + ALPHA * td_error * state
        w[action] += alpha * td_error * state
        #print(f'\tafter: action = {action}, w = {w[action]}')

        # long form of the above
        #w[action][0] = w[action][0] + ALPHA * td_error * state[0]
        #w[action][1] = w[action][1] + ALPHA * td_error * state[1]
        #w[action][2] = w[action][2] + ALPHA * td_error * state[2]
        #w[action][3] = w[action][3] + ALPHA * td_error * state[3]

        state = new_state
        action = next_action

    steps_per_episode.append(steps)
    ax.cla()
    ax.plot(steps_per_episode)
    ax.set_title(f'Episodes: {e}')
    ax.axhline(200, c='red')
    plt.draw()
    plt.pause(.1)

#plt.hlines(200, xmin=0, xmax=EPISODES, colors='r')
#plt.plot(steps_per_episode, label='steps per episode')
#plt.legend()
#plt.title(f'Number or episodes: {EPISODES}')
#plt.show()