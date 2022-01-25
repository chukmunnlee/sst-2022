# import matplotlib, numpy 
import gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(42)

EPISODES = 5000
MOD = EPISODES // 10
ALPHA = 0.1
EPSILON = [ .9, .8, .7, .6, .5, .3, .3, .1, .05, .02 ]

decay = 0.01
max_prob = .9
min_prob = .03

# exploration
def epsilon(ep):
    return min_prob + (max_prob - min_prob) * np.exp(-ep * decay)

def policy(st, eps):

    # exploit by taking the max action
    if np.random.random() > eps:
        curr_state = total_reward[st]
        q_values = np.array(curr_state) 
        return np.argmax(q_values)

    # explore by randomly selecting an action
    return np.random.randint(0, 4)

def print_policy(q_values, cols):
    for i in range(len(q_values)):
        if 0 == (i % cols):
            print()
        dir = np.argmax(q_values[i])
        if (np.count_nonzero(q_values[i]) <= 0):
            print('x', end='')
        elif 0 == dir:
            print('<', end='')
        elif 1 == dir:
            print('V', end='')
        elif 2 == dir:
            print('>', end='')
        else:
            print('^', end='')

    print()

# create the environment
env = gym.make('FrozenLake-v1')

# array 16 x 4 - init 0
total_reward = np.zeros(shape=(16, 4))

# how many times to run the game
for e in tqdm(range(EPISODES)):
    # at the start of every episode, reset the environment
    state = env.reset()
    done = False
    rollouts = []

    # in one episode
    while not done:
        #env.render()
        # take a step
        eps = epsilon(e)
        action = policy(state, eps)
        new_state, reward, done, _ = env.step(action)
        reward = 0.5
        # (state, action, reward)
        rollouts.append((state, action, reward))
        state = new_state

    # first visit
    visited = set()
    for i in range(len(rollouts)):

        # check if rollouts[i] in visited
        # if it is take the next rollout
        st, act, _ = rollouts[i]
        if ((st, act) in visited):
            continue

        # if it is not, add to visited
        visited.add((st, act))

        # sum all rewards from current state, add to total_reward table
        gain = sum([ r[2] for r in rollouts[i:] ])
        total_reward[st, act] = total_reward[st, act] + ALPHA * (gain - total_reward[st, act]) 

print(total_reward)

print_policy(total_reward, 4)

#print('total_reward')
#print(total_reward)
#
#print()
#print('n_action')
#print(n_action)