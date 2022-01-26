# import all the relevant packages
import gym, pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#np.random.seed(32)

# tuneables
GAMMA = 1
ALPHA = .1
DECAY = .01
MIN_PROB = 0.02
MAX_PROB = 0.9

EPISODES = 100000

def decay(ep):
    return MIN_PROB + (MAX_PROB - MIN_PROB) * np.exp(-ep * DECAY)

def policy(st, eps):
    if np.random.random() > eps:
        #take the max action
        return np.argmax(q_table[st])

    # take a random action
    return np.random.randint(0, 2)

def q_value(st, ac):
    return q_table[st][ac]

# create the Blackjack environment
env = gym.make('Blackjack-v1')

q_table = {}
for h in range(1, 26):
    for d in range(1, 11):
        for a in [ True, False ]:
            q_table[ (h, d, a) ] = np.array([ 0, 0 ])

next_action = 0
win_rate = []
wins = 0

for e in tqdm(range(EPISODES)):
    # reset the environemt
    eps = decay(e)
    state = env.reset()
    action = policy(state, eps)

    # reset done
    done = False

    while not done:
        new_state, reward, done, _ = env.step(action)

        if done:
            td_target = reward
            if 1 == reward:
                wins += 1
            win_rate.append(wins)

        else: 
            # one step reward + prediction
            next_action = policy(new_state, eps)
            predicted_future_reward = q_value(new_state, next_action)
            td_target = reward + (GAMMA * predicted_future_reward)

        #SARSA - state, action, reward, new_state, next_action
        td_error = td_target - q_value(state, action)

        # update the q_value for state, action
        q_table[state][action] += ALPHA * td_error

        action = next_action
        state = new_state

# save the model
with open('./agent-smith.pickle', 'wb') as f:
    pickle.dump(q_table, f)

# plot the win rate
plt.plot(win_rate, label="win rate")
plt.legend()
plt.show()