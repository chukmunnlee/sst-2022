import numpy as np
from matplotlib import pyplot as plt

MAX_PROB = .9
MIN_PROB = .02
MIN_ALPHA = 0.001
MAX_ALPHA = 0.1
DECAY = .001

EPISODES = 10000

exploration = lambda ep: MIN_PROB + (MAX_PROB - MIN_PROB) * np.exp(-DECAY * ep)
learn_rate = lambda ep: MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * np.exp(-DECAY * ep)

e = []
lr = []

for i in range(EPISODES):
    e.append(exploration(i))
    lr.append(learn_rate(i))

fig = plt.figure();

ax = fig.add_subplot(121)
ax.plot(e, label='exploration')

ax = fig.add_subplot(122)
ax.plot(lr, label='learning rate')

plt.legend()
plt.show()