import numpy as np
from matplotlib import pyplot as plt

EPISODES = 1000

decay = 0.01
max_prob = .9
min_prob = .03

d = []

for t in range(EPISODES):
   v = min_prob + (max_prob - min_prob) * np.exp(-t * decay)
   d.append(v)

plt.plot(d)
plt.show()