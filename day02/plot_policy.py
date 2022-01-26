import pickle
import numpy as np
from matplotlib import pyplot as plt

ACTION = 0

#PICKLE_FILE_NAME = './agent-smith-v2.pickle'
#TITLE = "Minimum user hand == 16"

PICKLE_FILE_NAME = './agent-smith.pickle'
TITLE = "No minimum user hand"

print(f'Loading {PICKLE_FILE_NAME}')

with open(PICKLE_FILE_NAME, 'rb') as f:
    q_table = pickle.load(f)

no_usable_ace = [ v for v in q_table if (not v[2]) and q_table[v][0] != q_table[v][1]]
stick_action = [ v for v in no_usable_ace if np.argmax(q_table[v]) == ACTION ]
hand = [ v[0] for v in stick_action ]
dealer = [ v[1] for v in stick_action ]
plt.scatter(hand, dealer, alpha=.5, label="No usable ace", s=30)

usable_ace = [ v for v in q_table if v[2] and q_table[v][0] != q_table[v][1]]
stick_action = [ v for v in usable_ace if np.argmax(q_table[v]) == ACTION ]
hand = [ v[0] for v in stick_action ]
dealer = [ v[1] for v in stick_action ]
plt.scatter(hand, dealer, alpha=.5, label="Usable ace", s=100)

plt.title(TITLE)
plt.legend()
plt.show()