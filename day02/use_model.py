import pickle
import numpy as np

PICKLE_FILE_NAME = './agent-smith-v.pickle'

with open(PICKLE_FILE_NAME, 'rb') as f:
    q_table = pickle.load(f)

# start an infinite loop

# read 3 values - hand, dealer, usable_ace