import numpy as np

state = np.array([[ 1, 2, 3, 4 ]])
term1 = np.array([ 1, 2, 3, 4 ])
term2 = np.array([ 5, 6, 7, 8 ])

weights = np.array([
      [ 2, 3, 4, 5 ],
      [ 3, 4, 5, 6 ]
   ])

print('state.shape = ', state.shape)
print('weights.shape = ', weights.shape)
print('weights = \n', weights)

print('weights.shape = ', weights.T.shape)
print('weights.T = \n', weights.T)

value = np.dot(state, weights.T)
print('dot product = ', value)

value = np.multiply(term1, term2)
print('pairwise multiply = ', value)

