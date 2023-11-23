import numpy as np
A = np.matrix('1 2 3; 4 5 6')
print("Matrix is :\n", A)
#maximum indices
print("Maximum indices in A :\n", A.argmax(0))
#minimum indices
print("Minimum indices in A :\n", A.argmin(0))