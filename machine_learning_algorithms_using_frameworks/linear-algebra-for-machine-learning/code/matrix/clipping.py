import numpy as np
A = np.matrix('1 2 3; 4 5 6')
print("Matrix is :\n", A)
#clipping matrix
print("Clipped matrix is :\n", A.clip(1,4))