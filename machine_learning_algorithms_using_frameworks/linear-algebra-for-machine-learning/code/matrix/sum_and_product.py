import numpy as np
A = np.matrix('1 2 3; 4 5 6; 7,8,9')
print("Matrix is :\n", A)
#cumulative product along axis = 0
print("Cumulative product of elements along axis 0 is : \n", A.cumprod(0))
#cumulative sum along axis = 0
print("Cumulative sum of elements along axis 0 is : \n", A.cumsum(0))