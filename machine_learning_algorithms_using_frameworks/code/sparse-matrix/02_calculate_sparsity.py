# sparsity calculation
from numpy import array
from numpy import count_nonzero
# create dense matrix
A = array([
	[1, 0, 0, 1, 0, 0],
	[0, 0, 2, 0, 0, 1],
	[0, 0, 0, 2, 0, 0]])
print(A)
# calculate sparsity
sparsity = 1.0 - count_nonzero(A) / A.size
print(sparsity)