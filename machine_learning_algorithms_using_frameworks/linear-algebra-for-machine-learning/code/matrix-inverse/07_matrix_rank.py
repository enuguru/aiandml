# matrix rank
from numpy import array
from numpy.linalg import matrix_rank
# rank 0
M0 = array([
	[1, 2, 3], 
     [4, 5, 6],
     [7, 8, 9]])
print(M0)
mr0 = matrix_rank(M0)
print(mr0)
