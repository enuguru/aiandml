# matrix rank
from numpy import array
from numpy.linalg import matrix_rank
# rank 0
M0 = array([
	[0,0],
	[0,0]])
print(M0)
mr0 = matrix_rank(M0)
print(mr0)
# rank 1
M1 = array([
	[1,2],
	[1,2]])
print(M1)
mr1 = matrix_rank(M1)
print(mr1)
# rank 2
M2 = array([
	[1,2],
	[3,4]])
print(M2)
mr2 = matrix_rank(M2)
print(mr2)