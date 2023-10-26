# diagonal matrix
from numpy import array
from numpy import diag
# define square matrix
M = array([
	[1, 2, 3],
	[1, 2, 3],
	[1, 2, 3]])
print(M)
# extract diagonal vector
d = diag(M)
print(d)
# create diagonal matrix from vector
D = diag(d)
print(D)