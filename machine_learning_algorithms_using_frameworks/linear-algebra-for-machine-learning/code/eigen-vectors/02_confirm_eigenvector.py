# confirm eigenvector
from numpy import array
from numpy.linalg import eig
# define matrix
A = array([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9]])
# factorize
values, vectors = eig(A)
# confirm first eigenvector
B = A.dot(vectors[:, 0])
print(B)
C = vectors[:, 0] * values[0]
print(C)