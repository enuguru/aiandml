# invert matrix
from numpy import array
from numpy.linalg import inv
# define matrix
A = array([
	[1.0, 2.0],
	[3.0, 4.0]])
print(A)
# invert matrix
B = inv(A)
print(B)
# multiply A and B
I = A.dot(B)
print(I)