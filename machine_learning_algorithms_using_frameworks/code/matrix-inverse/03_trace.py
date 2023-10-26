# matrix trace
from numpy import array
from numpy import trace
# define matrix
A = array([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9]])
print(A)
# calculate trace
B = trace(A)
print(B)