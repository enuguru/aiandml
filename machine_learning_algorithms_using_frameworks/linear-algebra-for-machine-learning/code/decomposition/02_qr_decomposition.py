# QR decomposition
from numpy import array
from numpy.linalg import qr
# define rectangular matrix
A = array([
	[1, 2],
	[3, 4],
	[5, 6]])
print(A)
# factorize
Q, R = qr(A, 'complete')
print(Q)
print(R)
# reconstruct
B = Q.dot(R)
print(B)