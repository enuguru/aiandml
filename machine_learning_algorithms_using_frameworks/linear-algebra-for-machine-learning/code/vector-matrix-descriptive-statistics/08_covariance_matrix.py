# covariance matrix
from numpy import array
from numpy import cov
# define matrix of observations
X = array([
	[1, 5, 8],
	[3, 5, 11],
	[2, 4, 9],
	[3, 6, 10],
	[1, 5, 10]])
print(X)
# calculate covariance matrix
Sigma = cov(X.T)
print(Sigma)