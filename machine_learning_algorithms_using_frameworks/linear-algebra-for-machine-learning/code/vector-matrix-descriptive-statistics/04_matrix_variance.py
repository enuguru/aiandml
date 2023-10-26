# matrix variances
from numpy import array
from numpy import var
# define matrix
M = array([
	[1,2,3,4,5,6],
	[1,2,3,4,5,6]])
print(M)
# column variances
col_var = var(M, ddof=1, axis=0)
print(col_var)
# row variances
row_var = var(M, ddof=1, axis=1)
print(row_var)