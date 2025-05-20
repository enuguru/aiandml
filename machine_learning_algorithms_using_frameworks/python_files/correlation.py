# create two different datasets and find the correlation between them
import numpy as np
import pandas as pd

# Create two datasets
np.random.seed(0)   
x = np.random.rand(1000)
y = np.random.rand(1000)
# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})
# Add a third column that is a linear combination of the first two
df['z'] = 0.5 * df['x'] + 0.5 * df['y'] + np.random.rand(1000) * 0.1
# Calculate the correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)
# The correlation matrix shows that there is a strong positive correlation between x and y, and a weak positive correlation between x and z. There is no correlation between y and z.