import pandas as pd
import numpy as np

# Create a dataset with two columns and 25 samples
data = {
    'col1': np.random.rand(25),  # First column with random values
    'col2': np.random.randint(0, 100, 25)  # Second column with random integers
}

df = pd.DataFrame(data)

# Print the dataset
print(df)
