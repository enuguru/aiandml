import numpy as np
from statistics import mode
import random

# Generate a list of 25 random integers between 1 and 100
numbers = [random.randint(1, 100) for _ in range(25)]

# Calculate mean, median, and mode
mean = np.mean(numbers)
median = np.median(numbers)
mode_value = mode(numbers)

# Print the results
print(f"List of numbers: {numbers}")
print(f"Mean: {mean:.2f}")
print(f"Median: {median}")
print(f"Mode: {mode_value}")
