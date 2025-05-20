import pandas as pd

# Load the dataset
file_path = "Mall_Customers.csv"
data = pd.read_csv(file_path)

# Sort the dataset by 'Annual Income (k$)' in descending order
top_5_income = data.sort_values(by='Annual Income (k$)', ascending=False).head(5)

# Print the top 5 rows
print("Top 5 persons with the highest annual income:")
print(top_5_income)