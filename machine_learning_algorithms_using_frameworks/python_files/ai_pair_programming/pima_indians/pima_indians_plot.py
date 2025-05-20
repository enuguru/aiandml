
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the dataset
file_path = "../../../datasets/pima-indians_classification_train.csv"
data = pd.read_csv(file_path)

# Define column names
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]

# Assign column names
data.columns = column_names

# Convert all values explicitly to float64 to avoid potential dtype issues
data = data.astype(np.float64)

# Generate separate Probability Density Function (PDF) plots as continuous curves for each variable
for column in data.columns[:-1]:  # Exclude 'Outcome' column
    plt.figure(figsize=(8, 5))
    column_data = data[column].dropna().astype(np.float64).values  # Ensure clean numeric data

    # Compute Kernel Density Estimate
    kde = gaussian_kde(column_data)
    x_range = np.linspace(min(column_data), max(column_data), 1000)
    plt.plot(x_range, kde(x_range), label=f"KDE of {column}", linewidth=2)

    plt.title(f"Probability Density Function (PDF) of {column}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate separate Box-and-Whisker plots for each variable
for column in data.columns[:-1]:  # Exclude 'Outcome' column
    plt.figure(figsize=(6, 5))
    plt.boxplot(data[column].dropna().astype(np.float64).values, vert=True, patch_artist=True)
    plt.title(f"Box-and-Whisker Plot of {column}")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

# Generate separate Histograms for each variable
for column in data.columns[:-1]:  # Exclude 'Outcome' column
    plt.figure(figsize=(8, 5))
    plt.hist(data[column], bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of {column}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
