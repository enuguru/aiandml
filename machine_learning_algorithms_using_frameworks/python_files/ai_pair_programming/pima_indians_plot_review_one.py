import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "../../../datasets/pima-indians_classification_train.csv"
data = pd.read_csv(file_path)

# Exclude the target variable (last column)
feature_names = data.columns[:-1]

# Convert all columns explicitly to numeric, forcing coercion for any non-numeric values
data = data.apply(pd.to_numeric, errors='coerce')

# Replace zero values in selected medical-related features with the column mean
features_to_replace_zeros = ['6', '148', '72', '35', '0', '33.6']
for feature in features_to_replace_zeros:
    data[feature] = data[feature].replace(0, data[feature].mean())

# Apply log transformation to stabilize extreme values (avoiding log(0) by adding a small constant)
data_transformed = data.copy()
for feature in feature_names:
    data_transformed[feature] = np.log1p(data[feature])  # log(1 + x) to prevent log(0)

# Clip extreme outliers to the 99th percentile for better visualization
for feature in feature_names:
    upper_limit = np.percentile(data_transformed[feature], 99)
    data_transformed[feature] = np.clip(data_transformed[feature], None, upper_limit)

# Set the style for plots
sns.set(style="whitegrid")

# 📌 Generate Probability Density Function (PDF) using Histogram-based KDE
for feature in feature_names:
    plt.figure(figsize=(8, 5))
    sns.histplot(data_transformed[feature], kde=True, bins=30, color='b', edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"Probability Density Function (PDF) for {feature} (Histogram-based KDE)")
    plt.grid()
    plt.show()

# 📌 Generate Box-and-Whisker Plots
for feature in feature_names:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data_transformed[feature], color='orange')
    plt.xlabel(feature)
    plt.title(f"Box-and-Whisker Plot for {feature} (Log Transformed)")
    plt.grid()
    plt.show()

# 📌 Generate Histograms
for feature in feature_names:
    plt.figure(figsize=(8, 5))
    plt.hist(data_transformed[feature], bins=20, color='g', alpha=0.7, edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {feature} (Log Transformed)")
    plt.grid()
    plt.show()
