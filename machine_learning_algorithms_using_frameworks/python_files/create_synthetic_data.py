import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load your dataset
# Assuming you have a DataFrame named 'df' with features and target variable
df = pd.read_csv('path_to_your_auto_insurance_dataset.csvC:\Users\karth\aiandml\datasets\salary_regression_train.csv')

# Separate features and target variable
X = df.drop(columns=['target'])  # Replace 'target' with your actual target column name
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to generate synthetic data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Combine the synthetic data with the original data
augmented_df = pd.concat([pd.DataFrame(X_train_res), pd.DataFrame(y_train_res, columns=['target'])], axis=1)

# Save or use the augmented dataset
augmented_df.to_csv('augmented_auto_insurance_dataset.csv', index=False)

# Now you can use 'augmented_df' for training your regression model
