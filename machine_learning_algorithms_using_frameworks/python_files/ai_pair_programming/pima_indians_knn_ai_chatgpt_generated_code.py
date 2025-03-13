import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the training data
train_file_path = "../../../datasets/pima-indians_classification_train.csv"
test_file_path = "../../../datasets/pima-indians_classification_test.csv"

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Define column names
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]

# Assign column names
train_data.columns = column_names
test_data.columns = column_names[:-1]  # Exclude 'Outcome' for test data

# Split features and target variable
X_train = train_data.iloc[:, :-1].values  # First 8 columns as features
y_train = train_data.iloc[:, -1].values   # Last column as target

X_test = test_data.values  # First 8 columns as features (no target)

# Initialize the KNN classifier with K=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn.fit(X_train, y_train)

# Predict using the test dataset
y_pred = knn.predict(X_test)

# Display predictions
predictions_df = pd.DataFrame(test_data, columns=column_names[:-1])
predictions_df["Predicted Outcome"] = y_pred
print("\nPredictions:")
print(predictions_df)

# Evaluate the model using the train dataset
y_train_pred = knn.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_train, y_train_pred)
print("\nClassification Report:")
print(class_report)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
