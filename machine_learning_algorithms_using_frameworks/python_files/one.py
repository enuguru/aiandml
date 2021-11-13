# Importing all necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# Initializing the model and fitting the model with train data
model = LinearRegression()
model.fit(X_train,y_train)
# Generating predictions over test data
predictions = model.predict(X_test)
# Evaluating the model using MAE Evaluation Metric
print(mean_absolute_error(y_test, predictions))


# Importing all necessary libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Defining our own MSE function
def own_mean_squared_error(actual, predictions):
    return ((predictions - actual) ** 2).mean()
# Initializing the model and fitting the model with train data
model = RandomForestRegressor(
               n_estimators = 100,
               criterion = 'mse'
        )
model.fit(X_train,y_train)
# Generating predictions over test data
predictions = model.predict(X_test)
# Evaluating the model using MSE Evaluation Metric
print(mean_squared_error(y_test, predictions))
print(own_mean_squared_error(y_test, predictions))