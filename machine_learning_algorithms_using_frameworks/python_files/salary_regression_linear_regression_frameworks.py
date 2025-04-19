
# python implementation of simple Linear Regression on salary data of software engineers

# step 1: import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# step 2: reading the data and splitting it to input and output
dataset = pd.read_csv('C:\\Users\\karth\\aiandml\\datasets\\salary_regression_train.csv')
print(dataset)
#inputx = dataset.iloc[:, :-1].values
inputx = dataset.iloc[:, 0:1].values
outputy = dataset.iloc[:,1].values
#print(inputx)

# step 3: select one thirds of the data for testing and two thirds for training
input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/4, random_state = 0)


# step 4: selecting the simple Linear Regression model
model = LinearRegression()
print("\nThe parameters of the model are\n\n",model.get_params())
#print(model.set_params())
print("\nThe model we are using is ", model.fit(input_train, output_train))


# step 5: testing or model prediction using testinput
years = float(input("\nGive number of years of experience  "))
testinput = [[years]]
predicted_output = model.predict(testinput)
print('\nThe number of years of experience is ',testinput) 
print('\nThe salary for the number of years of experience is ',predicted_output) 
yes = input("\nCan I proceed\n")


# step 6: Visualising the training results
plt.scatter(input_train, output_train, color = 'red')
plt.plot(input_train, model.predict(input_train), color = 'yellow')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# step 7: Printing the testing results
print("\nThe test input (number of years of experience) is as follows \n")
print(input_test)
# model predicting the Test set results
predicted_output = model.predict(input_test)
print("\nThe output (salary) for the test input is as follows \n")
print(predicted_output)
