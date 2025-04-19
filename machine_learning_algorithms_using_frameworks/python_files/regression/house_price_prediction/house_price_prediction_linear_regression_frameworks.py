
# python implementation of simple Linear Regression on salary data of software engineers

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# step 1: reading the data and splitting it to input and output
dataset = pd.read_csv('home.csv')
#inputx = dataset.iloc[:, :-1].values
inputx = dataset.iloc[:, 0:2].values
outputy = dataset.iloc[:, 2].values
print(outputy)


# step 2: select one thirds of the data for testing and two thirds for training
input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/4, random_state = 0)


# step 3: selecting the simple Linear Regression model
model = LinearRegression()
print("\nThe parameters of the model are\n\n",model.get_params())
#print(model.set_params())
print("\nThe model we are using is ", model.fit(input_train, output_train))


# step 4: testing or model prediction using testinput
squarefeet = float(input("\nGive square feet of the house  "))
bedrooms = float(input("\nGive the number of bed rooms in the house  "))
testinput = [[squarefeet,bedrooms]]
predicted_output = model.predict(testinput)
print('\nThe test input is as follows ',testinput) 
print('\nThe predicted house price is ',predicted_output) 
yes = input("\nCan I proceed\n")


# step 5: Printing the testing results
print("\nThe test input (square feet and the number of bed rooms) is as follows \n")
print(input_test)
# model predicting the Test set results
predicted_output = model.predict(input_test)
print("\nThe predicted price of the house for the test input is as follows \n")
print(predicted_output)
