
# python implementation of simple Linear Regression on salary data of software engineers

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# step 1: reading the data and splitting it to input and output
dataset = pd.read_csv('../../../datasets/salary_regression_train.csv')
inputx = dataset.iloc[:, :-1].values
outputy = dataset.iloc[:, 1].values


# step 2: select one thirds of the data for testing and two thirds for training
input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/4, random_state = 0)

X_test = input_test
Y_test = output_test

pickle_fname = 'logreg_pickle.model'

#  - Load the model from disk
pickle_model = pickle.load(open(pickle_fname, 'rb'))
print(X_test)
# Sanity checks: Make prediction with all the saved models on unseen data.
result_2 = pickle_model.predict(X_test) # saved using picke
print("Result: {}".format(result_2))
