from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pickle

# load the iris datasets as an example 
dataset = datasets.load_iris()

# make predictions
X = dataset.data 
Y = dataset.target
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

pickle_fname = 'logreg_pickle.model'

#  - Load the model from disk
pickle_model = pickle.load(open(pickle_fname, 'rb'))

# Sanity checks: Make prediction with all the saved models on unseen data.
result_2 = pickle_model.score(X_test, Y_test) # saved using picke
print("Result: {}".format(result_2))
