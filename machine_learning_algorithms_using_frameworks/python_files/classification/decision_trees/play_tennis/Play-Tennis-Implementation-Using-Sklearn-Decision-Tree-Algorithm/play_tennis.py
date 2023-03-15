
# Load libraries
import numpy as np
import pandas as pd
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier # import the classifier
from sklearn.model_selection import train_test_split
import pydotplus

# read the data
df=pd.read_csv("Play Tennis.csv")
value=['Outlook','Temprature','Humidity','Wind']
print(df)

#machine learning algorithms can only learn from numbers (int, float, doubles .. )
#so let us encode it to int
string_to_int= preprocessing.LabelEncoder()  #encode your data
df=df.apply(string_to_int.fit_transform) #fit and transform it
print(df)

#To divide our data into attribute set and Label:
feature_cols = ['Outlook','Temprature','Humidity','Wind']
X = df[feature_cols ]          #contains the attribute 
y = df.Play_Tennis             #contains the label

#To divide our data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) 

# perform training 
# create a classifier object
classifier = DecisionTreeClassifier(criterion="entropy", random_state=100)
classifier.fit(X_train, y_train)  

#Predict the response for test dataset
y_pred= classifier.predict(X_test)  
data_p=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
print(data_p)

#draw the graph
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
        filled=True, rounded=True,
        special_characters=True,feature_names =value,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Play Tennis.png')
Image(graph.create_png())
