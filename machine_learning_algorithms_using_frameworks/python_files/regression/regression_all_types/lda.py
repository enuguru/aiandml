import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn import decomposition
from sklearn import datasets

# load dataset into Pandas DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/syamkakarla98/Linear-Discriminant-Analysis-Using-Python/master/iris.csv")
#df.to_csv('iris.csv')


from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
y=y.ravel()

##--------- Bar Graph for Explained Variance Ratio ------------
plt.bar([1,2],[97.876206, 1.751013],label='Principal Components',color='navy')
plt.legend()
plt.xlabel('Principal Components')
plt.xticks([1,2],['PC-1','PC-2'], fontsize=8, rotation=30)
plt.ylabel('Variance Ratio')
plt.title('Variance Ratio of IRIS  after LDA')
plt.show()

#------------------------------------------------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
reduced_data = lda.fit(x,y).transform(x)
principalDf = pd.DataFrame(data = reduced_data, columns = ['PC-1', 'PC-2'])
# Adding lables
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
# Plotting pc1 & pc2
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC-1', fontsize = 15)
ax.set_ylabel('PC-2', fontsize = 15)
ax.set_title('LDA on IRIS Dataset', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC-1']
               , finalDf.loc[indicesToKeep, 'PC-2']
               , c = color
               , s = 30)
ax.legend(targets)
ax.grid()
plt.show() # FOR SHOWING THE PLOT

#-------------------SENDING REDUCED DATA INTO CSV FILE------------
finalDf.to_csv('iris_after_LDA.csv')
