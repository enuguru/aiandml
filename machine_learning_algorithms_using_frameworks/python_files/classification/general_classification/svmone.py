import pandas as pd
# load data file
import numpy as np
df=pd.read_csv("https://reneshbedre.github.io/assets/posts/svm/hd_cleveland.csv")
df.head(2)
df.isna().sum()
df['ca'].fillna(value=df['ca'].mode()[0], inplace=True)
df['thal'].fillna(value=df['thal'].mode()[0], inplace=True)

# now check if there are any missing values
df.isna().any().any()
False  # there is no any missing values
from matplotlib import pyplot as plt
import seaborn as sns
ax = sns.countplot(x='disease', data=df)
plt.show()
from sklearn.model_selection import train_test_split
# Get the independent variables
X = df.iloc[:,1:13]
# Get the dependent variables
y = df['disease']

# split the dataset into 70% as training and 30% as testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# random_state parameter ensures that the train_test_split function will reproduce the same train and testing
# dataset every time. Set random_state to any integer.


from sklearn.svm import SVC
from array import array

svm = SVC(C=1, kernel='linear', random_state=1)
svm.fit(X=X_train, y=y_train)
y_pred = svm.predict(X=X_test)
y_pred
# output
array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
       0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 1, 1], dtype=np.int64)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# confusion matrix
confusion_matrix(y_true=y_test, y_pred=y_pred)
# output
array([[41,  7],
       [ 5, 38]], dtype=np.int64)

# fitted SVM model accuracy
accuracy_score(y_true=y_test, y_pred=y_pred)
# output
0.8681318681318682

from sklearn.metrics import roc_curve, auc, roc_auc_score
from bioinfokit.visuz import stat

y_score = svm.decision_function(X=X_test)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)
auc = roc_auc_score(y_true=y_test, y_score=y_score)
# plot ROC
stat.roc(fpr=fpr, tpr=tpr, auc=auc, shade_auc=True, per_class=True, legendpos='upper center', legendanchor=(0.5, 1.08), 
         legendcols=3)

from sklearn.metrics import precision_recall_curve, average_precision_score, plot_precision_recall_curve
import matplotlib.pyplot as plt

average_precision = average_precision_score(y_true=y_test, y_score=y_score)
# plot AUPRC
disp = plot_precision_recall_curve(estimator=svm, X=X_test, y=y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
plt.show()



