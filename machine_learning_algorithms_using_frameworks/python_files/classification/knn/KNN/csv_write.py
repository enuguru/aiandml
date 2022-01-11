
#!/usr/bin/env python
# coding: utf-8

# **This is my attempt to demonstrate my skill using Python to impliment K Nearest Neighbors.**
#
# I just grabbed the first random dataset from Kaggle that I thought I could use for classificiation.
# There was no "class" column so I made one from the Chance of Admit percentage column.
#
# **Please see knn.txt for additional details.**


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

# I just like to lowercase and remove spaces from my cols to minimize errors
knn_df = pd.read_csv("Admission_Predict.csv", names=["id", "gre", "toefl", "u_rating", "sop", "lor", "cgpa", "research", "pred"], header=0, index_col = 0)
#knn_df.head()
print("\n",knn_df)

# Scale the data.  I used different scalers for criteria and classes
# It just worked out to get me better results this time around.

# Use the MinMax Scaler and make the percentage True of False at 65% on the pred column
mm_scaler = MinMaxScaler()
knn_df['pred'] = mm_scaler.fit_transform(knn_df[['pred']])
knn_df['pred'] = knn_df['pred'].apply(lambda x: 0 if x <= 0.65 else 1)
print("\n",knn_df)
print(knn_df['pred'].value_counts()) # 0=220 1=180

knn_df.to_csv(r'new.csv')

# Use the Standard Scaler on the criteria
#s_scaler = StandardScaler()
#s_scaler.fit(knn_df.drop('pred',axis=1))
#scaled_features = s_scaler.transform(knn_df.drop('pred',axis=1))
