import pandas as pd
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
# Load the contents of the CSV file into a DataFrame
df = pd.read_csv("sportsdataone.csv")

# Swap the values of column A and B
df[['player_highest_wicket', 'Result']] = df[['Result', 'player_highest_wicket']]

label_encoder = preprocessing.LabelEncoder()
df['Result']= label_encoder.fit_transform(df['Result'])
df['Result'].unique()
df['Season']= label_encoder.fit_transform(df['Season'])
df['Season'].unique()
df['Opponent']= label_encoder.fit_transform(df['Opponent'])
df['Opponent'].unique()
df['First_selection']= label_encoder.fit_transform(df['First_selection'])
df['First_selection'].unique()
df['Offshore']= label_encoder.fit_transform(df['Offshore'])
df['Offshore'].unique()
df['Match_light_type']= label_encoder.fit_transform(df['Match_light_type'])
df['Match_light_type'].unique()
df['Match_format']= label_encoder.fit_transform(df['Match_format'])
df['Match_format'].unique()
print(df)

# Write the updated DataFrame back to the file
df.to_csv("file.csv", index=False)
