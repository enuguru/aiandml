
# KNN Classification for the cricket match win prediction

# step 1 import the modules
import pandas as pd
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")
# step 2 Load the contents of the CSV file into a DataFrame

filename = 'sportsdataone.csv'
headernames = ['Game_number','Result','Avg_team_Age','Match_light_type','Match_format','Bowlers_in_team','Wicket_keeper_in_team','All_rounder_in_team','First_selection','Opponent','Season','Audience_number','Offshore','Max_run_scored_1over','Max_wicket_taken_1over','Extra_bowls_bowled','Min_run_given_1over','Min_run_scored_1over','Max_run_given_1over','extra_bowls_opponent','player_highest_run','Players_scored_zero','player_highest_wicket']
df = read_csv(filename, names=headernames, na_values=["NA"])
df = df.dropna()
# Swap the values of column A and B
#df[['player_highest_wicket', 'Result']] = df[['Result', 'player_highest_wicket']]
print(df)
input("press ENTER to continue: ")

# step 3 perform label encoding of string data

label_encoder = preprocessing.LabelEncoder()
df['Result']= label_encoder.fit_transform(df['Result'])
df['Result'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Season']= label_encoder.fit_transform(df['Season'])
df['Season'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Opponent']= label_encoder.fit_transform(df['Opponent'])
df['Opponent'].unique()

label_encoder = preprocessing.LabelEncoder()
df['First_selection']= label_encoder.fit_transform(df['First_selection'])
df['First_selection'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Offshore']= label_encoder.fit_transform(df['Offshore'])
df['Offshore'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Match_light_type']= label_encoder.fit_transform(df['Match_light_type'])
df['Match_light_type'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Match_format']= label_encoder.fit_transform(df['Match_format'])
df['Match_format'].unique()

print(df)
input("press ENTER to continue: ")

# Write the updated DataFrame back to the file
df.to_csv("file.csv")#, index=False)`

label_encoder = preprocessing.LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        # encode the non-numeric column using LabelEncoder
        df[col] = label_encoder.fit_transform(df[col])


# step 4 seperate the input and output

array = df.values
inputx = array[:,2:22]
outputy = array[:,1]


# step 5 choose the KNN algorithm / model for doing classification

thismodel = KNeighborsClassifier()
print("\nThe model selected is",thismodel)
print("\nThe parameters of the model are\n\n",thismodel.get_params())
#print(thismodel.set_params())


# step 6: training the model
thismodel.fit(inputx,outputy)


# step 7: testing and model prediction

# step 7a: perform label encoding of string test data
filename = 'newtestoriginal.csv'
headernames = ['Game_number','Result','Avg_team_Age','Match_light_type','Match_format','Bowlers_in_team','Wicket_keeper_in_team','All_rounder_in_team','First_selection','Opponent','Season','Audience_number','Offshore','Max_run_scored_1over','Max_wicket_taken_1over','Extra_bowls_bowled','Min_run_given_1over','Min_run_scored_1over','Max_run_given_1over','extra_bowls_opponent','player_highest_run','Players_scored_zero','player_highest_wicket']
df = read_csv(filename, names=headernames, na_values=["NA"])
df = df.dropna()
# Swap the values of column A and B
#df[['player_highest_wicket', 'Result']] = df[['Result', 'player_highest_wicket']]
print(df)
input("press ENTER to continue: ")

label_encoder = preprocessing.LabelEncoder()
df['Result']= label_encoder.fit_transform(df['Result'])
df['Result'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Season']= label_encoder.fit_transform(df['Season'])
df['Season'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Opponent']= label_encoder.fit_transform(df['Opponent'])
df['Opponent'].unique()

label_encoder = preprocessing.LabelEncoder()
df['First_selection']= label_encoder.fit_transform(df['First_selection'])
df['First_selection'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Offshore']= label_encoder.fit_transform(df['Offshore'])
df['Offshore'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Match_light_type']= label_encoder.fit_transform(df['Match_light_type'])
df['Match_light_type'].unique()

label_encoder = preprocessing.LabelEncoder()
df['Match_format']= label_encoder.fit_transform(df['Match_format'])
df['Match_format'].unique()

print(df)
input("press ENTER to continue: ")

# Write the updated DataFrame back to the file
#df.to_csv("file.csv")#, index=False)`

label_encoder = preprocessing.LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        # encode the non-numeric column using LabelEncoder
        df[col] = label_encoder.fit_transform(df[col])


# Write the first 200 rows to a csv file
df.to_csv("newtest.csv", index=False)
filename = 'newtest.csv'
headernames = ['Game_number','Result','Avg_team_Age','Match_light_type','Match_format','Bowlers_in_team','Wicket_keeper_in_team','All_rounder_in_team','First_selection','Opponent','Season','Audience_number','Offshore','Max_run_scored_1over','Max_wicket_taken_1over','Extra_bowls_bowled','Min_run_given_1over','Min_run_scored_1over','Max_run_given_1over','extra_bowls_opponent','player_highest_run','Players_scored_zero','player_highest_wicket']
newdataframe = read_csv(filename)#, names=headernames)
#newdataframe = newdataframe.dropna()
array = newdataframe.values
testinputz = array[:,2:22]
print("\n\nThe test inputs are\n\n",newdataframe)
input("press ENTER to continue: ")
res=thismodel.predict(testinputz)
print(res)
input("press ENTER to continue: ")

# step 5: visualizing the test results
reslist=[]
for val in res:
    if val==0:
        reslist.append("We Lose")
    else:
        reslist.append("We Win")
print("\nThe test results are\n\n",reslist)
#res=thismodel.predict(testinputz)
#print(thismodel.predict(testinputz))