label_encoder = preprocessing.LabelEncoder()
df['flower_name']= label_encoder.fit_transform(df['flower_name'])
df['flower_name'].unique()