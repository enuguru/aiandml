
# deploying the ML model as a web service using Flask web framework

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from flask import Flask

df = pd.read_csv('titanic.csv')

x = df[df.columns.difference(['Survived'])
y = df['Survived']
       
classifier = RandomForestClassifier()
classifier.fit(x, y)

from sklearn.externals import joblib

joblib.dump(classifier, 'classifier.pkl')


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)

     classifier = joblib.load('classifier.pkl')
     prediction = classifier.predict(query)
     return jsonify({'prediction': list(prediction)})


app.run(port=8080)
