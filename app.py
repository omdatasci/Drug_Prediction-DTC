import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import pickle

from flask import Flask, render_template, request
app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    Age = float(request.form['age'])
    Sex = request.form['sex']
    BP = request.form['bp']
    Cholesterol = request.form['cholesterol']
    Na_to_k = float(request.form['na_to_k'])

    # Preprocess input data
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['M', 'F'])
    sex_encoded = le_sex.transform([Sex])[0]
        
    le_bp = preprocessing.LabelEncoder()
    le_bp.fit(['HIGH', 'NORMAL'])
    bp_encoded = le_bp.transform([BP])[0]

    le_chol = preprocessing.LabelEncoder()
    le_chol.fit(['HIGH', 'NORMAL'])
    chol_encoded = le_chol.transform([Cholesterol])[0]

    input_data = [[Age, sex_encoded, bp_encoded, chol_encoded, Na_to_k]]
        
    # Make predictions using the loaded model
    predicted_drug = model.predict(input_data)

    return render_template('result.html', drug=predicted_drug[0])



if(__name__)=='__main__':
    app.run(debug=True)

