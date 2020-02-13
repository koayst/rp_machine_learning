from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

import numpy as np
import os
import pickle

app = Flask(__name__, template_folder='.')

#load the model
print('Load model.\n')
model_filename='pima-indians-xgboost.pkl'
filedir=os.path.join(os.getcwd(), '..\\model')
filepath = os.path.join(filedir, model_filename)
loaded_model = pickle.load(open(filepath, 'rb'))
print(filepath)

#load the scaler
print('Load scaler.\n')
scaler_filename = 'pima-indians-scaler.pkl'
filedir = os.path.join(os.getcwd(), '..\\model')
filepath = os.path.join(filedir, scaler_filename)
loaded_scaler = pickle.load(open(filepath, 'rb'))
print(filepath)

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    glucose = np.int64(request.form.get('glucose'))
    bmi = np.float64(request.form.get('bmi'))
    age = np.int64(request.form.get('age'))
    pregnancies = np.int64(request.form.get('pregnancies'))

    print('-------------')
    print(glucose, bmi, age, pregnancies)
    print('-------------')

    data = [[glucose, bmi, age, pregnancies]]
    data = loaded_scaler.transform(data)

    response_text = ""
    prediction = loaded_model.predict(data)
    if prediction[0] == 1:
        response_text = 'Diabetes risk is High'
    else:
        response_text = 'Diabetes risk is Low'

    confidence = loaded_model.predict_proba(data)
    response_text = response_text + ' - ' + \
                    'Confidence: ' + str(round(confidence[0][prediction[0]] * 100, 2)) + '%'

    return render_template('index.html', prediction_text = response_text)

if __name__ == "__main__":
    app.run(debug=True, port=6800)