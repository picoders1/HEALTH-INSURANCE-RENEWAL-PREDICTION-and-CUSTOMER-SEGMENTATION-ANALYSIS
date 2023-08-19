# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:39:51 2021

@author: pankaj
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('pckl_model.pkl', 'rb'))

@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/prediction.html')
def pred():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction)
    output = round(prediction[0], 2)
    print(output)
    if (output==1):
        a="Customer Renewing Policy: Yes"
        print(a)
    else:
        a="Customer Renewing Policy : No" 
        print(a)

    return render_template('prediction.html',prediction_text=a)
if __name__ == "__main__":
    app.run(debug=True)