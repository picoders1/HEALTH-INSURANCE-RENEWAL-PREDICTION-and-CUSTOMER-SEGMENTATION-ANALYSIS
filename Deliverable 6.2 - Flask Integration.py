
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('predict_model.pkl', 'rb'))

@app.route('/')
@app.route('/new_index.html')
def home():
    return render_template('new_index.html')

@app.route('/prediction.html')
def pred():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    global final_features
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction)
    if (prediction==0):
        a="Chances of Customer Renewing Policy: Yes"
        print(a)
    else:
        a="Chances of Customer Renewing Policy : No" 
        print(a)

    return render_template('prediction.html',prediction_text=a)
if __name__ == "__main__":
    app.run(debug=False)


#final_features.to_csv(r"C:\Users\sprav\Desktop\deployment\deployment\final_features.csv")
