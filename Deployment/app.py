from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import pickle
import tensorflow

# Load the scaler and model
filename = 'scaler.pkl'
scaler = pickle.load(open(filename, 'rb'))

model = tensorflow.keras.models.load_model('churn_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():
	int_features = [int(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	my_prediction = model.predict(final_features)
	return render_template('result.html', prediction=my_prediction)

# Run the app
if __name__ == "__main__":
    app.run()




