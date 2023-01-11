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
	if request.method == "POST":
		creditscore = int(request.form['creditscore']) 
		age = int(request.form['age'])
		tenure = int(request.form['tenure'])
		balance = int(request.form['balance'])
		no_of_products = int(request.form['no_of_products'])
		hascrcard = int(request.form['hascrcard'])
		isactivemember = int(request.form['isactivemember'])
		estimatedsalary = int(request.form['estimatedsalary'])
		geography_germany = int(request.form['geography_germany'])
		geography_spain = int(request.form['geography_spain'])
		gender_male = int(request.form['gender_male'])
		data = np.array([[creditscore, age, tenure, balance, no_of_products, hascrcard, isactivemember, estimatedsalary, geography_germany, geography_spain,  gender_male]])
		data_scaled = scaler.transform(data)
		print(data_scaled.shape)
		final_data = data_scaled.reshape(1,11,1)
		my_prediction = model.predict(final_data)
		return render_template('result.html', prediction=my_prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
