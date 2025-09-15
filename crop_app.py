from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('crop_app.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input values from the form
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorus'])
            K = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Create a numpy array from the inputs
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Make the prediction (directly gives crop name, e.g., "rice")
            prediction = model.predict(features)
            result = prediction[0]   # take the string result directly

            # Pass the prediction to the HTML template
            return render_template(
                'index.html',
                prediction_text=f'The recommended crop is {result}.'
            )

        except Exception as e:
            return render_template(
                'index.html',
                prediction_text=f'Error: {str(e)}'
            )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
