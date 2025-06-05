from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS

# Load the trained model 
model = load('stroke_prediction_model.joblib')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        df = request.json
        print("Received JSON:", df)

        data = pd.DataFrame([df])  # wrap in list to ensure one row
        prediction = model.predict(data)[0]
        print(f"Prediction: {prediction}")
        return jsonify({'prediction': int(prediction)}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def home():
    return "Welcome to the Stroke Prediction API!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
# The app will run on port 5000 and can be accessed from any IP address.
    print("Server is running on http://localhost:5000")
    print("Use the /predict endpoint to make predictions.")
    