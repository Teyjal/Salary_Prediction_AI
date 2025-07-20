from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('salary_prediction_model_compressed.pkl')
encoders = joblib.load('C:/Users/TEYJAL SRI/Desktop/salary-prediction/encoders (1).pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form

        print("Received form data:", input_data)  # For debugging

        # Define fields and their processing method
        fields = [
            ('age', int),
            ('workclass', int),

            ('fnlwgt', int),
            ('education', lambda x: encoders['education'].transform([x])[0]),
            ('educational-num', int),
            ('marital-status', lambda x: encoders['marital-status'].transform([x])[0]),
            ('occupation', lambda x: encoders['occupation'].transform([x])[0]),
            ('relationship', lambda x: encoders['relationship'].transform([x])[0]),
            ('race', lambda x: encoders['race'].transform([x])[0]),
            ('gender', lambda x: encoders['gender'].transform([x])[0]),
            ('capital-gain', int),
            ('capital-loss', int),
            ('hours-per-week', int),
            ('native-country', lambda x: encoders['native-country'].transform([x])[0])
        ]

        # Process each field safely
        features = []
        for field_name, processor in fields:
            try:
                raw_value = input_data[field_name]
                processed_value = processor(raw_value)
                features.append(processed_value)
            except Exception as e:
                return jsonify({"error": f"Error in field '{field_name}': {str(e)}"})

        # Predict
        prediction = model.predict([features])[0]
        result = '>50K' if prediction == 1 else '<=50K'

        return render_template('index.html', prediction_text=f"Predicted Salary Class: {result}")

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
