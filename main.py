from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Global variables to store trained models
diabetes_predictor = None
heart_predictor = None
diabetes_trained = False
heart_trained = False


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For other numpy types
        return obj.item()
    else:
        return obj


def initialize_models():
    """Initialize and train the ML models"""
    global diabetes_predictor, heart_predictor, diabetes_trained, heart_trained

    try:
        # Initialize Diabetes Predictor
        from diabetes_predictor import DiabetesPredictor
        diabetes_predictor = DiabetesPredictor()

        # Load and train diabetes model
        diabetes_data = diabetes_predictor.load_and_preprocess_data('datasets/diabetes.csv')
        if diabetes_data is not None:
            diabetes_trained = diabetes_predictor.train_model(diabetes_data)
            if diabetes_trained:
                print("✅ Diabetes model trained successfully!")
            else:
                print("❌ Failed to train diabetes model")

        # Initialize Heart Disease Predictor
        from heart_predictor import HeartDiseasePredictor
        heart_predictor = HeartDiseasePredictor()

        # Load and train heart disease model
        heart_data = heart_predictor.load_and_preprocess_data('datasets/heart.csv')
        if heart_data is not None:
            heart_trained = heart_predictor.train_model(heart_data)
            if heart_trained:
                print("✅ Heart disease model trained successfully!")
            else:
                print("❌ Failed to train heart disease model")

    except Exception as e:
        print(f"❌ Error initializing models: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        if not diabetes_trained or diabetes_predictor is None:
            return jsonify({
                'error': 'Diabetes model not trained yet. Please try again later.'
            }), 500

        # Get data from request
        data = request.json
        print(f"📊 Received diabetes data: {data}")

        # Convert to features array in correct order
        features = [
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('blood_pressure', 0)),
            float(data.get('skin_thickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('dpf', 0)),
            float(data.get('age', 0))
        ]

        print(f"🔧 Features for prediction: {features}")

        # Make prediction
        result = diabetes_predictor.predict_diabetes(features)

        if result:
            # Convert numpy types to native Python types for JSON serialization
            serializable_result = {
                'prediction': convert_to_serializable(result['prediction']),
                'probability': convert_to_serializable(result['probability']),
                'risk_level': result['risk_level']
            }

            # Add recommendation based on risk level
            if serializable_result['risk_level'] == 'High Risk':
                recommendation = "Please consult a doctor immediately for proper diagnosis and treatment."
            elif serializable_result['risk_level'] == 'Medium Risk':
                recommendation = "Monitor your health regularly and consider consulting a healthcare professional."
            else:
                recommendation = "Maintain a healthy lifestyle with regular exercise and balanced diet."

            serializable_result['recommendation'] = recommendation

            print(f"🎯 Diabetes prediction result: {serializable_result}")
            return jsonify(serializable_result)
        else:
            return jsonify({
                'error': 'Prediction failed. Please check your input values.'
            }), 500

    except Exception as e:
        print(f"❌ Error in diabetes prediction: {e}")
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    try:
        if not heart_trained or heart_predictor is None:
            return jsonify({
                'error': 'Heart disease model not trained yet. Please try again later.'
            }), 500

        # Get data from request
        data = request.json
        print(f"📊 Received heart data: {data}")

        # Convert to features array in correct order for heart dataset
        features = [
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('cp', 0)),
            float(data.get('trestbps', 0)),
            float(data.get('chol', 0)),
            float(data.get('fbs', 0)),
            float(data.get('restecg', 0)),
            float(data.get('thalach', 0)),
            float(data.get('exang', 0)),
            float(data.get('oldpeak', 0)),
            float(data.get('slope', 0)),
            float(data.get('ca', 0)),
            float(data.get('thal', 0))
        ]

        print(f"🔧 Features for prediction: {features}")

        # Make prediction
        result = heart_predictor.predict_heart_disease(features)

        if result:
            # Convert numpy types to native Python types for JSON serialization
            serializable_result = {
                'prediction': convert_to_serializable(result['prediction']),
                'probability': convert_to_serializable(result['probability']),
                'risk_level': result['risk_level']
            }

            # Add recommendation based on risk level
            if serializable_result['risk_level'] == 'High Risk':
                recommendation = "Urgent: Please consult a cardiologist immediately for comprehensive evaluation."
            elif serializable_result['risk_level'] == 'Medium Risk':
                recommendation = "Schedule a check-up with your doctor and maintain heart-healthy habits."
            else:
                recommendation = "Continue with regular exercise and heart-healthy diet for maintenance."

            serializable_result['recommendation'] = recommendation

            print(f"🎯 Heart prediction result: {serializable_result}")
            return jsonify(serializable_result)
        else:
            return jsonify({
                'error': 'Prediction failed. Please check your input values.'
            }), 500

    except Exception as e:
        print(f"❌ Error in heart prediction: {e}")
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/model_status')
def model_status():
    """Check the status of ML models"""
    return jsonify({
        'diabetes_trained': diabetes_trained,
        'heart_trained': heart_trained,
        'diabetes_ready': diabetes_trained and diabetes_predictor is not None,
        'heart_ready': heart_trained and heart_predictor is not None
    })


@app.route('/sample_data')
def sample_data():
    """Provide sample data for testing forms"""
    return jsonify({
        'diabetes': {
            'pregnancies': 2,
            'glucose': 120,
            'blood_pressure': 70,
            'skin_thickness': 20,
            'insulin': 79,
            'bmi': 25.0,
            'dpf': 0.5,
            'age': 30
        },
        'heart': {
            'age': 52,
            'sex': 1,
            'cp': 0,
            'trestbps': 125,
            'chol': 212,
            'fbs': 0,
            'restecg': 1,
            'thalach': 168,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 2,
            'ca': 2,
            'thal': 3
        }
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🏥 Medical Prediction System Starting...")
    print("=" * 50)

    # Initialize and train models
    print("\n🔧 Initializing Machine Learning Models...")
    initialize_models()

    print("\n🌐 Starting Flask Server...")
    print("📍 Server running at: http://127.0.0.1:5000")
    print("📊 Model Status:")
    print(f"   - Diabetes Predictor: {'✅ Ready' if diabetes_trained else '❌ Not Ready'}")
    print(f"   - Heart Disease Predictor: {'✅ Ready' if heart_trained else '❌ Not Ready'}")
    print("\n⚡ Use Ctrl+C to stop the server")
    print("=" * 50)

    app.run(debug=True, host='127.0.0.1', port=5000)