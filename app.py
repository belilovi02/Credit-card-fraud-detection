from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check package versions
try:
    import sklearn
    import xgboost
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"xgboost version: {xgboost.__version__}")
except ImportError as e:
    print(f"Import error: {e}")

# Load models and scalers with version checking
models = {}
scalers = {}

def load_models():
    """Safely load models with version checking"""
    try:
        models['random_forest'] = joblib.load('models/random_forest_model.pkl')
        models['xgboost'] = joblib.load('models/xgboost_model.pkl')
        models['logistic_regression'] = joblib.load('models/logistic_regression_model.pkl')
        
        scalers['amount'] = joblib.load('models/scaler_amount.pkl')
        scalers['time'] = joblib.load('models/scaler_time.pkl')
        
        print("All models and scalers loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        traceback.print_exc()
        return False

# Initialize models at startup
if not load_models():
    print("Failed to load models. Please check model files and versions.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input validation
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.json
        
        # Required fields check
        if 'amount' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: amount'
            }), 400
        
        # Prepare transaction data with default values
        transaction_data = {
            'Time': data.get('time', 0),
            'Amount': float(data.get('amount', 0))  # Ensure numeric
        }
        
        # Add PCA features with validation
        for i in range(1, 29):
            try:
                transaction_data[f'V{i}'] = float(data.get(f'v{i}', 0))
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'Invalid value for V{i}'
                }), 400
        
        # Create DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Feature scaling
        try:
            if 'Amount' in df.columns:
                df['Amount_scaled'] = scalers['amount'].transform(df[['Amount']])
                df['Time_scaled'] = scalers['time'].transform(df[['Time']])
                df = df.drop(['Amount', 'Time'], axis=1)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Feature scaling failed: {str(e)}'
            }), 500
        
        # Model selection
        model_type = data.get('model_type', 'random_forest')
        if model_type not in models:
            return jsonify({
                'success': False,
                'error': f'Invalid model type: {model_type}'
            }), 400
        
        model = models[model_type]
        
        # Prediction
        try:
            prediction = model.predict(df)
            probability = model.predict_proba(df)[:, 1][0]
            
            return jsonify({
                'success': True,
                'is_fraud': bool(prediction[0]),
                'probability': float(probability),
                'model_type': model_type,
                'features_used': list(df.columns)  # Debug info
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500

@app.route('/model_metrics', methods=['GET'])
def get_model_metrics():
    try:
        if not os.path.exists('models/model_metrics.csv'):
            return jsonify({
                'success': False,
                'error': 'Metrics file not found'
            }), 404
            
        metrics = pd.read_csv('models/model_metrics.csv').to_dict('records')
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'scalers_loaded': len(scalers) > 0
    })

if __name__ == '__main__':
    if not os.path.exists('models'):
        print("Error: Models directory not found. Please run data_processing.py first.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)