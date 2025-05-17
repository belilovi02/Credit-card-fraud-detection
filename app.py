from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Omogući CORS za sve rute

# Učitavanje modela i skalera
models = {
    'random_forest': joblib.load('models/random_forest_model.pkl'),
    'xgboost': joblib.load('models/xgboost_model.pkl'),
    'logistic_regression': joblib.load('models/logistic_regression_model.pkl')
}

# Učitavanje skalera
scaler_amount = joblib.load('models/scaler_amount.pkl')
scaler_time = joblib.load('models/scaler_time.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Priprema podataka za predikciju
        transaction_data = {
            'Time': data.get('time', 0),
            'Amount': data.get('amount', 0)
        }
        
        # Dodaj PCA karakteristike
        for i in range(1, 29):
            transaction_data[f'V{i}'] = data.get(f'v{i}', 0)
        
        # Kreiraj DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Skaliranje
        if 'Amount' in df.columns:
            df['Amount_scaled'] = scaler_amount.transform(df[['Amount']])
            df['Time_scaled'] = scaler_time.transform(df[['Time']])
            df = df.drop(['Amount', 'Time'], axis=1)
        
        # Odabir modela
        model_type = data.get('model_type', 'random_forest')
        model = models.get(model_type, models['random_forest'])
        
        # Predikcija
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1][0]
        
        return jsonify({
            'success': True,
            'is_fraud': bool(prediction[0]),
            'probability': float(probability),
            'model_type': model_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model_metrics', methods=['GET'])
def get_model_metrics():
    try:
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

if __name__ == '__main__':
    if not os.path.exists('models'):
        print("Error: Models directory not found. Please run data_processing.py first.")
    else:
        app.run(debug=True, port=5000)