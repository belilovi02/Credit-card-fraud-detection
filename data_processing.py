# data_processing.py
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                           average_precision_score, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os

def create_models_directory():
    """Creates models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')

def load_and_process_data():
    """Loads and processes the data"""
    print("Loading data...")
    # Load from CSV instead of pickle to avoid version conflicts
    df = pd.read_csv(r"C:\Users\amers\OneDrive\Desktop\belma\data\creditcard.csv")
    
    # Data processing
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Amount and Time
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    X['Amount_scaled'] = scaler_amount.fit_transform(X[['Amount']])
    X['Time_scaled'] = scaler_time.fit_transform(X[['Time']])
    X = X.drop(['Amount', 'Time'], axis=1)
    
    # Save scalers
    joblib.dump(scaler_amount, 'models/scaler_amount.pkl')
    joblib.dump(scaler_time, 'models/scaler_time.pkl')
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Save processed data
    processed_df = pd.concat([X, y], axis=1)
    processed_df.to_pickle(r"C:\Users\amers\OneDrive\Desktop\belma\data\processed_data.pkl")
    
    return X_train_resampled, X_test, y_train_resampled, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains and evaluates models"""
    results = {}
    
    # Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest", results)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    
    # XGBoost
    print("Training XGBoost model...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                             scale_pos_weight=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost", results)
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression", results)
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    
    return results

def evaluate_model(model, X_test, y_test, model_name, results_dict):
    """Evaluates model and stores results"""
    print(f"Evaluating {model_name} model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    ap_score = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Store results
    results_dict[model_name] = {
        'average_precision': ap_score,
        'confusion_matrix': cm,
        'classification_report': cr,
        'precision_recall_curve': {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    }
    
    # Create plots
    plot_precision_recall(recall, precision, ap_score, model_name)
    plot_precision_recall_vs_threshold(precision, recall, thresholds, model_name)
    
    print(f"{model_name} - Average Precision Score: {ap_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")

def plot_precision_recall(recall, precision, ap_score, model_name):
    """Creates Precision-Recall curve plot"""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {ap_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_pr_curve.png')
    plt.close()

def plot_precision_recall_vs_threshold(precision, recall, thresholds, model_name):
    """Creates Precision-Recall vs Threshold plot"""
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'{model_name} - Precision-Recall vs Threshold')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_pr_threshold.png')
    plt.close()

def save_results_to_csv(results):
    """Saves results to CSV file"""
    metrics_df = pd.DataFrame()
    
    for model_name, metrics in results.items():
        model_metrics = {
            'Model': model_name,
            'Average Precision': metrics['average_precision'],
            'Accuracy': metrics['classification_report']['accuracy'],
            'Precision (Class 0)': metrics['classification_report']['0']['precision'],
            'Recall (Class 0)': metrics['classification_report']['0']['recall'],
            'F1 (Class 0)': metrics['classification_report']['0']['f1-score'],
            'Precision (Class 1)': metrics['classification_report']['1']['precision'],
            'Recall (Class 1)': metrics['classification_report']['1']['recall'],
            'F1 (Class 1)': metrics['classification_report']['1']['f1-score']
        }
        
        metrics_df = metrics_df.append(model_metrics, ignore_index=True)
    
    metrics_df.to_csv('models/model_metrics.csv', index=False)
    print("Results saved to models/model_metrics.csv")

def main():
    """Main function to run the process"""
    create_models_directory()
    X_train, X_test, y_train, y_test = load_and_process_data()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_results_to_csv(results)
    print("Data processing and model training completed!")

if __name__ == "__main__":
    main()