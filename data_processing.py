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
    """Kreira direktorij za modele ako ne postoji"""
    if not os.path.exists('models'):
        os.makedirs('models')

def load_and_process_data():
    """Učitava i procesira podatke"""
    print("Učitavanje podataka...")
    df = pd.read_pickle(r"C:\Users\amers\OneDrive\Desktop\belma\data\processed_data.pkl")
    
    # Podjela podataka
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Skaliranje Amount i Time
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    if 'Amount_scaled' not in X.columns:
        X['Amount_scaled'] = scaler_amount.fit_transform(X[['Amount']])
        X['Time_scaled'] = scaler_time.fit_transform(X[['Time']])
        X = X.drop(['Amount', 'Time'], axis=1)
    
    # Sačuvaj skalere
    joblib.dump(scaler_amount, 'models/scaler_amount.pkl')
    joblib.dump(scaler_time, 'models/scaler_time.pkl')
    
    # Podjela na train i test skup
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Primjena SMOTE-a
    print("Primjena SMOTE-a...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trenira i evaluira modele"""
    results = {}
    
    # Random Forest
    print("Treniranje Random Forest modela...")
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest", results)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    
    # XGBoost
    print("Treniranje XGBoost modela...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                             scale_pos_weight=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost", results)
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    
    # Logistička regresija
    print("Treniranje Logističke regresije...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression", results)
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    
    return results

def evaluate_model(model, X_test, y_test, model_name, results_dict):
    """Evaluira model i sprema rezultate"""
    print(f"Evaluacija {model_name} modela...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrike
    ap_score = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    # Precision-Recall kriva
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Spremanje rezultata
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
    
    # Kreiranje grafika
    plot_precision_recall(recall, precision, ap_score, model_name)
    plot_precision_recall_vs_threshold(precision, recall, thresholds, model_name)
    
    print(f"{model_name} - Average Precision Score: {ap_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")

def plot_precision_recall(recall, precision, ap_score, model_name):
    """Kreira Precision-Recall krivu"""
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
    """Kreira grafik Precision-Recall vs Threshold"""
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
    """Sprema rezultate u CSV fajl"""
    metrics_df = pd.DataFrame()
    
    for model_name, metrics in results.items():
        # Osnovne metrike
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
    print("Rezultati sačuvani u models/model_metrics.csv")

def main():
    """Glavna funkcija za pokretanje procesa"""
    create_models_directory()
    X_train, X_test, y_train, y_test = load_and_process_data()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_results_to_csv(results)
    print("Procesiranje podataka i treniranje modela završeno!")

if __name__ == "__main__":
    main()