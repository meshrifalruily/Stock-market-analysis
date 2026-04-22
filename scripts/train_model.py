import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_tasi_model(processed_file_path, model_output_path):
    if not os.path.exists(processed_file_path):
        print(f"Error: {processed_file_path} not found.")
        return
    
    df = pd.read_csv(processed_file_path)
    
    # Define features and target
    features = [
        'Close', 'Volume', 'SMA_20', 'SMA_50', 'SMA_200', 
        'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
        'BB_Upper', 'BB_Lower', 'ADX', 'Support', 'Resistance',
        'Daily_Return', 'Weekly_Return', 'Monthly_Return'
    ]
    target = 'Target_Next_Day_Return'
    
    X = df[features]
    y = df[target]
    
    # Split data (time-based split)
    split_idx = int(len(df) * 0.8)
    df = df.sort_values('Date')
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Random Forest model on {len(X_train)} samples...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse:.6f}")
    
    # Save model and feature list
    if not os.path.exists("models"):
        os.makedirs("models")
    
    joblib.dump(model, model_output_path)
    joblib.dump(features, "models/feature_names.joblib")
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train_tasi_model("data/tasi_processed.csv", "models/tasi_rf_model.joblib")
