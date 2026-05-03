import pandas as pd
import joblib
import os

MODEL_PATH = "models/tasi_rf_model_daily.joblib"
DATA_PATH = "data/tasi_processed.csv"
FEATURES_PATH = "models/feature_names.joblib"
FEATURE_MEDIANS_PATH = "models/feature_medians.joblib"

def test():
    if not os.path.exists(DATA_PATH):
        print("Data path not found")
        return
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    medians = joblib.load(FEATURE_MEDIANS_PATH) if os.path.exists(FEATURE_MEDIANS_PATH) else None
    
    print(f"Features in model: {features}")
    print(f"Columns in data: {df.columns.tolist()}")
    
    latest_data = []
    for (symbol, name), group in df.groupby(['symbol', 'company name']):
        print(f"Testing {symbol} - {name}")
        group = group.sort_values('date')
        latest_row = group.iloc[-1:].copy()
        
        X = latest_row[features].replace([float('inf'), float('-inf')], pd.NA)
        if medians is not None:
            X = X.fillna(medians)
        X = X.fillna(0)
        print(f"X shape: {X.shape}")
        preds = model.predict(X)
        print(f"Prediction: {preds}")

if __name__ == "__main__":
    test()
