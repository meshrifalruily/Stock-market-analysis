import pandas as pd
import joblib
import os

MODEL_PATH = "models/tasi_rf_model.joblib"
DATA_PATH = "data/tasi_processed.csv"
FEATURES_PATH = "models/feature_names.joblib"

def test():
    if not os.path.exists(DATA_PATH):
        print("Data path not found")
        return
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    
    print(f"Features in model: {features}")
    print(f"Columns in data: {df.columns.tolist()}")
    
    latest_data = []
    for (symbol, name), group in df.groupby(['Symbol', 'Company Name']):
        print(f"Testing {symbol} - {name}")
        group = group.sort_values('Date')
        latest_row = group.iloc[-1:].copy()
        
        X = latest_row[features]
        print(f"X shape: {X.shape}")
        preds = model.predict(X)
        print(f"Prediction: {preds}")

if __name__ == "__main__":
    test()
