import pandas as pd
import joblib
import os

MODEL_PATH = "models/tasi_reg_model_next_day.joblib"
DATA_PATH = "data/tasi_processed.csv"
FEATURES_PATH = "models/regression_feature_names.joblib"
FEATURE_MEDIANS_PATH = "models/regression_feature_medians.joblib"

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
        current_price = latest_row['close'].iloc[0]
        expected_return = max(min(preds[0], 0.2), -0.2)
        next_close = current_price * (1 + expected_return)
        print(f"Next close prediction: {next_close:.2f} | expected return: {expected_return:.2%}")

if __name__ == "__main__":
    test()
