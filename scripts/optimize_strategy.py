import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/tasi_processed.csv')
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

features = joblib.load('models/feature_names.joblib')
target = 'target_next_week_return'

clean_df = df.dropna(subset=[target]).copy()
X = clean_df[features].replace([np.inf, -np.inf], np.nan)
medians = X.median(numeric_only=True).fillna(0)
X = X.fillna(medians).fillna(0)
y = clean_df[target]

# Train test split on time
dates = np.sort(clean_df['date'].unique())
split_idx = int(len(dates) * 0.8)
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_mask = clean_df['date'].isin(train_dates)
test_mask = clean_df['date'].isin(test_dates)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

models = {
    "HistGB_0.05_63_0.1": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=0.1, random_state=42),
    "HistGB_0.03_31_0.2": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=0.2, random_state=42),
    "RF_100_8": RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "RF_300_10": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "ET_100_8": ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "ET_300_10": ExtraTreesRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
}

best_score = float('-inf')
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    test_res = clean_df[test_mask].copy()
    test_res['pred'] = preds
    
    top3 = test_res.sort_values(['date', 'pred'], ascending=[True, False]).groupby('date').head(3)
    top3_ret = top3['actual_return'] if 'actual_return' in top3 else top3[target]
    top3_mean = top3_ret.mean()
    
    print(f"Model: {name} | MAE: {mae:.5f} | Top3 Mean Ret: {top3_mean:.5f}")
    if top3_mean > best_score:
        best_score = top3_mean
        best_name = name

print(f"Best model: {best_name} with score {best_score}")
