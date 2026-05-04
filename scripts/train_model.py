import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib
import os
import json

MODEL_DIR = "models"
MIN_AVG_TRADED_VALUE = float(os.getenv("TASI_MIN_AVG_TRADED_VALUE", "1000000"))

def get_candidate_models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, l2_regularization=0.1, random_state=42
        ),
    }

def chronological_split(df, validation_ratio=0.2):
    dates = np.array(sorted(pd.to_datetime(df['date']).unique()))
    if len(dates) < 10:
        return df.iloc[:0], df
    split_idx = max(1, int(len(dates) * (1 - validation_ratio)))
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]
    return df[df['date'].isin(train_dates)].copy(), df[df['date'].isin(val_dates)].copy()

def prepare_frame(df, features, target, medians=None):
    clean = df.dropna(subset=[target]).copy()
    X = clean[features].replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = X.median(numeric_only=True).fillna(0)
    X = X.fillna(medians).fillna(0)
    y = clean[target]
    return clean, X, y, medians

def evaluate_model(model, val_df, X_val, y_val):
    preds = model.predict(X_val)
    metrics = {
        "mae": float(mean_absolute_error(y_val, preds)),
        "rmse": float(root_mean_squared_error(y_val, preds)),
        "direction_accuracy": float((np.sign(preds) == np.sign(y_val)).mean()),
    }

    ranked = val_df[['date', 'symbol']].copy()
    ranked['pred_return'] = preds
    ranked['actual_return'] = y_val.to_numpy()
    if 'avg_traded_value_20d' in val_df.columns:
        ranked['is_liquid'] = val_df['avg_traded_value_20d'].to_numpy() >= MIN_AVG_TRADED_VALUE
        ranked = ranked[ranked['is_liquid']]
    top_3_returns = ranked.sort_values(['date', 'pred_return'], ascending=[True, False]).groupby('date').head(3)
    metrics["top3_avg_actual_return"] = float(top_3_returns['actual_return'].mean())
    metrics["score"] = metrics["top3_avg_actual_return"] - metrics["mae"]
    return metrics

def train_tasi_models(processed_file_path):
    if not os.path.exists(processed_file_path):
        print(f"خطأ: {processed_file_path} غير موجود.")
        return
    
    df = pd.read_csv(processed_file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # تحديد الميزات الشاملة (تشمل المشاعر وTradingView)
    features = [
        'close', 'volume', 'sma_20', 'sma_50', 'sma_200', 
        'ema_20', 'rsi', 'macd', 'macd_signal', 'atr',
        'pe_ratio', 'div_yield', 'beta', 'oil_correlation',
        'macro_bz_f', 'macro_tasi_proxy', 'tv_signal', 'tv_adx', 'sentiment',
        'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'weekly_return_hist', 'volatility_10d', 'volatility_20d', 'traded_value',
        'avg_traded_value_20d', 'volume_ratio_20d', 'volume_ratio_5d',
        'momentum_20d', 'momentum_10d',
        'atr_pct', 'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_sma_200',
        'market_return', 'relative_return_1d', 'relative_return_5d',
        'oil_return', 'day_of_week', 'month', 'market_breadth_sma20',
        'market_breadth_sma50', 'sector_return_1d', 'sector_relative_return_1d',
        'sector_momentum_20d'
    ]
    
    # تنظيف الميزات
    for col in ['pe_ratio', 'div_yield', 'beta', 'oil_correlation', 'sentiment']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    features = [f for f in features if f in df.columns]
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_metrics = {}
    feature_medians = df[features].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0)

    def train_one_target(target, output_path):
        target_df = df.dropna(subset=[target]).sort_values('date').copy()
        train_df, val_df = chronological_split(target_df)
        if train_df.empty or val_df.empty:
            print(f"تحذير: بيانات تحقق غير كافية لـ {target}، سيتم تدريب RandomForest فقط.")
            model = get_candidate_models()["random_forest"]
            _, X_all, y_all, _ = prepare_frame(target_df, features, target, feature_medians)
            model.fit(X_all, y_all)
            joblib.dump(model, output_path)
            return {"selected_model": "random_forest", "warning": "insufficient_validation_data"}

        _, X_train, y_train, medians = prepare_frame(train_df, features, target)
        val_clean, X_val, y_val, _ = prepare_frame(val_df, features, target, medians)
        candidates = get_candidate_models()
        results = {}
        best_name = None
        best_score = -np.inf

        for name, candidate in candidates.items():
            model = clone(candidate)
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, val_clean, X_val, y_val)
            results[name] = metrics
            print(
                f"{target} | {name}: MAE={metrics['mae']:.5f}, "
                f"Top3={metrics['top3_avg_actual_return']:.5f}, Dir={metrics['direction_accuracy']:.2%}"
            )
            if metrics["score"] > best_score:
                best_name = name
                best_score = metrics["score"]

        final_model = clone(candidates[best_name])
        _, X_all, y_all, _ = prepare_frame(target_df, features, target, feature_medians)
        final_model.fit(X_all, y_all)
        joblib.dump(final_model, output_path)
        return {"selected_model": best_name, "validation": results}
    
    # 1. تدريب النموذج اليومي (Daily Model)
    target_daily = 'target_next_day_return'
    df_daily = df.dropna(subset=[target_daily]).copy()
    
    print(f"جاري تدريب النموذج اليومي على {len(df_daily)} عينة...")
    model_metrics["daily"] = train_one_target(target_daily, os.path.join(MODEL_DIR, "tasi_rf_model_daily.joblib"))
    
    # 2. تدريب النموذج الأسبوعي (Weekly Model)
    target_weekly = 'target_next_week_return'
    df_weekly = df.dropna(subset=[target_weekly]).copy()
    
    print(f"جاري تدريب النموذج الأسبوعي على {len(df_weekly)} عينة...")
    model_metrics["weekly"] = train_one_target(target_weekly, os.path.join(MODEL_DIR, "tasi_rf_model_weekly.joblib"))
    
    # حفظ النماذج
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(feature_medians, os.path.join(MODEL_DIR, "feature_medians.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"تم حفظ النماذج بنجاح (يومي وأسبوعي).")

if __name__ == "__main__":
    train_tasi_models("data/tasi_processed.csv")
