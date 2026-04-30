import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_tasi_models(processed_file_path):
    if not os.path.exists(processed_file_path):
        print(f"خطأ: {processed_file_path} غير موجود.")
        return
    
    df = pd.read_csv(processed_file_path)
    
    # تحديد الميزات الشاملة (تشمل المشاعر وTradingView)
    features = [
        'close', 'volume', 'sma_20', 'sma_50', 'sma_200', 
        'ema_20', 'rsi', 'macd', 'macd_signal', 'atr',
        'pe_ratio', 'div_yield', 'beta', 'oil_correlation',
        'macro_bz_f', 'macro_tasi_proxy', 'tv_signal', 'tv_adx', 'sentiment'
    ]
    
    # تنظيف الميزات
    for col in ['pe_ratio', 'div_yield', 'beta', 'oil_correlation', 'sentiment']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    features = [f for f in features if f in df.columns]
    
    # 1. تدريب النموذج اليومي (Daily Model)
    target_daily = 'target_next_day_return'
    df_daily = df.dropna(subset=[target_daily]).copy()
    
    print(f"جاري تدريب النموذج اليومي على {len(df_daily)} عينة...")
    model_daily = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model_daily.fit(df_daily[features], df_daily[target_daily])
    
    # 2. تدريب النموذج الأسبوعي (Weekly Model)
    target_weekly = 'target_next_week_return'
    df_weekly = df.dropna(subset=[target_weekly]).copy()
    
    print(f"جاري تدريب النموذج الأسبوعي على {len(df_weekly)} عينة...")
    model_weekly = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model_weekly.fit(df_weekly[features], df_weekly[target_weekly])
    
    # حفظ النماذج
    if not os.path.exists("models"): os.makedirs("models")
    
    joblib.dump(model_daily, "models/tasi_rf_model_daily.joblib")
    joblib.dump(model_weekly, "models/tasi_rf_model_weekly.joblib")
    joblib.dump(features, "models/feature_names.joblib")
    
    print(f"تم حفظ النماذج بنجاح (يومي وأسبوعي).")

if __name__ == "__main__":
    train_tasi_models("data/tasi_processed.csv")
