import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_tasi_model(processed_file_path, model_output_path):
    if not os.path.exists(processed_file_path):
        print(f"خطأ: {processed_file_path} غير موجود.")
        return
    
    df = pd.read_csv(processed_file_path)
    
    # تحديد الميزات والهدف
    features = [
        'close', 'volume', 'sma_20', 'sma_50', 'sma_200', 
        'ema_20', 'rsi', 'macd', 'macd_signal', 'atr',
        'pe_ratio', 'div_yield', 'beta', 'oil_correlation',
        'macro_bz_f', 'macro_tasi_proxy', 'tv_signal', 'tv_adx'
    ]
    
    # معالجة القيم المفقودة في البيانات الأساسية (التعبئة بالوسيط)
    for col in ['pe_ratio', 'div_yield', 'beta', 'oil_correlation']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    target = 'target_next_day_return'
    
    # تصفية الميزات المتاحة فقط
    features = [f for f in features if f in df.columns]
    
    # حذف الصفوف التي ليس لها هدف (غالباً الصف الأخير لكل شركة) قبل التدريب
    df_train = df.dropna(subset=[target]).copy()
    
    X = df_train[features]
    y = df_train[target]
    
    # تقسيم البيانات بناءً على الوقت (80% تدريب، 20% اختبار)
    split_idx = int(len(df_train) * 0.8)
    df_train = df_train.sort_values('date')
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"جاري تدريب نموذج الغابة العشوائية على {len(X_train)} عينة مع {len(features)} ميزة...")
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"متوسط مربع الخطأ للنموذج: {mse:.6f}")
    
    # حفظ النموذج وقائمة الميزات
    if not os.path.exists("models"):
        os.makedirs("models")
    
    joblib.dump(model, model_output_path)
    joblib.dump(features, "models/feature_names.joblib")
    print(f"تم حفظ النموذج المتقدم في {model_output_path}")

if __name__ == "__main__":
    train_tasi_model("data/tasi_processed.csv", "models/tasi_rf_model.joblib")
