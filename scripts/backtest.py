import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import quantstats as qs
from sklearn.ensemble import HistGradientBoostingRegressor

MIN_AVG_TRADED_VALUE = float(os.getenv("TASI_MIN_AVG_TRADED_VALUE", "1000000"))

def prepare_features(df, features, medians):
    X = df[features].replace([np.inf, -np.inf], np.nan)
    return X.fillna(medians).fillna(0)

def train_walk_forward_model(history_df, features, target):
    train_df = history_df.dropna(subset=[target]).copy()
    X_raw = train_df[features].replace([np.inf, -np.inf], np.nan)
    medians = X_raw.median(numeric_only=True).fillna(0)
    X = X_raw.fillna(medians).fillna(0)
    y = train_df[target]
    model = HistGradientBoostingRegressor(
        max_iter=80, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.05, random_state=42
    )
    model.fit(X, y)
    return model, medians

def run_backtest(processed_file_path, model_path, features_path):
    if not os.path.exists(processed_file_path) or not os.path.exists(features_path):
        print("الميزات أو البيانات غير موجودة للاختبار العكسي.")
        return
    
    df = pd.read_csv(processed_file_path)
    
    # تحويل التاريخ وتجريده من المنطقة الزمنية فوراً وبشكل صارم
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    features = joblib.load(features_path)
    target = 'target_next_day_return'
    
    # الاختبار العكسي لآخر 6 أشهر
    max_date = df['date'].max()
    start_date = max_date - pd.Timedelta(days=180)
    test_df = df[df['date'] >= start_date].copy()
    
    dates = sorted(test_df['date'].unique())
    portfolio_value = 100.0 
    portfolio_history = []
    COMMISSION = 0.00155 
    SLIPPAGE = 0.0005
    MIN_TRAIN_DAYS = 252
    RETRAIN_EVERY_N_DAYS = 20
    model = None
    medians = None
    last_retrain_idx = None
    
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        history_df = df[df['date'] < current_date].copy()
        train_dates = history_df['date'].nunique()
        if train_dates < MIN_TRAIN_DAYS:
            continue

        if model is None or last_retrain_idx is None or (i - last_retrain_idx) >= RETRAIN_EVERY_N_DAYS:
            print(f"إعادة تدريب walk-forward حتى {current_date.date()} ({train_dates} أيام تدريب)...")
            model, medians = train_walk_forward_model(history_df, features, target)
            last_retrain_idx = i

        day_data = test_df[test_df['date'] == current_date].copy()
        if 'avg_traded_value_20d' in day_data.columns:
            day_data = day_data[day_data['avg_traded_value_20d'] >= MIN_AVG_TRADED_VALUE]
        if day_data.empty:
            portfolio_history.append({
                'date': next_date,
                'returns': 0,
                'value': portfolio_value,
                'selected_symbols': "",
                'avg_pred_return': 0.0
            })
            continue
        
        X = prepare_features(day_data, features, medians)
        day_data['pred_return'] = model.predict(X)
        top_3 = day_data.sort_values('pred_return', ascending=False).head(3)
        
        next_day_all = test_df[test_df['date'] == next_date]
        daily_returns = []
        for symbol in top_3['symbol']:
            next_day_stock = next_day_all[next_day_all['symbol'] == symbol]
            if not next_day_stock.empty:
                actual_ret = next_day_stock['daily_return'].values[0]
                if not pd.isna(actual_ret):
                    net_ret = actual_ret - ((COMMISSION + SLIPPAGE) * 2) 
                    daily_returns.append(net_ret)
        
        avg_daily_ret = np.mean(daily_returns) if daily_returns else 0
        if not pd.isna(avg_daily_ret):
            portfolio_value *= (1 + avg_daily_ret)
        portfolio_history.append({
            'date': next_date,
            'returns': avg_daily_ret,
            'value': portfolio_value,
            'selected_symbols': ",".join(top_3['symbol'].astype(str).tolist()),
            'avg_pred_return': float(top_3['pred_return'].mean()) if not top_3.empty else 0.0
        })

    if not portfolio_history:
        print("لا توجد بيانات كافية لتشغيل الاختبار العكسي.")
        return

    perf_df = pd.DataFrame(portfolio_history).set_index('date')
    # تجريد المنطقة الزمنية من الفهرس
    perf_df.index = pd.to_datetime(perf_df.index).tz_localize(None)
    
    # تجهيز المعيار (الراجحي) وتجريده من المنطقة الزمنية
    benchmark_df = df[df['symbol'] == '1120.SR'][['date', 'daily_return']].copy()
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date']).dt.tz_localize(None)
    benchmark = benchmark_df.set_index('date')['daily_return']
    
    # تنظيف المكررات وتوحيد الفهرس
    perf_df = perf_df[~perf_df.index.duplicated(keep='first')]
    benchmark = benchmark[~benchmark.index.duplicated(keep='first')]
    
    common_idx = perf_df.index.intersection(benchmark.index)
    returns_series = perf_df.loc[common_idx, 'returns']
    benchmark_series = benchmark.loc[common_idx]

    # التأكد النهائي الحاسم من تجريد المناطق الزمنية
    returns_series.index = returns_series.index.tz_localize(None)
    benchmark_series.index = benchmark_series.index.tz_localize(None)

    print("\n--- نتائج الاختبار العكسي الواقعي Walk-Forward (آخر 6 أشهر) ---")
    print(f"قيمة المحفظة النهائية: {portfolio_value:.2f}")
    print(f"إجمالي عائد الاستراتيجية: {((portfolio_value / 100.0) - 1) * 100:.2f}%")
    print(f"تم احتساب عمولة {COMMISSION:.3%} وانزلاق سعري {SLIPPAGE:.3%} لكل دخول/خروج.")
    
    if not os.path.exists("reports"):
        os.makedirs("reports")
    
    try:
        # إرسال البيانات كـ Series "خام" تماماً
        qs.reports.html(returns_series, benchmark=benchmark_series, output='reports/tasi_ai_backtest_report.html', title='استراتيجية تاسي الذكية ضد المعيار')
        print("تم حفظ التقرير الكامل بنجاح في reports/tasi_ai_backtest_report.html")
    except Exception as e:
        print(f"خطأ أثناء توليد HTML: {e}")
        print(f"نسبة شارب: {qs.stats.sharpe(returns_series):.2f}")
    
    perf_df.to_csv("data/backtest_results.csv")

if __name__ == "__main__":
    # استخدام النموذج اليومي الجديد للاختبار العكسي
    run_backtest("data/tasi_processed.csv", "models/tasi_rf_model_daily.joblib", "models/feature_names.joblib")
