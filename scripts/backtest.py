import pandas as pd
import numpy as np
import joblib
import os
import quantstats as qs

def run_backtest(processed_file_path, model_path, features_path):
    if not os.path.exists(processed_file_path) or not os.path.exists(model_path):
        print("النموذج أو البيانات غير موجودة للاختبار العكسي.")
        return
    
    df = pd.read_csv(processed_file_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    
    # الاختبار العكسي لآخر 6 أشهر
    max_date = df['date'].max()
    start_date = max_date - pd.Timedelta(days=180)
    test_df = df[df['date'] >= start_date].copy()
    
    dates = sorted(test_df['date'].unique())
    portfolio_value = 100.0 
    portfolio_history = []
    COMMISSION = 0.00155 
    
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        day_data = test_df[test_df['date'] == current_date].copy()
        
        X = day_data[features].fillna(0)
        day_data['pred_return'] = model.predict(X)
        top_3 = day_data.sort_values('pred_return', ascending=False).head(3)
        
        next_day_all = test_df[test_df['date'] == next_date]
        daily_returns = []
        for symbol in top_3['symbol']:
            next_day_stock = next_day_all[next_day_all['symbol'] == symbol]
            if not next_day_stock.empty:
                actual_ret = next_day_stock['daily_return'].values[0]
                net_ret = actual_ret - (COMMISSION * 2) 
                daily_returns.append(net_ret)
        
        avg_daily_ret = np.mean(daily_returns) if daily_returns else 0
        portfolio_value *= (1 + avg_daily_ret)
        portfolio_history.append({'date': next_date, 'returns': avg_daily_ret, 'value': portfolio_value})

    if not portfolio_history:
        print("لا توجد بيانات كافية لتشغيل الاختبار العكسي.")
        return

    perf_df = pd.DataFrame(portfolio_history).set_index('date')
    benchmark_df = df[df['symbol'] == '1120.SR'][['date', 'daily_return']].copy()
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], utc=True)
    benchmark = benchmark_df[benchmark_df['date'].isin(perf_df.index)].set_index('date')['daily_return']
    
    perf_df = perf_df[~perf_df.index.duplicated(keep='first')]
    benchmark = benchmark[~benchmark.index.duplicated(keep='first')]
    common_idx = perf_df.index.intersection(benchmark.index)
    returns_series = perf_df.loc[common_idx, 'returns']
    benchmark_series = benchmark.loc[common_idx]

    print("\n--- نتائج الاختبار العكسي (آخر 6 أشهر) ---")
    print(f"قيمة المحفظة النهائية: {portfolio_value:.2f}")
    print(f"إجمالي عائد الاستراتيجية: {((portfolio_value / 100.0) - 1) * 100:.2f}%")
    
    if not os.path.exists("reports"):
        os.makedirs("reports")
    
    try:
        qs.reports.html(returns_series, benchmark=benchmark_series, output='reports/tasi_ai_backtest_report.html', title='استراتيجية تاسي الذكية ضد المعيار')
        print("تم حفظ التقرير الكامل في reports/tasi_ai_backtest_report.html")
    except Exception as e:
        print(f"لم يتمكن النظام من توليد تقرير HTML: {e}")
        print(f"نسبة شارب: {qs.stats.sharpe(returns_series):.2f}")
        print(f"أقصى تراجع: {qs.stats.max_drawdown(returns_series)*100:.2f}%")
    
    perf_df.to_csv("data/backtest_results.csv")

if __name__ == "__main__":
    run_backtest("data/tasi_processed.csv", "models/tasi_rf_model.joblib", "models/feature_names.joblib")
