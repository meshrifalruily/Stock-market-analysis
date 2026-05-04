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
        max_iter=100, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=0.2, random_state=42
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
    target = 'target_next_week_return' # التغيير للتنبؤ الأسبوعي لتقليل الضوضاء
    
    # الاختبار العكسي لآخر 6 أشهر
    max_date = df['date'].max()
    start_date = max_date - pd.Timedelta(days=180)
    test_df = df[df['date'] >= start_date].copy()
    
    dates = sorted(test_df['date'].unique())
    portfolio_value = 100.0 
    portfolio_history = []
    COMMISSION = 0.00155 
    SLIPPAGE = 0.0005
    MIN_WEEKLY_THRESHOLD = 0.018 # Higher conviction entry
    STOP_LOSS_ATR_MULT = 1.5
    TAKE_PROFIT_MULT = 5.0 # Increase reward/risk to cover fees

    
    MIN_TRAIN_DAYS = 252
    RETRAIN_EVERY_N_DAYS = 20
    model = None
    medians = None
    last_retrain_idx = None
    
    current_holdings = {} # {symbol: {'entry_price': price, 'stop_loss': price}}
    
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        
        # 1. التدريب التدريجي
        history_df = df[df['date'] < current_date].copy()
        train_dates = history_df['date'].nunique()
        if train_dates < MIN_TRAIN_DAYS:
            continue

        if model is None or last_retrain_idx is None or (i - last_retrain_idx) >= RETRAIN_EVERY_N_DAYS:
            print(f"إعادة تدريب walk-forward حتى {current_date.date()} ({train_dates} أيام تدريب)...")
            model, medians = train_walk_forward_model(history_df, features, target)
            last_retrain_idx = i

        # 2. جلب بيانات اليوم الحالي وتصفية السيولة
        day_data = test_df[test_df['date'] == current_date].copy()
        if 'avg_traded_value_20d' in day_data.columns:
            day_data = day_data[day_data['avg_traded_value_20d'] >= MIN_AVG_TRADED_VALUE]
        
        if day_data.empty:
            portfolio_history.append({'date': next_date, 'returns': 0, 'value': portfolio_value, 'selected_symbols': "", 'avg_pred_return': 0.0})
            continue

        # 3. التنبؤ واختيار الأسهم
        X = prepare_features(day_data, features, medians)
        day_data['pred_return'] = model.predict(X)

        # Regime Filter: فحص حالة السوق العامة
        market_breadth = day_data['market_breadth_sma50'].iloc[0] if 'market_breadth_sma50' in day_data.columns else 0.5

        # 4. إدارة المحفظة
        next_day_all = test_df[test_df['date'] == next_date]
        daily_returns_sum = 0
        new_holdings = {}
        
        top_candidates = day_data.sort_values('pred_return', ascending=False)
        top_symbols = top_candidates.head(20)['symbol'].tolist() # بقاء السهم ضمن أفضل 20 يمنع البيع العبثي
        
        # معالجة الأسهم الحالية
        for symbol, info in current_holdings.items():
            stock_current = day_data[day_data['symbol'] == symbol]
            stock_next = next_day_all[next_day_all['symbol'] == symbol]
            
            if stock_next.empty or stock_current.empty:
                # خروج اضطراري لعدم وجود بيانات (بيع)
                portfolio_value *= (1 - (COMMISSION + SLIPPAGE))
                continue
                
            curr_price = stock_current['close'].values[0]
            next_ret = stock_next['daily_return'].values[0]
            atr = stock_current['atr'].values[0] if 'atr' in stock_current.columns else 0
            
            # تحديث وقف الخسارة المتحرك
            new_stop = max(info['stop_loss'], curr_price - (STOP_LOSS_ATR_MULT * atr))
            
            # قرار البيع: كسر وقف الخسارة، الوصول للهدف، أو خروج من التوب 20، أو تنبؤ سلبي جداً، أو انهيار السوق
            pred_val = stock_current['pred_return'].values[0]
            should_sell = (curr_price < info['stop_loss']) or \
                          (curr_price > info['take_profit']) or \
                          (symbol not in top_symbols) or \
                          (pred_val < -0.005) or \
                          (market_breadth < 0.2)
            
            if should_sell:
                # بيع
                portfolio_value *= (1 + next_ret) 
                portfolio_value *= (1 - (COMMISSION + SLIPPAGE)) 
            else:
                # استمرار الاحتفاظ
                daily_returns_sum += next_ret
                new_holdings[symbol] = {'entry_price': info['entry_price'], 'stop_loss': new_stop, 'take_profit': info['take_profit']}

        # 5. الدخول في أسهم جديدة (فقط إذا كان السوق آمناً)
        available_slots = 3 - len(new_holdings)
        if available_slots > 0 and market_breadth > 0.4:
            potential_buys = top_candidates[
                (top_candidates['pred_return'] >= MIN_WEEKLY_THRESHOLD) & 
                (~top_candidates['symbol'].isin(new_holdings.keys()))
            ].head(available_slots)
            
            for _, row in potential_buys.iterrows():
                symbol = row['symbol']
                stock_next = next_day_all[next_day_all['symbol'] == symbol]
                if not stock_next.empty:
                    next_ret = stock_next['daily_return'].values[0]
                    # تكلفة الشراء
                    portfolio_value *= (1 - (COMMISSION + SLIPPAGE))
                    # عائد اليوم الأول
                    daily_returns_sum += next_ret
                    atr = row['atr'] if 'atr' in row else row['close'] * 0.02
                    new_holdings[symbol] = {
                        'entry_price': row['close'],
                        'stop_loss': row['close'] - (STOP_LOSS_ATR_MULT * atr),
                        'take_profit': row['close'] + (TAKE_PROFIT_MULT * STOP_LOSS_ATR_MULT * atr)
                    }

        # حساب العائد اليومي للمحفظة ككل (تبسيط: نفترض الوزن متساوي)
        # ملاحظة: تم خصم التكاليف مباشرة من portfolio_value، لذا returns هنا للعرض
        active_count = len(new_holdings)
        avg_ret = (daily_returns_sum / active_count) if active_count > 0 else 0
        if active_count > 0:
            portfolio_value *= (1 + avg_ret)
            
        current_holdings = new_holdings
        
        portfolio_history.append({
            'date': next_date,
            'returns': avg_ret,
            'value': portfolio_value,
            'selected_symbols': ",".join(current_holdings.keys()),
            'avg_pred_return': float(day_data[day_data['symbol'].isin(current_holdings.keys())]['pred_return'].mean()) if current_holdings else 0.0
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
