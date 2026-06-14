import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings

# فترة الاختبار العكسي بالأيام — قابلة للتمديد عبر متغير البيئة.
# الافتراضي 365 يوماً (سنة كاملة) بدل 180 لزيادة الدلالة الإحصائية.
BACKTEST_LOOKBACK_DAYS = int(os.getenv("TASI_BACKTEST_DAYS", "365"))
import matplotlib
matplotlib.use("Agg")
import quantstats as qs
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.strategy_config import load_strategy_config
from scripts.strategy_rules import add_hybrid_scores, entry_candidates, filter_main_market, summarize_returns

# تجاهل التحذيرات الرياضية المتوقعة عند التعامل مع التباين الصفري
warnings.filterwarnings('ignore', category=RuntimeWarning)

def prepare_features(df, features, medians):
    X = df[features].replace([np.inf, -np.inf], np.nan)
    return X.fillna(medians).fillna(0)

def train_walk_forward_model(history_df, features, target):
    # استخدام آخر 3 سنوات للتدريب لضمان استقرار النماذج
    train_df = history_df.sort_values('date').tail(756 * 200).dropna(subset=[target]).copy() 
    if len(train_df) < 2000:
        train_df = history_df.dropna(subset=[target]).copy()
        
    X_raw = train_df[features].replace([np.inf, -np.inf], np.nan)
    medians = X_raw.median(numeric_only=True).fillna(0)
    X = X_raw.fillna(medians).fillna(0)
    y = train_df[target]
    
    # نموذج متوازن وسريع
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.03, max_leaf_nodes=63, l2_regularization=1.5, random_state=42, class_weight='balanced'
    )
    model.fit(X, y)
    reg_model = None
    if 'target_next_day_return' in history_df.columns:
        reg_train_df = history_df.sort_values('date').tail(756 * 200).dropna(subset=['target_next_day_return']).copy()
        if len(reg_train_df) < 2000:
            reg_train_df = history_df.dropna(subset=['target_next_day_return']).copy()
        if len(reg_train_df) >= 2000:
            X_reg = reg_train_df[features].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
            y_reg = reg_train_df['target_next_day_return'].clip(-0.2, 0.2)
            reg_model = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.03,
                max_leaf_nodes=31,
                l2_regularization=1.5,
                random_state=42,
            )
            reg_model.fit(X_reg, y_reg)
    return model, medians, reg_model

def run_backtest(processed_file_path, model_path, features_path):
    config = load_strategy_config()
    if not os.path.exists(processed_file_path) or not os.path.exists(features_path):
        print("الميزات أو البيانات غير موجودة للاختبار العكسي.")
        return

    df = pd.read_csv(processed_file_path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    before_symbols = df['symbol'].nunique()
    df = filter_main_market(df)
    print(f"الاختبار العكسي على السوق الرئيسي فقط: {df['symbol'].nunique()} من أصل {before_symbols} رمز.")

    features = joblib.load(features_path)
    target = 'target_alpha_weekly' # استهداف المتفوقين أسبوعياً
    
    # الاختبار العكسي للفترة المحددة (افتراضياً سنة كاملة)
    max_date = df['date'].max()
    start_date = max_date - pd.Timedelta(days=BACKTEST_LOOKBACK_DAYS)
    test_df = df[df['date'] >= start_date].copy()
    print(f"فترة الاختبار العكسي: {BACKTEST_LOOKBACK_DAYS} يوماً (من {start_date.date()} إلى {max_date.date()}).")

    dates = sorted(test_df['date'].unique())
    portfolio_value = 100.0 
    portfolio_history = []
    COMMISSION = 0.00155 
    SLIPPAGE = 0.0005
    
    MIN_TRAIN_DAYS = 252
    RETRAIN_EVERY_N_DAYS = 90 
    model = None
    reg_model = None
    medians = None
    last_retrain_idx = None

    current_holdings = {} # {symbol: {'entry_price': price, 'stop_loss': price, 'weight': weight, 'days_held': 0}}

    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]

        # 1. التدريب التدريجي
        history_df = df[df['date'] < current_date].copy()
        train_dates = history_df['date'].nunique()
        if train_dates < MIN_TRAIN_DAYS:
            continue

        if model is None or last_retrain_idx is None or (i - last_retrain_idx) >= RETRAIN_EVERY_N_DAYS:
            print(f"إعادة تدريب Alpha-Model V6 حتى {current_date.date()}...")
            model, medians, reg_model = train_walk_forward_model(history_df, features, target)
            last_retrain_idx = i

        # 2. جلب بيانات اليوم الحالي وتصفية السيولة
        day_data = test_df[test_df['date'] == current_date].copy()
        if 'avg_traded_value_20d' in day_data.columns:
            day_data = day_data[day_data['avg_traded_value_20d'] >= config["min_avg_traded_value"]]

        if day_data.empty:
            portfolio_history.append({'date': next_date, 'returns': 0, 'value': portfolio_value, 'selected_symbols': "", 'avg_prob': 0.0})
            continue

        # 3. التنبؤ وحساب جودة الإشارة الهجينة
        X = prepare_features(day_data, features, medians)
        day_data['prob_win'] = model.predict_proba(X)[:, 1]
        day_data['predicted_next_return'] = (
            np.clip(reg_model.predict(X), -0.2, 0.2)
            if reg_model is not None
            else 0.0
        )
        day_data = add_hybrid_scores(day_data)

        # Regime Filter
        market_breadth = day_data['market_breadth_sma50'].iloc[0] if 'market_breadth_sma50' in day_data.columns else 0.5

        if i % 20 == 0:
            print(f"Date: {current_date.date()} | Breadth: {market_breadth:.2f} | Best Prob: {day_data['prob_win'].max():.4f}")

        # 4. إدارة المحفظة
        next_day_all = test_df[test_df['date'] == next_date]
        daily_portfolio_return = 0.0
        new_holdings = {}

        for symbol, info in current_holdings.items():
            stock_current = day_data[day_data['symbol'] == symbol]
            stock_next = next_day_all[next_day_all['symbol'] == symbol]
            weight = info.get('weight', 1/3.0)

            if stock_next.empty or stock_current.empty:
                daily_portfolio_return -= (COMMISSION + SLIPPAGE) * weight
                continue

            curr_price = stock_current['close'].values[0]
            next_ret = stock_next['daily_return'].values[0]
            atr = stock_current['atr'].values[0] if 'atr' in stock_current.columns else 0
            
            new_stop = max(info['stop_loss'], curr_price - (config["stop_loss_atr_mult"] * atr))
            if curr_price >= info['entry_price'] * 1.012:
                new_stop = max(new_stop, info['entry_price'])

            daily_portfolio_return += next_ret * weight
            
            prob_val = stock_current['prob_win'].values[0]
            rank_val = stock_current['rank'].values[0]
            days_held = info.get('days_held', 0) + 1

            # خروج ذكي للانعكاس: العودة للوسط. اشترينا تراجعاً قصير المدى، فنجني
            # الربح عند ارتداد السعر فوق متوسطه القصير (sma_20) ونحن في ربح —
            # هذا يلتقط الارتداد المتوقّع بدل انتظار هدف ATR ثابت قد لا يتحقّق.
            sma20_now = stock_current['sma_20'].values[0] if 'sma_20' in stock_current.columns else np.nan
            reverted_to_mean = (
                not np.isnan(sma20_now)
                and curr_price >= sma20_now
                and curr_price > info['entry_price']
            )

            # خروج مناسب للانعكاس: زمني/مخاطر/عودة-للوسط. أُزيل البيع المبكر المبني
            # على احتمال النموذج (IC سالب)، لأنه يقصّ الرابحين ويبقي الخاسرين.
            should_sell = (curr_price < info['stop_loss']) or \
                          (curr_price > info['take_profit']) or \
                          reverted_to_mean or \
                          (market_breadth < config["market_breadth_exit"]) or \
                          (days_held >= config["max_hold_days"])

            if should_sell:
                daily_portfolio_return -= (COMMISSION + SLIPPAGE) * weight
            else:
                new_holdings[symbol] = {
                    'entry_price': info['entry_price'], 
                    'stop_loss': new_stop, 
                    'take_profit': info['take_profit'],
                    'weight': weight,
                    'days_held': days_held
                }

        # 5. الدخول في صفقات جديدة
        available_slots = config["max_positions"] - len(new_holdings)
        if available_slots > 0 and market_breadth > config["min_entry_market_breadth"]:
            potential_buys = entry_candidates(day_data, new_holdings.keys(), config).head(available_slots)

            for _, row in potential_buys.iterrows():
                symbol = row['symbol']
                stock_next = next_day_all[next_day_all['symbol'] == symbol]
                if not stock_next.empty:
                    next_ret = stock_next['daily_return'].values[0]
                    atr = row['atr'] if 'atr' in row else row['close'] * 0.02
                    atr_pct = atr / row['close'] if row['close'] > 0 else 0.02
                    
                    weight = min(1 / config["max_positions"], config["risk_per_position"] / max(0.001, atr_pct))
                    
                    daily_portfolio_return -= (COMMISSION + SLIPPAGE) * weight
                    daily_portfolio_return += next_ret * weight
                    
                    new_holdings[symbol] = {
                        'entry_price': row['close'],
                        'stop_loss': row['close'] - (config["stop_loss_atr_mult"] * atr),
                        'take_profit': row['close'] + (config["take_profit_atr_mult"] * config["stop_loss_atr_mult"] * atr),
                        'weight': weight,
                        'days_held': 0
                    }

        portfolio_value *= (1 + daily_portfolio_return)
        current_holdings = new_holdings

        portfolio_history.append({
            'date': next_date,
            'returns': daily_portfolio_return,
            'value': portfolio_value,
            'selected_symbols': ",".join(current_holdings.keys()),
            'avg_prob': float(day_data[day_data['symbol'].isin(current_holdings.keys())]['prob_win'].mean()) if current_holdings else 0.0
        })

    if not portfolio_history:
        print("لا توجد بيانات كافية لتشغيل الاختبار العكسي.")
        return

    perf_df = pd.DataFrame(portfolio_history).set_index('date')
    perf_df.index = pd.to_datetime(perf_df.index).tz_localize(None)
    
    # المعيار المرجعي = مؤشر السوق الحقيقي وليس سهماً منفرداً.
    # نفضّل مؤشر تاسي الفعلي (macro_tasi_proxy) إن توفر، وإلا نبني مؤشراً
    # متساوي الأوزان من متوسط عوائد جميع أسهم السوق الرئيسي يومياً.
    if 'macro_tasi_proxy' in df.columns and df['macro_tasi_proxy'].notna().any():
        tasi_index = (
            df.dropna(subset=['macro_tasi_proxy'])
            .groupby('date')['macro_tasi_proxy'].first()
            .sort_index()
        )
        benchmark = tasi_index.pct_change().fillna(0)
        print("المعيار المرجعي: مؤشر تاسي الفعلي (macro_tasi_proxy).")
    else:
        benchmark = df.groupby('date')['daily_return'].mean().sort_index()
        print("المعيار المرجعي: مؤشر تاسي تقريبي متساوي الأوزان (متوسط عوائد السوق).")
    benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
    
    perf_df = perf_df[~perf_df.index.duplicated(keep='first')]
    benchmark = benchmark[~benchmark.index.duplicated(keep='first')]
    
    common_idx = perf_df.index.intersection(benchmark.index)
    returns_series = perf_df.loc[common_idx, 'returns']
    benchmark_series = benchmark.loc[common_idx]

    if returns_series.std() == 0:
        returns_series = returns_series + np.random.normal(0, 1e-10, len(returns_series))

    print("\n--- استراتيجية الانعكاس قصير المدى (مرجعية تشخيصية فقط) ---")
    print("⚠️ هذه ليست الاستراتيجية الموصى بها — أُثبت أنها لا تتغلّب على التكاليف.")
    print(f"قيمة المحفظة النهائية: {portfolio_value:.2f}")
    print(f"إجمالي العائد الصافي: {((portfolio_value / 100.0) - 1) * 100:.2f}%")
    summary = summarize_returns(returns_series)
    print(f"شارب: {summary['sharpe']:.2f} | أقصى هبوط: {summary['max_drawdown'] * 100:.2f}% | أيام النشاط: {summary['active_days']}")

    if not os.path.exists("reports"): os.makedirs("reports")
    try:
        qs.reports.html(returns_series, benchmark=benchmark_series, output='reports/tasi_reversal_reference_report.html')
    except: pass

    # نكتب مرجع الانعكاس في ملف منفصل حتى لا يطغى على الباك تست الرئيسي (الزخم).
    perf_df.to_csv("data/backtest_reversal_reference.csv")

if __name__ == "__main__":
    # الباك تست الرئيسي = استراتيجية الزخم الموصى بها (يكتب data/backtest_results.csv).
    from scripts.momentum_strategy import save_validation_report
    print("=== الباك تست الرئيسي: استراتيجية الزخم طويلة الأفق (الموصى بها) ===")
    res = save_validation_report("data/tasi_processed.csv")
    if res:
        print(f"إجمالي العائد الصافي: {res['strat_total']*100:.2f}%  |  السوق: {res['market_total']*100:.2f}%  "
              f"|  Alpha-Sharpe: {res['alpha_sharpe']}  |  ألفا موجب: {res['positive_alpha_years']}/{res['total_years']} سنة")

    # استراتيجية الانعكاس كمرجع تشخيصي فقط (بطيئة) — تُشغَّل عند الطلب الصريح.
    if os.getenv("TASI_RUN_REVERSAL_BACKTEST", "0") == "1":
        run_backtest("data/tasi_processed.csv", "models/tasi_rf_model_weekly.joblib", "models/feature_names.joblib")
