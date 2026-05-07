import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os
import json
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.strategy_config import load_strategy_config
from scripts.strategy_rules import add_hybrid_scores, entry_candidates, filter_main_market

MODEL_DIR = "models"
NOMU_PREFIXES = ("95", "96")

def is_main_market_symbol(symbol):
    code = str(symbol).replace(".SR", "").strip()
    return not code.startswith(NOMU_PREFIXES)

def get_candidate_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=15, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=15, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.02, max_leaf_nodes=127, l2_regularization=1.5, random_state=42, class_weight='balanced'
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

def evaluate_model(model, val_df, X_val, y_val, config):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds)),
        "auc": float(roc_auc_score(y_val, probs)),
    }
    
    # تحديد عمود العائد الحقيقي للمقارنة (regression reference)
    if 'day' in y_val.name:
        actual_ret_col = 'target_next_day_return'
    elif 'weekly' in y_val.name:
        actual_ret_col = 'target_next_week_return'
    else:
        actual_ret_col = 'target_return_10d'

    val_df = val_df.copy()
    val_df['prob_win'] = probs
    if 'avg_traded_value_20d' in val_df.columns:
        val_df = val_df[val_df['avg_traded_value_20d'] >= config["min_avg_traded_value"]]
    val_df = add_hybrid_scores(val_df)
    
    strategy_rets = []
    selected_days = 0
    for date, group in val_df.groupby('date'):
        market_breadth = group['market_breadth_sma50'].iloc[0] if 'market_breadth_sma50' in group.columns else 0.5
        if market_breadth <= config["min_entry_market_breadth"]:
            strategy_rets.append(0.0)
            continue
        picks = entry_candidates(group, [], config).head(config["max_positions"])
        if picks.empty:
            strategy_rets.append(0.0)
            continue
        selected_days += 1
        strategy_rets.append(picks[actual_ret_col].mean())
    
    strategy_rets = pd.Series(strategy_rets).replace([np.inf, -np.inf], np.nan).fillna(0)
    metrics["strategy_avg_return"] = float(strategy_rets.mean())
    metrics["strategy_total_return"] = float((1 + strategy_rets).prod() - 1)
    metrics["selected_days"] = selected_days
    metrics["score"] = (metrics["strategy_total_return"] * 100) + (metrics["auc"] * 10)
    return metrics

def train_tasi_models(processed_file_path):
    if not os.path.exists(processed_file_path):
        print(f"خطأ: {processed_file_path} غير موجود.")
        return
    
    df = pd.read_csv(processed_file_path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    before_symbols = df['symbol'].nunique()
    df = filter_main_market(df)
    print(f"تدريب النماذج على السوق الرئيسي فقط: {df['symbol'].nunique()} من أصل {before_symbols} رمز.")
    config = load_strategy_config()
    
    # تحديد الميزات الشاملة (Alpha High-Signal Features)
    features = [
        'rsi', 'ema_20', 'macd', 'macd_signal', 'atr_pct',
        'pe_ratio', 'div_yield', 'beta', 'oil_correlation',
        'tv_signal', 'tv_adx', 'sentiment',
        'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'weekly_return_hist', 'volatility_10d', 'volatility_20d', 
        'volume_ratio_20d', 'volume_ratio_5d',
        'volume_zscore', 'traded_value_zscore', 'volatility_20d_zscore',
        'momentum_20d', 'momentum_10d',
        'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_sma_200',
        'market_return', 'relative_return_1d', 'relative_return_5d',
        'oil_return', 'day_of_week', 'month', 'market_breadth_sma20',
        'market_breadth_sma50', 'sector_return_1d', 'sector_relative_return_1d',
        'sector_momentum_20d', 'relative_sector_alpha',
        'sector_alpha_zscore', 'is_breakout_20d', 'is_breakout_50d',
        'relative_volume', 'vol_shock', 'price_velocity', 'distance_from_high_20d',
        'rsi_slope', 'rsi_relative', 'macd_histogram_slope', 'bb_width',
        'price_vs_high_20d', 'stoch_rsi_k', 'stoch_rsi_d', 'atr_slope'
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
        # تحديد عمود العائد الحقيقي المرتبط بالهدف
        if 'day' in target:
            actual_ret_col = 'target_next_day_return'
        elif 'weekly' in target:
            actual_ret_col = 'target_next_week_return'
        else:
            actual_ret_col = 'target_return_10d'

        # تنظيف البيانات من القيم المفقودة في الهدف والعائد الحقيقي
        target_df = df.dropna(subset=[target, actual_ret_col]).sort_values('date').copy()
        train_df, val_df = chronological_split(target_df)
        
        if train_df.empty or val_df.empty:
            print(f"بيانات ناقصة لـ {target}")
            return None

        _, X_train, y_train, medians = prepare_frame(train_df, features, target)
        val_clean, X_val, y_val, _ = prepare_frame(val_df, features, target, medians)
        
        candidates = get_candidate_models()
        results = {}
        best_name = None
        best_score = -np.inf

        for name, candidate in candidates.items():
            model = clone(candidate)
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, val_clean, X_val, y_val, config)
            results[name] = metrics
            print(
                f"{target} | {name}: ACC={metrics['accuracy']:.2%}, AUC={metrics['auc']:.4f}, "
                f"StrategyRet={metrics['strategy_total_return']:.2%}, Days={metrics['selected_days']}"
            )
            
            if metrics["score"] > best_score:
                best_name = name
                best_score = metrics["score"]

        final_model = clone(candidates[best_name])
        _, X_all, y_all, _ = prepare_frame(target_df, features, target, feature_medians)
        final_model.fit(X_all, y_all)
        joblib.dump(final_model, output_path)
        return {"selected_model": best_name, "validation": results}
    
    # 1. تدريب النموذج اليومي (Alpha Outperformer)
    print("--- تدريب النموذج اليومي ---")
    model_metrics["daily"] = train_one_target('target_outperformer_1d', os.path.join(MODEL_DIR, "tasi_rf_model_daily.joblib"))
    
    # 2. تدريب النموذج الأسبوعي (Top 25% Winner)
    print("\n--- تدريب النموذج الأسبوعي (5 أيام) ---")
    model_metrics["weekly"] = train_one_target('target_alpha_weekly', os.path.join(MODEL_DIR, "tasi_rf_model_weekly.joblib"))
    
    # 3. تدريب النموذج المتوسط (أسبوعين)
    print("\n--- تدريب النموذج المتوسط (10 أيام) ---")
    model_metrics["medium"] = train_one_target('target_alpha_medium', os.path.join(MODEL_DIR, "tasi_rf_model_medium.joblib"))

    # حفظ الأسماء والبيانات
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(feature_medians, os.path.join(MODEL_DIR, "feature_medians.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\nتم بنجاح تدريب وحفظ جميع نماذج Alpha V6.0.")

if __name__ == "__main__":
    train_tasi_models("data/tasi_processed.csv")
