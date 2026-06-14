import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
try:
    # sklearn >= 1.6: cv='prefit' أُزيل، ويُستبدل بـ FrozenEstimator.
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover
    FrozenEstimator = None
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
PRICE_LAG_DAYS = 10
# أقصى أفق نظر مستقبلي للأهداف (target_return_10d ينظر 10 أيام).
# نترك فجوة embargo بهذا الحجم بين التدريب والتحقق لمنع تسريب المستقبل.
LABEL_HORIZON_DAYS = 10

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

def get_candidate_regressors():
    return {
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=63, l2_regularization=1.0, random_state=42
        ),
    }

def chronological_split(df, validation_ratio=0.2, embargo_days=LABEL_HORIZON_DAYS):
    """فصل زمني مع فجوة embargo بين التدريب والتحقق.

    الأهداف تنظر حتى ``embargo_days`` يوماً للمستقبل، لذا فإن آخر أيام التدريب
    تحمل عوائد محسوبة من أسعار تقع داخل فترة التحقق. نحذف هذه الأيام (purge)
    لمنع تسريب المستقبل وضمان أن مقاييس التحقق صادقة.
    """
    dates = np.array(sorted(pd.to_datetime(df['date']).unique()))
    if len(dates) < 10:
        return df.iloc[:0], df
    split_idx = max(1, int(len(dates) * (1 - validation_ratio)))
    # حذف آخر embargo_days من التدريب لمنع تداخل نوافذ الأهداف مع التحقق.
    purged_train_end = max(1, split_idx - embargo_days)
    train_dates = dates[:purged_train_end]
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

def safe_corr(left, right, method="spearman"):
    """Correlation without pandas/scipy ConstantInputWarning on flat samples."""
    data = pd.concat(
        [
            pd.Series(left, dtype="float64").reset_index(drop=True),
            pd.Series(right, dtype="float64").reset_index(drop=True),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return 0.0
    value = data.iloc[:, 0].corr(data.iloc[:, 1], method=method)
    return 0.0 if pd.isna(value) else float(value)

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

    # ── القيمة الاقتصادية الحقيقية (تُنتج إشارة دائماً، بلا فلاتر صارمة) ──
    # نرتّب أسهم كل يوم حسب احتمالية الفوز، ونقارن عائد الخُمس الأعلى بالأدنى.
    # هذا المقياس لا "ينهار إلى صفر" مثل فلاتر الدخول الصارمة، فيعكس فعلياً
    # ما إذا كان النموذج يميّز الرابحين عن الخاسرين عبر المقطع العرضي.
    top_quintile_rets, bottom_quintile_rets, daily_ic = [], [], []
    for _, group in val_df.groupby('date'):
        group = group.dropna(subset=[actual_ret_col, 'prob_win'])
        if len(group) < 5:
            continue
        n = max(1, int(len(group) * 0.20))
        ranked = group.sort_values('prob_win', ascending=False)
        top_quintile_rets.append(ranked.head(n)[actual_ret_col].mean())
        bottom_quintile_rets.append(ranked.tail(n)[actual_ret_col].mean())
        daily_ic.append(safe_corr(group['prob_win'], group[actual_ret_col], method="spearman"))

    top_q = float(np.mean(top_quintile_rets)) if top_quintile_rets else 0.0
    bottom_q = float(np.mean(bottom_quintile_rets)) if bottom_quintile_rets else 0.0
    long_short_spread = top_q - bottom_q
    mean_ic = float(np.mean(daily_ic)) if daily_ic else 0.0
    ic_std = float(np.std(daily_ic)) if daily_ic else 0.0
    ic_ir = float(mean_ic / ic_std) if ic_std > 0 else 0.0  # Information Ratio

    metrics["top_quintile_return"] = top_q
    metrics["long_short_spread"] = long_short_spread
    metrics["mean_daily_ic"] = mean_ic
    metrics["ic_information_ratio"] = ic_ir

    # ── الاستراتيجية الفعلية بالفلاتر الصارمة (للمرجع، قد تكون أيامها قليلة) ──
    strategy_rets = []
    selected_days = 0
    for _, group in val_df.groupby('date'):
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
    metrics["validation_days"] = int(val_df['date'].nunique()) if 'date' in val_df.columns else 0

    # ── معيار الاختيار: القيمة الاقتصادية الحقيقية أولاً ──
    # نعتمد على الفرق بين الخُمس الأعلى والأدنى (إشارة دائمة) + IC + AUC،
    # ونضيف عائد الاستراتيجية كمكافأة إن وُجد، بدل أن يكون الأساس الوحيد.
    metrics["score"] = (
        (long_short_spread * 200)
        + (mean_ic * 50)
        + (metrics["auc"] * 10)
        + (metrics["strategy_total_return"] * 20)
    )
    return metrics

def evaluate_regressor(model, val_df, X_val, y_val):
    """تقييم نموذج انحدار العائد بمقاييس صادقة موجهة للعائد.

    ملاحظة منهجية مهمة: المقياس r²/MAPE على *سعر* الإغلاق مضلّل بشدة لأن
    إغلاق الغد ≈ إغلاق اليوم، فيظهر r²≈0.999 دون أي قيمة تنبؤية حقيقية.
    المقياس الصحيح لجودة توقع العائد هو:
      - Information Coefficient (IC): ارتباط Spearman بين العائد المتوقع والفعلي.
      - direction_accuracy: دقة توقع الاتجاه (صعود/هبوط).
    نحتفظ بمقاييس السعر للمرجع فقط (مع وسمها) ولا نبني عليها اختيار النموذج.
    """
    predicted_returns = np.clip(model.predict(X_val), -0.2, 0.2)
    actual_returns = pd.Series(np.asarray(y_val, dtype=float))
    pred_series = pd.Series(np.asarray(predicted_returns, dtype=float))

    direction_actual = actual_returns > 0
    direction_predicted = pred_series.values > 0
    direction_accuracy = float((direction_actual.values == direction_predicted).mean())

    # Information Coefficient — المقياس الذهبي لجودة توقع العائد.
    ic_spearman = safe_corr(pred_series, actual_returns, method="spearman")
    ic_pearson = safe_corr(pred_series, actual_returns, method="pearson")

    # خطأ العائد (وليس السعر) — المقياس الصحيح لحجم الخطأ.
    return_mae = float(mean_absolute_error(actual_returns, pred_series))
    return_rmse = float(np.sqrt(mean_squared_error(actual_returns, pred_series)))

    # مقاييس السعر للمرجع فقط — مضلّلة ولا تُستخدم في الاختيار.
    current_prices = val_df['close'].replace(0, np.nan).reset_index(drop=True)
    actual_close = val_df['target_next_day_close'].reset_index(drop=True)
    price_preds = np.maximum(current_prices * (1 + pred_series.values), 0.01)
    price_r2_inflated = float(r2_score(actual_close, price_preds))

    return {
        "ic_spearman": ic_spearman,
        "ic_pearson": ic_pearson,
        "direction_accuracy": direction_accuracy,
        "return_mae": return_mae,
        "return_rmse": return_rmse,
        "price_r2_inflated_ref_only": price_r2_inflated,
        # الاختيار مبني على القيمة التنبؤية الحقيقية: IC + دقة الاتجاه.
        "score": float((ic_spearman * 100) + (direction_accuracy - 0.5) * 100),
    }

def walk_forward_validation(df, features, target, config, n_windows=4, embargo_days=LABEL_HORIZON_DAYS):
    """تحقق walk-forward متعدد النوافذ بفجوة purge/embargo.

    بدل فصل واحد 80/20، نقسّم المحور الزمني إلى ``n_windows`` نافذة متتابعة.
    في كل نافذة: ندرّب على كل ما سبقها (مع حذف آخر embargo_days)، ونقيّم على
    النافذة نفسها. هذا يكشف هل الحافة التنبؤية ثابتة عبر الزمن أم وليدة فترة
    واحدة. نُرجع متوسط/انحراف AUC و IO و long-short spread عبر النوافذ.
    """
    target_df = df.dropna(subset=[target]).sort_values('date').copy()
    dates = np.array(sorted(target_df['date'].unique()))
    if len(dates) < n_windows * 3:
        return {"windows": 0, "note": "بيانات غير كافية للتحقق المتعدد"}

    fold_edges = np.array_split(dates, n_windows + 1)
    aucs, ics, spreads = [], [], []
    for w in range(1, n_windows + 1):
        val_dates = fold_edges[w]
        train_cutoff_idx = np.searchsorted(dates, val_dates[0]) - embargo_days
        if train_cutoff_idx < 30:
            continue
        train_dates = dates[:train_cutoff_idx]
        tr = target_df[target_df['date'].isin(train_dates)]
        va = target_df[target_df['date'].isin(val_dates)]
        if tr.empty or va.empty or tr[target].nunique() < 2:
            continue
        _, X_tr, y_tr, medians = prepare_frame(tr, features, target)
        va_clean, X_va, y_va, _ = prepare_frame(va, features, target, medians)
        if y_va.nunique() < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
            l2_regularization=1.5, random_state=42, class_weight='balanced'
        ).fit(X_tr, y_tr)
        m = evaluate_model(model, va_clean, X_va, y_va, config)
        aucs.append(m["auc"])
        ics.append(m["mean_daily_ic"])
        spreads.append(m["long_short_spread"])

    if not aucs:
        return {"windows": 0, "note": "تعذّر تقييم أي نافذة"}
    return {
        "windows": len(aucs),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "ic_mean": float(np.mean(ics)), "ic_std": float(np.std(ics)),
        "spread_mean": float(np.mean(spreads)), "spread_std": float(np.std(spreads)),
        # هل الحافة ثابتة؟ نسبة النوافذ ذات IC موجب.
        "positive_ic_ratio": float(np.mean([1 if v > 0 else 0 for v in ics])),
    }

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
        'price_vs_high_20d', 'stoch_rsi_k', 'stoch_rsi_d', 'atr_slope',
        # ميزات microstructure (C2)
        'close_location_value', 'gap_open', 'intraday_range_pct',
        'close_vs_open', 'upper_shadow_pct', 'lower_shadow_pct', 'clv_mean_5d',
        # درجة الانعكاس المركّبة (الإشارة المكتشفة عبر بحث IC)
        'reversal_score'
    ]
    
    # تنظيف الميزات
    for col in ['pe_ratio', 'div_yield', 'beta', 'oil_correlation', 'sentiment']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    features = [f for f in features if f in df.columns]
    price_regression_features = features + [
        col for col in (
            [f'close_lag_{lag}' for lag in range(1, PRICE_LAG_DAYS + 1)] +
            [f'volume_lag_{lag}' for lag in range(1, PRICE_LAG_DAYS + 1)] +
            [f'return_lag_{lag}' for lag in range(1, PRICE_LAG_DAYS + 1)] +
            ['close_mean_10d', 'close_std_10d', 'volume_mean_10d', 'return_sum_10d', 'close']
        )
        if col in df.columns and col not in features
    ]
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_metrics = {}
    feature_medians = df[features].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0)
    regression_feature_medians = (
        df[price_regression_features]
        .replace([np.inf, -np.inf], np.nan)
        .median(numeric_only=True)
        .fillna(0)
    )

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
                f"{target} | {name}: AUC={metrics['auc']:.4f}, IC={metrics['mean_daily_ic']:.4f}, "
                f"L/S spread={metrics['long_short_spread']:.4f}, TopQ={metrics['top_quintile_return']:.4f}, "
                f"StrategyDays={metrics['selected_days']}/{metrics['validation_days']}"
            )
            
            if metrics["score"] > best_score:
                best_name = name
                best_score = metrics["score"]

        # B1: معايرة الاحتمالات. ندرّب النموذج الأساس على أول 85% زمنياً ثم
        # نعايره (isotonic) على آخر 15% (prefit) لاحترام الترتيب الزمني،
        # بحيث تصبح p_weekly احتمالات حقيقية موثوقة لعتبات الواجهة.
        _, X_all, y_all, _ = prepare_frame(target_df, features, target, feature_medians)
        calib_fit_df, calib_hold_df = chronological_split(target_df, validation_ratio=0.15)
        calibrated = None
        if not calib_fit_df.empty and not calib_hold_df.empty:
            _, X_cfit, y_cfit, _ = prepare_frame(calib_fit_df, features, target, feature_medians)
            _, X_chold, y_chold, _ = prepare_frame(calib_hold_df, features, target, feature_medians)
            # نحتاج صنفين على الأقل في عيّنة المعايرة.
            if y_cfit.nunique() > 1 and y_chold.nunique() > 1 and FrozenEstimator is not None:
                base = clone(candidates[best_name]).fit(X_cfit, y_cfit)
                try:
                    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
                    calibrated.fit(X_chold, y_chold)
                except Exception as exc:
                    print(f"تعذّرت المعايرة لـ {target}: {exc}")
                    calibrated = None

        if calibrated is not None:
            joblib.dump(calibrated, output_path)
            calibrated_flag = True
        else:
            # رجوع آمن: نموذج غير معاير مدرَّب على كل البيانات.
            final_model = clone(candidates[best_name]).fit(X_all, y_all)
            joblib.dump(final_model, output_path)
            calibrated_flag = False
        return {"selected_model": best_name, "calibrated": calibrated_flag, "validation": results}

    def train_price_regression_target(target, output_path):
        target_df = df.dropna(subset=[target, 'target_next_day_close']).sort_values('date').copy()
        train_df, val_df = chronological_split(target_df)

        if train_df.empty or val_df.empty:
            print(f"بيانات ناقصة لـ {target}")
            return None

        _, X_train, y_train, medians = prepare_frame(train_df, price_regression_features, target)
        val_clean, X_val, y_val, _ = prepare_frame(val_df, price_regression_features, target, medians)

        candidates = get_candidate_regressors()
        results = {}
        best_name = None
        best_score = -np.inf

        for name, candidate in candidates.items():
            model = clone(candidate)
            model.fit(X_train, y_train)
            metrics = evaluate_regressor(model, val_clean, X_val, y_val)
            results[name] = metrics
            print(
                f"{target} | {name}: IC={metrics['ic_spearman']:.4f}, "
                f"Direction={metrics['direction_accuracy']:.2%}, "
                f"ReturnMAE={metrics['return_mae']:.4f} "
                f"(price_r2={metrics['price_r2_inflated_ref_only']:.3f} ← مضلّل)"
            )

            if metrics["score"] > best_score:
                best_name = name
                best_score = metrics["score"]

        # D2: split conformal prediction. نحسب كمية المتبقيات على عيّنة معايرة
        # زمنية لإعطاء فترة ثقة حقيقية للعائد: [pred - qhat, pred + qhat]
        # بتغطية ~80% و~90%، بدل رقم نقطي بلا قياس لعدم اليقين.
        conformal = {}
        conf_fit_df, conf_hold_df = chronological_split(target_df, validation_ratio=0.15)
        if not conf_fit_df.empty and not conf_hold_df.empty:
            _, X_cfit, y_cfit, _ = prepare_frame(conf_fit_df, price_regression_features, target, regression_feature_medians)
            _, X_chold, y_chold, _ = prepare_frame(conf_hold_df, price_regression_features, target, regression_feature_medians)
            cal_model = clone(candidates[best_name]).fit(X_cfit, y_cfit)
            residuals = np.abs(y_chold.values - np.clip(cal_model.predict(X_chold), -0.2, 0.2))
            if len(residuals) > 0:
                conformal = {
                    "qhat_80": float(np.quantile(residuals, 0.80)),
                    "qhat_90": float(np.quantile(residuals, 0.90)),
                    "n_calibration": int(len(residuals)),
                }

        final_model = clone(candidates[best_name])
        _, X_all, y_all, _ = prepare_frame(target_df, price_regression_features, target, regression_feature_medians)
        final_model.fit(X_all, y_all)
        joblib.dump(final_model, output_path)
        if conformal:
            joblib.dump(conformal, os.path.join(MODEL_DIR, "regression_conformal.joblib"))
        return {"selected_model": best_name, "conformal": conformal, "validation": results}

    print("--- تدريب نموذج انحدار سعر اليوم التالي (آخر أسبوعين) ---")
    model_metrics["next_day_price_regression"] = train_price_regression_target(
        'target_next_day_return',
        os.path.join(MODEL_DIR, "tasi_reg_model_next_day.joblib")
    )
    
    # 1. تدريب النموذج اليومي (Alpha Outperformer)
    print("--- تدريب النموذج اليومي ---")
    model_metrics["daily"] = train_one_target('target_outperformer_1d', os.path.join(MODEL_DIR, "tasi_rf_model_daily.joblib"))
    
    # 2. تدريب النموذج الأسبوعي (Top 25% Winner)
    print("\n--- تدريب النموذج الأسبوعي (5 أيام) ---")
    model_metrics["weekly"] = train_one_target('target_alpha_weekly', os.path.join(MODEL_DIR, "tasi_rf_model_weekly.joblib"))

    # B2: تحقق walk-forward متعدد النوافذ لقياس ثبات الحافة عبر الزمن.
    print("\n--- تحقق Walk-Forward متعدد النوافذ (الأسبوعي) ---")
    wf = walk_forward_validation(df, features, 'target_alpha_weekly', config)
    model_metrics["weekly_walk_forward"] = wf
    if wf.get("windows", 0) > 0:
        print(
            f"نوافذ={wf['windows']} | AUC={wf['auc_mean']:.4f}±{wf['auc_std']:.4f} | "
            f"IC={wf['ic_mean']:.4f}±{wf['ic_std']:.4f} | "
            f"L/S={wf['spread_mean']:.4f} | نوافذ IC موجب={wf['positive_ic_ratio']:.0%}"
        )
    else:
        print(wf.get("note", "تعذّر التحقق المتعدد"))
    
    # 3. تدريب النموذج المتوسط (أسبوعين)
    print("\n--- تدريب النموذج المتوسط (10 أيام) ---")
    model_metrics["medium"] = train_one_target('target_alpha_medium', os.path.join(MODEL_DIR, "tasi_rf_model_medium.joblib"))

    # حفظ الأسماء والبيانات
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(feature_medians, os.path.join(MODEL_DIR, "feature_medians.joblib"))
    joblib.dump(price_regression_features, os.path.join(MODEL_DIR, "regression_feature_names.joblib"))
    joblib.dump(regression_feature_medians, os.path.join(MODEL_DIR, "regression_feature_medians.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\nتم بنجاح تدريب وحفظ جميع نماذج Alpha V6.0.")

if __name__ == "__main__":
    train_tasi_models("data/tasi_processed.csv")
