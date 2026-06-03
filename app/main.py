from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Dict
from scripts.company_names import get_arabic_company_name
from scripts.paper_portfolio import reset_portfolio, sync_portfolio_with_recommendations
from scripts.strategy_config import load_strategy_config
from scripts.strategy_rules import add_hybrid_scores, entry_diagnostics, filter_main_market
import json

app = FastAPI(title="منصة تاسي الذكية", description="منصة تحليل احتمالي لأسهم السوق السعودي الرئيسي")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DAILY_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_daily.joblib")
MODEL_WEEKLY_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_weekly.joblib")
MODEL_MEDIUM_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_medium.joblib")
REGRESSION_NEXT_DAY_PATH = os.path.join(ROOT_DIR, "models/tasi_reg_model_next_day.joblib")
DATA_PATH = os.path.join(ROOT_DIR, "data/tasi_processed.csv")
FEATURES_PATH = os.path.join(ROOT_DIR, "models/feature_names.joblib")
FEATURE_MEDIANS_PATH = os.path.join(ROOT_DIR, "models/feature_medians.joblib")
REGRESSION_FEATURES_PATH = os.path.join(ROOT_DIR, "models/regression_feature_names.joblib")
REGRESSION_FEATURE_MEDIANS_PATH = os.path.join(ROOT_DIR, "models/regression_feature_medians.joblib")
BACKTEST_PATH = os.path.join(ROOT_DIR, "data/backtest_results.csv")
PAPER_PORTFOLIO_PATH = os.path.join(ROOT_DIR, "data/paper_portfolio.json")
RECOMMENDATION_LOG_PATH = os.path.join(ROOT_DIR, "data/recommendation_log.json")

HEATMAP_LIMIT = int(os.getenv("TASI_HEATMAP_LIMIT", "60"))
MARKET_SCOPE_LABEL = "السوق السعودي الرئيسي فقط"
NOMU_PREFIXES = ("95", "96")
STRATEGY_CONFIG = load_strategy_config()
CONFIDENCE_THRESHOLD_DAILY = 0.65
CONFIDENCE_THRESHOLD_WEEKLY = STRATEGY_CONFIG["buy_prob_threshold"]
CONFIDENCE_THRESHOLD_MEDIUM = 0.70
MARKET_CLOSE_HOUR = int(os.getenv("TASI_MARKET_CLOSE_HOUR", "15"))
MARKET_CLOSE_MINUTE = int(os.getenv("TASI_MARKET_CLOSE_MINUTE", "20"))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

def load_system_assets():
    if not all(os.path.exists(p) for p in [MODEL_DAILY_PATH, MODEL_WEEKLY_PATH, DATA_PATH]):
        return None, None, None, None, None, None, None, None, None
    try:
        m_daily = joblib.load(MODEL_DAILY_PATH)
        m_weekly = joblib.load(MODEL_WEEKLY_PATH)
        m_medium = joblib.load(MODEL_MEDIUM_PATH) if os.path.exists(MODEL_MEDIUM_PATH) else None
        m_regression = joblib.load(REGRESSION_NEXT_DAY_PATH) if os.path.exists(REGRESSION_NEXT_DAY_PATH) else None
        features = joblib.load(FEATURES_PATH)
        feature_medians = joblib.load(FEATURE_MEDIANS_PATH) if os.path.exists(FEATURE_MEDIANS_PATH) else None
        regression_features = joblib.load(REGRESSION_FEATURES_PATH) if os.path.exists(REGRESSION_FEATURES_PATH) else None
        regression_feature_medians = (
            joblib.load(REGRESSION_FEATURE_MEDIANS_PATH)
            if os.path.exists(REGRESSION_FEATURE_MEDIANS_PATH)
            else None
        )
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = filter_main_market(df)
        df['اسم الشركة'] = df.apply(
            lambda row: get_arabic_company_name(
                row['symbol'],
                row.get('company name arabic') or row.get('company name', "")
            ),
            axis=1
        )
        return df, m_daily, m_weekly, m_medium, features, feature_medians, m_regression, regression_features, regression_feature_medians
    except: return None, None, None, None, None, None, None, None, None

def prepare_prediction_features(rows, features, medians):
    X = rows[features].replace([np.inf, -np.inf], np.nan)
    if medians is not None:
        X = X.fillna(medians)
    return X.fillna(0)

def get_xai_reason(row):
    reasons = []
    if row['rsi'] > 60: reasons.append("زخم سعري قوي")
    if row['sentiment'] > 0.7: reasons.append("تدفق أخبار إيجابي")
    if row['tv_signal'] >= 1: reasons.append("إجماع فني (T.View)")
    if row['relative_sector_alpha'] > 0: reasons.append("أداء أقوى من القطاع")
    return " + ".join(reasons[:2]) if reasons else "قوة نسبية متزايدة"

def latest_completed_saudi_trading_day(now=None):
    current = now or datetime.now()
    trade_day = pd.Timestamp(current).normalize()
    close_time = current.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    if current < close_time:
        trade_day -= pd.Timedelta(days=1)

    while trade_day.weekday() in (4, 5):  # Friday, Saturday
        trade_day -= pd.Timedelta(days=1)
    return trade_day

def get_prediction_base_date(df):
    completed_day = latest_completed_saudi_trading_day()
    dates = sorted(pd.Timestamp(date_value) for date_value in df['date'].dropna().unique())
    if not dates:
        return None
    eligible_dates = [date_value for date_value in dates if date_value <= completed_day]
    if eligible_dates:
        return eligible_dates[-1]
    return dates[0]

def build_trade_setup(current_price, atr, p_daily, p_weekly, p_medium, is_liquid=True, return_5d=0, hybrid_rank=999, rsi=50, adx=0, above_sma20=False, predicted_next_close=None, predicted_next_return=None):
    valid_price = current_price > 0
    valid_atr = atr > 0 and valid_price
    atr_risk = atr if valid_atr else current_price * 0.02
    stop_distance = STRATEGY_CONFIG["stop_loss_atr_mult"] * atr_risk
    target_distance = STRATEGY_CONFIG["take_profit_atr_mult"] * stop_distance

    daily_target = predicted_next_close if valid_price and predicted_next_close and predicted_next_close > 0 else None
    weekly_target = current_price + max(current_price * 0.03, target_distance) if valid_price and p_weekly > CONFIDENCE_THRESHOLD_WEEKLY else None
    medium_target = current_price + max(current_price * 0.06, target_distance * 1.5) if valid_price and p_medium and p_medium > CONFIDENCE_THRESHOLD_MEDIUM else None

    stop_price = max(0.01, current_price - stop_distance) if valid_price else 0
    
    # حماية ضد الشراء في القمة أو محاولة التقاط سكين هابط
    is_not_overextended = STRATEGY_CONFIG["min_entry_return_5d"] <= return_5d <= STRATEGY_CONFIG["max_entry_return_5d"]
    min_next_return = STRATEGY_CONFIG.get("min_predicted_next_return", 0.0)
    has_safe_next_day_forecast = predicted_next_return is None or predicted_next_return >= min_next_return
    passes_technical_gate = (
        hybrid_rank <= STRATEGY_CONFIG["max_entry_rank"] and
        STRATEGY_CONFIG["min_entry_rsi"] <= rsi <= STRATEGY_CONFIG["max_entry_rsi"] and
        adx >= STRATEGY_CONFIG["min_entry_adx"] and
        above_sma20
    )
    
    is_actionable = bool(valid_price and is_liquid and is_not_overextended and has_safe_next_day_forecast and passes_technical_gate and (p_weekly > CONFIDENCE_THRESHOLD_WEEKLY))

    if not valid_price:
        action = "بيانات سعر غير كافية"
        action_class = "avoid"
    elif not is_liquid:
        action = "سيولة منخفضة"
        action_class = "avoid"
    elif not is_not_overextended:
        action = "متضخم سعرياً"
        action_class = "avoid"
    elif not has_safe_next_day_forecast:
        action = "انتظار: توقع الغد سلبي"
        action_class = "avoid"
    elif is_actionable and p_weekly > 0.60:
        action = "شراء عالي الجودة"
        action_class = "buy"
    elif is_actionable:
        action = "شراء مشروط"
        action_class = "buy"
    elif p_daily > CONFIDENCE_THRESHOLD_DAILY:
        action = "مراقبة زخم يومي"
        action_class = "watch"
    else:
        action = "انتظار"
        action_class = "avoid"

    return {
        "target_daily": daily_target,
        "target_weekly": weekly_target,
        "target_medium": medium_target,
        "stop": stop_price,
        "is_actionable": is_actionable,
        "action": action,
        "action_class": action_class,
    }

def get_market_intelligence(prediction_date_override=None):
    df, m_daily, m_weekly, m_medium, features, feature_medians, m_regression, regression_features, regression_feature_medians = load_system_assets()
    if df is None: return []

    latest_data = []
    prediction_date = pd.Timestamp(prediction_date_override) if prediction_date_override else get_prediction_base_date(df)
    if prediction_date is None:
        return []
    
    # جلب حالة السوق العامة
    market_row = df[df['date'] == prediction_date]
    market_breadth = market_row['market_breadth_sma50'].iloc[0] if 'market_breadth_sma50' in market_row.columns else 0.5
    
    # 1. استخراج صف آخر إغلاق مكتمل لكل سهم دفعة واحدة
    latest_rows = df[df['date'] <= prediction_date].sort_values('date').groupby(['symbol', 'اسم الشركة']).tail(1).copy()
    
    # 2. تجهيز الميزات للجميع مرة واحدة
    X_all = prepare_prediction_features(latest_rows, features, feature_medians)
    
    # 3. التنبؤ للجميع (Probabilities)
    latest_rows['p_daily'] = m_daily.predict_proba(X_all)[:, 1]
    latest_rows['p_weekly'] = m_weekly.predict_proba(X_all)[:, 1]
    latest_rows['p_medium'] = m_medium.predict_proba(X_all)[:, 1] if m_medium else 0.5
    if m_regression is not None and regression_features:
        X_regression = prepare_prediction_features(latest_rows, regression_features, regression_feature_medians)
        latest_rows['predicted_next_return'] = np.clip(m_regression.predict(X_regression), -0.2, 0.2)
        latest_rows['predicted_next_close'] = np.maximum(
            latest_rows['close'].replace(0, np.nan) * (1 + latest_rows['predicted_next_return']),
            0.01
        )
        latest_rows['p_daily'] = (0.5 + latest_rows['predicted_next_return'].fillna(0) * 10).clip(0, 1)
    else:
        latest_rows['predicted_next_close'] = np.nan
        latest_rows['predicted_next_return'] = np.nan
    latest_rows['prob_win'] = latest_rows['p_weekly']
    latest_rows = add_hybrid_scores(latest_rows)
    latest_rows['hybrid_rank'] = latest_rows['rank']
    
    for _, row in latest_rows.iterrows():
        symbol = row['symbol']
        name = row['اسم الشركة']
        p_daily = float(row['p_daily'])
        p_weekly = float(row['p_weekly'])
        p_medium = float(row['p_medium'])
        
        # حماية ضد القيم المفقودة
        current_price = float(row['close']) if not pd.isna(row['close']) else 0
        predicted_next_close = (
            float(row['predicted_next_close'])
            if 'predicted_next_close' in latest_rows.columns and not pd.isna(row['predicted_next_close'])
            else None
        )
        predicted_next_return = (
            float(row['predicted_next_return'])
            if 'predicted_next_return' in latest_rows.columns and not pd.isna(row['predicted_next_return'])
            else None
        )
        atr = float(row['atr']) if 'atr' in latest_rows.columns and not pd.isna(row['atr']) else 0
        sentiment_val = float(row['sentiment']) if 'sentiment' in latest_rows.columns and not pd.isna(row['sentiment']) else 0.5
        ret_5d = float(row['return_5d']) if 'return_5d' in latest_rows.columns else 0
        current_sector = row['sector'] if 'sector' in latest_rows.columns else "عام"
        avg_traded_value = (
            float(row['avg_traded_value_20d'])
            if 'avg_traded_value_20d' in latest_rows.columns and not pd.isna(row['avg_traded_value_20d'])
            else 0
        )
        is_liquid = avg_traded_value >= STRATEGY_CONFIG["min_avg_traded_value"]
        
        setup = build_trade_setup(
            current_price, atr, p_daily, p_weekly, p_medium, is_liquid, ret_5d,
            hybrid_rank=float(row['hybrid_rank']),
            rsi=float(row['rsi']) if 'rsi' in latest_rows.columns and not pd.isna(row['rsi']) else 50,
            adx=float(row['tv_adx']) if 'tv_adx' in latest_rows.columns and not pd.isna(row['tv_adx']) else 0,
            above_sma20=bool(row['close'] > row['sma_20']) if 'sma_20' in latest_rows.columns else False,
            predicted_next_close=predicted_next_close,
            predicted_next_return=predicted_next_return
        )
        
        # تعديل الحالة بناءً على وضع السوق
        action = setup['action']
        is_actionable = setup['is_actionable']
        action_class = setup['action_class']
        
        if market_breadth < STRATEGY_CONFIG["min_entry_market_breadth"] and is_actionable:
            action = "تحذير: ضعف عام"
            is_actionable = False
            action_class = "avoid"
        elif market_breadth < STRATEGY_CONFIG["market_breadth_exit"]:
            action = "خطر: خروج عام"
            is_actionable = False
            action_class = "avoid"

        stop_price = setup['stop'] if not np.isnan(setup['stop']) else current_price * 0.97
        rejection_reason = entry_diagnostics(row, STRATEGY_CONFIG, market_breadth)

        latest_data.append({
            'company_name': name,
            'symbol': symbol,
            'sector': current_sector,
            'market': row['market'] if 'market' in latest_rows.columns else "السوق الرئيسي",
            'reason': get_xai_reason(row),
            'entry_status': rejection_reason,
            'prediction_date': prediction_date.strftime("%Y-%m-%d"),
            'current_price': round(current_price, 2),
            'sentiment': round(sentiment_val * 100, 1),
            'predicted_daily': round((predicted_next_return or 0) * 100, 2), # العائد المتوقع للغد
            'predicted_next_close': round(predicted_next_close, 2) if predicted_next_close is not None else None,
            'predicted_weekly': round(p_weekly * 100, 1), # ثقة الأسبوعي
            'predicted_medium': round(p_medium * 100, 1), # ثقة أسبوعين
            'entry': round(current_price, 2),
            'target_daily': round(setup['target_daily'], 2) if setup['target_daily'] is not None else None,
            'target_weekly': round(setup['target_weekly'], 2) if setup['target_weekly'] is not None else None,
            'target_medium': round(setup['target_medium'], 2) if setup['target_medium'] is not None else None,
            'stop': round(stop_price, 2),
            'sharpe': round(float(row['sharpe_ratio_rolling']), 2) if 'sharpe_ratio_rolling' in latest_rows.columns else 0,
            'tv_adx': round(float(row['tv_adx']), 1) if 'tv_adx' in latest_rows.columns else 0,
            'is_actionable': is_actionable,
            'action': action,
            'action_class': action_class,
            'is_liquid': is_liquid,
            'avg_traded_value_20d': round(avg_traded_value, 0),
            'market_breadth': market_breadth,
            'hybrid_score': round(float(row['hybrid_score']), 4),
            'hybrid_rank': int(row['hybrid_rank'])
        })
    
    return sorted(
        latest_data,
        key=lambda x: (bool(x['is_actionable']), -x['hybrid_rank'], x['predicted_weekly'] or 0.0),
        reverse=True
    )

def load_recommendation_log():
    if not os.path.exists(RECOMMENDATION_LOG_PATH):
        return []
    try:
        with open(RECOMMENDATION_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def import_recommendations_from_paper_trades(records, df):
    if not os.path.exists(PAPER_PORTFOLIO_PATH):
        return records
    try:
        with open(PAPER_PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            portfolio = json.load(f)
    except (OSError, json.JSONDecodeError):
        return records

    existing_keys = {(item.get("recommendation_date"), item.get("symbol")) for item in records}
    imported = []
    for trade in portfolio.get("trades", []):
        if trade.get("side") != "شراء":
            continue
        rec_date = str(trade.get("date") or "")
        symbol = trade.get("symbol")
        if not rec_date or not symbol or (rec_date, symbol) in existing_keys:
            continue

        trade_date = pd.Timestamp(rec_date)
        history = df[(df["symbol"] == symbol) & (df["date"] <= trade_date)].sort_values("date")
        if history.empty:
            continue
        row = history.iloc[-1]
        entry = float(trade.get("price") or row.get("close") or 0)
        atr = float(row.get("atr") or 0)
        atr_risk = atr if atr > 0 and entry > 0 else entry * 0.02
        stop_distance = STRATEGY_CONFIG["stop_loss_atr_mult"] * atr_risk
        target_distance = STRATEGY_CONFIG["take_profit_atr_mult"] * stop_distance
        target_weekly = entry + max(entry * 0.03, target_distance) if entry > 0 else None
        stop = max(0.01, entry - stop_distance) if entry > 0 else 0

        imported.append({
            "recommendation_date": rec_date,
            "recorded_at": trade.get("timestamp") or "",
            "symbol": symbol,
            "company_name": trade.get("company_name", symbol),
            "sector": row.get("sector", "عام"),
            "entry": round(entry, 2),
            "target_daily": None,
            "target_weekly": round(target_weekly, 2) if target_weekly else None,
            "stop": round(stop, 2),
            "predicted_daily": 0,
            "predicted_weekly": 0,
            "hybrid_rank": 999,
            "action": trade.get("reason", "شراء"),
            "source": "paper_portfolio",
        })

    if not imported:
        return records

    combined_records = records + imported
    combined_records.sort(key=lambda item: (item.get("recommendation_date", ""), item.get("hybrid_rank", 999)))
    return combined_records

def get_recommendation_evaluations():
    df, _, _, _, _, _, _, _, _ = load_system_assets()
    if df is None or df.empty:
        return [], {}

    dates = sorted(pd.Timestamp(value) for value in df['date'].dropna().unique())
    completed_day = latest_completed_saudi_trading_day()
    completed_dates = [date_value for date_value in dates if date_value <= completed_day]
    latest_close_date = completed_dates[-1] if completed_dates else (dates[-1] if dates else None)
    previous_close_date = None
    if latest_close_date is not None:
        previous_dates = [date_value for date_value in dates if date_value < latest_close_date]
        previous_close_date = previous_dates[-1] if previous_dates else None

    base_summary = {
        "count": 0,
        "completed_count": 0,
        "pending_count": 0,
        "hit_count": 0,
        "progress_count": 0,
        "away_count": 0,
        "stop_count": 0,
        "latest_cycle_count": 0,
        "latest_cycle_completed_count": 0,
        "latest_recommendation_date": previous_close_date.strftime("%Y-%m-%d") if previous_close_date is not None else None,
        "latest_evaluation_close_date": latest_close_date.strftime("%Y-%m-%d") if latest_close_date is not None else None,
    }

    records = load_recommendation_log()
    records = import_recommendations_from_paper_trades(records, df)
    if not records:
        return [], base_summary

    rows = []
    for rec in records:
        rec_date = pd.Timestamp(rec.get("recommendation_date"))
        symbol = rec.get("symbol")
        next_dates = [date_value for date_value in dates if date_value > rec_date]
        next_date = next_dates[0] if next_dates else None
        entry = float(rec.get("entry") or 0)
        target = rec.get("target_daily") or rec.get("target_weekly")
        target = float(target) if target else None
        stop = float(rec.get("stop") or 0)

        evaluation = {
            **rec,
            "next_date": next_date.strftime("%Y-%m-%d") if next_date is not None else None,
            "next_close": None,
            "next_high": None,
            "next_low": None,
            "return_pct": None,
            "target_gap_pct": None,
            "status": "بانتظار إغلاق الجلسة التالية",
            "status_class": "pending",
        }

        if next_date is not None:
            match = df[(df['symbol'] == symbol) & (df['date'] == next_date)]
            if not match.empty:
                row = match.iloc[0]
                next_close = float(row['close'])
                next_high = float(row['high']) if 'high' in match.columns and not pd.isna(row['high']) else next_close
                next_low = float(row['low']) if 'low' in match.columns and not pd.isna(row['low']) else next_close
                return_pct = ((next_close / entry) - 1) * 100 if entry else 0.0

                evaluation.update({
                    "next_close": round(next_close, 2),
                    "next_high": round(next_high, 2),
                    "next_low": round(next_low, 2),
                    "return_pct": round(return_pct, 2),
                })

                if target:
                    gap = ((target / next_close) - 1) * 100 if next_close else 0.0
                    evaluation["target_gap_pct"] = round(gap, 2)
                    if next_high >= target:
                        evaluation["status"] = "تحقق الهدف"
                        evaluation["status_class"] = "hit"
                    elif stop > 0 and next_low <= stop:
                        evaluation["status"] = "ضرب وقف الخسارة"
                        evaluation["status_class"] = "stop"
                    elif next_close > entry:
                        evaluation["status"] = "ارتفع ويتجه للهدف"
                        evaluation["status_class"] = "progress"
                    else:
                        evaluation["status"] = "ابتعد عن الهدف"
                        evaluation["status_class"] = "away"
                elif next_close > entry:
                    evaluation["status"] = "ارتفع بدون هدف محفوظ"
                    evaluation["status_class"] = "progress"
                else:
                    evaluation["status"] = "تراجع بدون هدف محفوظ"
                    evaluation["status_class"] = "away"

        rows.append(evaluation)

    rows = sorted(rows, key=lambda item: (item.get("recommendation_date") or "", item.get("hybrid_rank") or 999), reverse=True)
    completed = [item for item in rows if item.get("next_close") is not None]
    latest_cycle = [
        item for item in rows
        if item.get("recommendation_date") == base_summary["latest_recommendation_date"]
    ]
    latest_cycle_completed = [item for item in latest_cycle if item.get("next_close") is not None]
    summary = {
        **base_summary,
        "count": len(rows),
        "completed_count": len(completed),
        "pending_count": len(rows) - len(completed),
        "hit_count": sum(1 for item in rows if item.get("status_class") == "hit"),
        "progress_count": sum(1 for item in rows if item.get("status_class") == "progress"),
        "away_count": sum(1 for item in rows if item.get("status_class") == "away"),
        "stop_count": sum(1 for item in rows if item.get("status_class") == "stop"),
        "latest_cycle_count": len(latest_cycle),
        "latest_cycle_completed_count": len(latest_cycle_completed),
    }
    return rows, summary

def get_market_prices(mode="previous_close"):
    df, _, _, _, _, _, _, _, _ = load_system_assets()
    if df is None or df.empty:
        return None, [], {}

    available_dates = sorted(df['date'].dropna().unique())
    if not available_dates:
        return None, [], {}

    if mode == "current":
        selected_date = available_dates[-1]
    else:
        selected_date = get_prediction_base_date(df)

    selected_date = pd.Timestamp(selected_date)
    latest_rows = df[df['date'] == selected_date].copy()
    if latest_rows.empty:
        return None, [], {}

    latest_rows['اسم الشركة'] = latest_rows.apply(
        lambda row: get_arabic_company_name(
            row['symbol'],
            row.get('company name arabic') or row.get('company name', "")
        ),
        axis=1
    )

    if 'traded_value' not in latest_rows.columns:
        latest_rows['traded_value'] = latest_rows['close'] * latest_rows['volume']

    closes = []
    for _, row in latest_rows.sort_values('symbol').iterrows():
        daily_return = float(row['daily_return']) if 'daily_return' in latest_rows.columns and not pd.isna(row['daily_return']) else 0.0
        volume = float(row['volume']) if 'volume' in latest_rows.columns and not pd.isna(row['volume']) else 0.0
        traded_value = float(row['traded_value']) if not pd.isna(row.get('traded_value', np.nan)) else 0.0
        close = float(row['close']) if not pd.isna(row['close']) else 0.0
        open_price = float(row['open']) if 'open' in latest_rows.columns and not pd.isna(row['open']) else close
        high = float(row['high']) if 'high' in latest_rows.columns and not pd.isna(row['high']) else close
        low = float(row['low']) if 'low' in latest_rows.columns and not pd.isna(row['low']) else close

        closes.append({
            "symbol": row['symbol'],
            "company_name": row['اسم الشركة'],
            "sector": row['sector'] if 'sector' in latest_rows.columns else "عام",
            "market": row['market'] if 'market' in latest_rows.columns else MARKET_SCOPE_LABEL,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "daily_return": round(daily_return * 100, 2),
            "volume": volume,
            "traded_value": traded_value,
        })

    positive_count = sum(1 for item in closes if item["daily_return"] > 0)
    negative_count = sum(1 for item in closes if item["daily_return"] < 0)
    unchanged_count = len(closes) - positive_count - negative_count
    total_traded_value = sum(item["traded_value"] for item in closes)
    best = max(closes, key=lambda item: item["daily_return"], default=None)
    worst = min(closes, key=lambda item: item["daily_return"], default=None)

    summary = {
        "date": selected_date.strftime("%Y-%m-%d"),
        "count": len(closes),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "unchanged_count": unchanged_count,
        "total_traded_value": total_traded_value,
        "best": best,
        "worst": worst,
    }
    return selected_date, closes, summary

def get_market_closes():
    return get_market_prices(mode="previous_close")

def get_current_market_prices():
    return get_market_prices(mode="current")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    df, _, _, _, _, _, _, _, _ = load_system_assets()
    intelligence = get_market_intelligence()
    
    # تحليلات إضافية للواجهة
    sectors = {}
    if intelligence:
        pdf = pd.DataFrame(intelligence)
        sectors = pdf.groupby('sector')['predicted_weekly'].mean().to_dict()
        
    top_3 = [stock for stock in intelligence if stock['is_actionable']][:3]
    heatmap_predictions = intelligence[:HEATMAP_LIMIT]

    data_date = df['date'].max().strftime('%Y-%m-%d') if df is not None else "N/A"
    
    backtest = None
    if os.path.exists(BACKTEST_PATH):
        bt_df = pd.read_csv(BACKTEST_PATH)
        backtest = {'return': round(float(bt_df['value'].iloc[-1] - 100.0), 2)}

    market_breadth = intelligence[0]['market_breadth'] if intelligence else 0.5
    prediction_date = intelligence[0].get('prediction_date') if intelligence else data_date
    paper_portfolio = sync_portfolio_with_recommendations(
        intelligence,
        prediction_date,
        STRATEGY_CONFIG,
        path=PAPER_PORTFOLIO_PATH,
        initial_capital=1000.0,
    ) if intelligence else None

    return templates.TemplateResponse(
        request=request, name="index.html",
        context={
            "request": request, "predictions": intelligence,
            "top_3": top_3, "heatmap_predictions": heatmap_predictions,
            "sectors": sectors, "backtest": backtest, "data_date": data_date,
            "market_breadth": round(market_breadth * 100, 1),
            "coverage_count": len(intelligence),
            "market_scope": MARKET_SCOPE_LABEL,
            "strategy_config": STRATEGY_CONFIG,
            "paper_portfolio": paper_portfolio,
            "prediction_date": prediction_date,
            "last_update": datetime.now().strftime("%H:%M")
        }
    )

@app.get("/recommendation-evaluations", response_class=HTMLResponse)
async def recommendation_evaluations(request: Request):
    evaluations, summary = get_recommendation_evaluations()
    return templates.TemplateResponse(
        request=request, name="recommendation_evaluations.html",
        context={
            "request": request,
            "evaluations": evaluations,
            "summary": summary,
            "market_scope": MARKET_SCOPE_LABEL,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )

@app.get("/market-closes", response_class=HTMLResponse)
async def market_closes(request: Request):
    _, closes, summary = get_market_closes()
    if not summary:
        return JSONResponse(status_code=404, content={"message": "بيانات الإغلاقات غير متاحة حالياً"})

    return templates.TemplateResponse(
        request=request, name="market_closes.html",
        context={
            "request": request,
            "closes": closes,
            "summary": summary,
            "page_title": "إغلاقات السوق",
            "page_subtitle": "إغلاقات اليوم السابق المتاح",
            "table_title": "جدول إغلاقات الجلسة السابقة",
            "table_description": "يعرض أسعار الافتتاح، الأعلى، الأدنى، الإغلاق، التغير، والحجم لآخر جلسة مكتملة قبل بيانات اليوم الحالية.",
            "search_placeholder": "بحث باسم الشركة أو الرمز",
            "market_scope": MARKET_SCOPE_LABEL,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )

@app.get("/current-prices", response_class=HTMLResponse)
async def current_prices(request: Request):
    _, prices, summary = get_current_market_prices()
    if not summary:
        return JSONResponse(status_code=404, content={"message": "بيانات الأسعار الحالية غير متاحة حالياً"})

    return templates.TemplateResponse(
        request=request, name="market_closes.html",
        context={
            "request": request,
            "closes": prices,
            "summary": summary,
            "page_title": "الأسعار الحالية",
            "page_subtitle": "آخر أسعار متاحة أثناء جلسة السوق",
            "table_title": "جدول الأسعار الحالية",
            "table_description": "يعرض آخر أسعار متاحة من ملف البيانات الحالي أثناء عمل السوق، وقد تختلف عن إغلاقات الجلسة السابقة.",
            "search_placeholder": "بحث باسم الشركة أو الرمز",
            "market_scope": MARKET_SCOPE_LABEL,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )

@app.get("/stock/{symbol}", response_class=HTMLResponse)
async def stock_detail(request: Request, symbol: str):
    intelligence = get_market_intelligence()
    stock = next((item for item in intelligence if item['symbol'] == symbol), None)
    
    if not stock:
        return JSONResponse(status_code=404, content={"message": "السهم غير موجود ضمن نطاق السوق الرئيسي"})
        
    # جلب متوسط القطاع للمقارنة
    sector_avg = np.mean([s['predicted_weekly'] for s in intelligence if s['sector'] == stock['sector']])
    
    return templates.TemplateResponse(
        request=request, name="stock_detail.html",
        context={
            "request": request,
            "stock": stock,
            "sector_avg": round(sector_avg, 2),
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    )

@app.post("/api/update")
async def update_system(background_tasks: BackgroundTasks):
    def run():
        try:
            for s in ["fetch_data.py", "preprocess.py", "train_model.py", "backtest.py"]:
                subprocess.run([sys.executable, os.path.join(ROOT_DIR, f"scripts/{s}")], check=True)
        except: pass
    background_tasks.add_task(run)
    return {"status": "جاري التحديث الشامل للأنظمة في الخلفية..."}

@app.post("/api/portfolio/reset")
async def reset_paper_portfolio():
    reset_portfolio(path=PAPER_PORTFOLIO_PATH, initial_capital=1000.0)
    return {"status": "تمت إعادة المحفظة الافتراضية إلى 1,000 ريال."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
