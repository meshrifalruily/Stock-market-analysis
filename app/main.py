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

app = FastAPI(title="منصة تاسي الذكية", description="منصة تحليل احتمالي لأسهم السوق السعودي الرئيسي")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DAILY_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_daily.joblib")
MODEL_WEEKLY_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_weekly.joblib")
MODEL_MEDIUM_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_medium.joblib")
DATA_PATH = os.path.join(ROOT_DIR, "data/tasi_processed.csv")
FEATURES_PATH = os.path.join(ROOT_DIR, "models/feature_names.joblib")
FEATURE_MEDIANS_PATH = os.path.join(ROOT_DIR, "models/feature_medians.joblib")
BACKTEST_PATH = os.path.join(ROOT_DIR, "data/backtest_results.csv")
PAPER_PORTFOLIO_PATH = os.path.join(ROOT_DIR, "data/paper_portfolio.json")

HEATMAP_LIMIT = int(os.getenv("TASI_HEATMAP_LIMIT", "60"))
MARKET_SCOPE_LABEL = "السوق السعودي الرئيسي فقط"
NOMU_PREFIXES = ("95", "96")
STRATEGY_CONFIG = load_strategy_config()
CONFIDENCE_THRESHOLD_DAILY = 0.65
CONFIDENCE_THRESHOLD_WEEKLY = STRATEGY_CONFIG["buy_prob_threshold"]
CONFIDENCE_THRESHOLD_MEDIUM = 0.70

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

def load_system_assets():
    if not all(os.path.exists(p) for p in [MODEL_DAILY_PATH, MODEL_WEEKLY_PATH, DATA_PATH]):
        return None, None, None, None, None, None
    try:
        m_daily = joblib.load(MODEL_DAILY_PATH)
        m_weekly = joblib.load(MODEL_WEEKLY_PATH)
        m_medium = joblib.load(MODEL_MEDIUM_PATH) if os.path.exists(MODEL_MEDIUM_PATH) else None
        features = joblib.load(FEATURES_PATH)
        feature_medians = joblib.load(FEATURE_MEDIANS_PATH) if os.path.exists(FEATURE_MEDIANS_PATH) else None
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
        return df, m_daily, m_weekly, m_medium, features, feature_medians
    except: return None, None, None, None, None, None

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

def build_trade_setup(current_price, atr, p_daily, p_weekly, p_medium, is_liquid=True, return_5d=0, hybrid_rank=999, rsi=50, adx=0, above_sma20=False):
    valid_price = current_price > 0
    valid_atr = atr > 0 and valid_price
    atr_risk = atr if valid_atr else current_price * 0.02
    stop_distance = STRATEGY_CONFIG["stop_loss_atr_mult"] * atr_risk
    target_distance = STRATEGY_CONFIG["take_profit_atr_mult"] * stop_distance

    daily_target = current_price + max(current_price * 0.01, stop_distance) if valid_price and p_daily > CONFIDENCE_THRESHOLD_DAILY else None
    weekly_target = current_price + max(current_price * 0.03, target_distance) if valid_price and p_weekly > CONFIDENCE_THRESHOLD_WEEKLY else None
    medium_target = current_price + max(current_price * 0.06, target_distance * 1.5) if valid_price and p_medium and p_medium > CONFIDENCE_THRESHOLD_MEDIUM else None

    stop_price = max(0.01, current_price - stop_distance) if valid_price else 0
    
    # حماية ضد الشراء في القمة أو محاولة التقاط سكين هابط
    is_not_overextended = STRATEGY_CONFIG["min_entry_return_5d"] <= return_5d <= STRATEGY_CONFIG["max_entry_return_5d"]
    passes_technical_gate = (
        hybrid_rank <= STRATEGY_CONFIG["max_entry_rank"] and
        STRATEGY_CONFIG["min_entry_rsi"] <= rsi <= STRATEGY_CONFIG["max_entry_rsi"] and
        adx >= STRATEGY_CONFIG["min_entry_adx"] and
        above_sma20
    )
    
    is_actionable = bool(valid_price and is_liquid and is_not_overextended and passes_technical_gate and (p_weekly > CONFIDENCE_THRESHOLD_WEEKLY))

    if not valid_price:
        action = "بيانات سعر غير كافية"
        action_class = "avoid"
    elif not is_liquid:
        action = "سيولة منخفضة"
        action_class = "avoid"
    elif not is_not_overextended:
        action = "متضخم سعرياً"
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

def get_market_intelligence():
    df, m_daily, m_weekly, m_medium, features, feature_medians = load_system_assets()
    if df is None: return []

    latest_data = []
    
    # جلب حالة السوق العامة
    current_date_max = df['date'].max()
    market_row = df[df['date'] == current_date_max]
    market_breadth = market_row['market_breadth_sma50'].iloc[0] if 'market_breadth_sma50' in market_row.columns else 0.5
    
    # 1. استخراج أحدث صف لكل سهم دفعة واحدة
    latest_rows = df.sort_values('date').groupby(['symbol', 'اسم الشركة']).tail(1).copy()
    
    # 2. تجهيز الميزات للجميع مرة واحدة
    X_all = prepare_prediction_features(latest_rows, features, feature_medians)
    
    # 3. التنبؤ للجميع (Probabilities)
    latest_rows['p_daily'] = m_daily.predict_proba(X_all)[:, 1]
    latest_rows['p_weekly'] = m_weekly.predict_proba(X_all)[:, 1]
    latest_rows['p_medium'] = m_medium.predict_proba(X_all)[:, 1] if m_medium else 0.5
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
            above_sma20=bool(row['close'] > row['sma_20']) if 'sma_20' in latest_rows.columns else False
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
            'current_price': round(current_price, 2),
            'sentiment': round(sentiment_val * 100, 1),
            'predicted_daily': round(p_daily * 100, 1), # ثقة اليومي
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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    df, _, _, _, _, _ = load_system_assets()
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
    paper_portfolio = sync_portfolio_with_recommendations(
        intelligence,
        data_date,
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
            "last_update": datetime.now().strftime("%H:%M")
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
