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

app = FastAPI(title="TASI AI Pro Suite", description="منصة التحليل المالي المتطورة")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DAILY_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_daily.joblib")
MODEL_WEEKLY_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model_weekly.joblib")
DATA_PATH = os.path.join(ROOT_DIR, "data/tasi_processed.csv")
FEATURES_PATH = os.path.join(ROOT_DIR, "models/feature_names.joblib")
FEATURE_MEDIANS_PATH = os.path.join(ROOT_DIR, "models/feature_medians.joblib")
BACKTEST_PATH = os.path.join(ROOT_DIR, "data/backtest_results.csv")
MIN_DAILY_BUY_RETURN = 0.005
MIN_WEEKLY_BUY_RETURN = 0.018 # المتوافق مع الاختبار العكسي الناجح
MARKET_BREATH_ENTRY = 0.4
MARKET_BREATH_EXIT = 0.2
MIN_AVG_TRADED_VALUE = float(os.getenv("TASI_MIN_AVG_TRADED_VALUE", "1000000"))
HEATMAP_LIMIT = int(os.getenv("TASI_HEATMAP_LIMIT", "60"))

# Arabic Names Mapping
ARABIC_NAMES = {
    "1120.SR": "مصرف الراجحي", "1180.SR": "الأهلي السعودي", "2222.SR": "أرامكو السعودية",
    "2010.SR": "سابك", "7010.SR": "إس تي سي", "1150.SR": "مصرف الإنماء",
    "2350.SR": "كيان السعودية", "2020.SR": "سابك للمغذيات", "4003.SR": "إكسترا",
    "1010.SR": "بنك الرياض", "1111.SR": "تداول السعودية", "7020.SR": "اتحاد اتصالات",
    "5110.SR": "كهرباء السعودية", "2080.SR": "مجموعة تداول", "1211.SR": "معادن",
    "4260.SR": "بدجت السعودية", "4030.SR": "البحري", "2280.SR": "المراعي",
    "4190.SR": "جرير", "1060.SR": "البنك السعودي الفرنسي"
}

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

def load_system_assets():
    if not all(os.path.exists(p) for p in [MODEL_DAILY_PATH, MODEL_WEEKLY_PATH, DATA_PATH]):
        return None, None, None, None, None
    try:
        m_daily = joblib.load(MODEL_DAILY_PATH)
        m_weekly = joblib.load(MODEL_WEEKLY_PATH)
        features = joblib.load(FEATURES_PATH)
        feature_medians = joblib.load(FEATURE_MEDIANS_PATH) if os.path.exists(FEATURE_MEDIANS_PATH) else None
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df['اسم الشركة'] = df['symbol'].map(ARABIC_NAMES).fillna(df['company name'])
        return df, m_daily, m_weekly, features, feature_medians
    except: return None, None, None, None, None

def prepare_prediction_features(rows, features, medians):
    X = rows[features].replace([np.inf, -np.inf], np.nan)
    if medians is not None:
        X = X.fillna(medians)
    return X.fillna(0)

def get_xai_reason(row):
    reasons = []
    if row['rsi'] > 60: reasons.append("زخم سعري (RSI)")
    if row['sentiment'] > 0.7: reasons.append("أخبار إيجابية")
    if row['tv_signal'] >= 1: reasons.append("توصية TradingView")
    if row['oil_correlation'] > 0.6: reasons.append("دعم النفط")
    return " + ".join(reasons[:2]) if reasons else "نمط فني صاعد"

def build_trade_setup(current_price, atr, p_daily, p_weekly, is_liquid=True):
    daily_target = current_price * (1 + p_daily) if p_daily > MIN_DAILY_BUY_RETURN else None
    weekly_target = current_price * (1 + p_weekly) if p_weekly > MIN_WEEKLY_BUY_RETURN else None
    stop_price = current_price - (1.5 * atr) if atr > 0 else current_price * 0.97
    is_actionable = is_liquid and (daily_target is not None or weekly_target is not None)

    if not is_liquid:
        action = "سيولة منخفضة"
        action_class = "avoid"
    elif daily_target is not None:
        action = "شراء"
        action_class = "buy"
    elif weekly_target is not None:
        action = "مراقبة للمدى الأسبوعي"
        action_class = "watch"
    else:
        action = "انتظار"
        action_class = "avoid"

    return {
        "target_daily": daily_target,
        "target_weekly": weekly_target,
        "stop": stop_price,
        "is_actionable": is_actionable,
        "action": action,
        "action_class": action_class,
    }

def get_market_intelligence():
    df, m_daily, m_weekly, features, feature_medians = load_system_assets()
    if df is None: return []

    latest_data = []
    
    # جلب حالة السوق العامة
    current_date_max = df['date'].max()
    market_row = df[df['date'] == current_date_max]
    market_breadth = market_row['market_breadth_sma50'].iloc[0] if 'market_breadth_sma50' in market_row.columns else 0.5
    
    for (symbol, name), group in df.groupby(['symbol', 'اسم الشركة']):
        group = group.sort_values('date')
        latest_row = group.iloc[-1:].copy()
        
        X = prepare_prediction_features(latest_row, features, feature_medians)
        p_daily = float(m_daily.predict(X)[0])
        p_weekly = float(m_weekly.predict(X)[0])
        
        # حماية ضد القيم المفقودة
        current_price = float(latest_row['close'].values[0]) if not pd.isna(latest_row['close'].values[0]) else 0
        atr = float(latest_row['atr'].values[0]) if 'atr' in latest_row.columns and not pd.isna(latest_row['atr'].values[0]) else 0
        sentiment_val = float(latest_row['sentiment'].values[0]) if 'sentiment' in latest_row.columns and not pd.isna(latest_row['sentiment'].values[0]) else 0.5
        current_sector = latest_row['sector'].values[0] if 'sector' in latest_row.columns else "عام"
        avg_traded_value = (
            float(latest_row['avg_traded_value_20d'].values[0])
            if 'avg_traded_value_20d' in latest_row.columns and not pd.isna(latest_row['avg_traded_value_20d'].values[0])
            else 0
        )
        is_liquid = avg_traded_value >= MIN_AVG_TRADED_VALUE
        
        setup = build_trade_setup(current_price, atr, p_daily, p_weekly, is_liquid)
        
        # تعديل الحالة بناءً على وضع السوق
        action = setup['action']
        is_actionable = setup['is_actionable']
        action_class = setup['action_class']
        
        if market_breadth < MARKET_BREATH_ENTRY and is_actionable:
            action = "تحذير: ضعف عام"
            is_actionable = False
            action_class = "avoid"
        elif market_breadth < MARKET_BREATH_EXIT:
            action = "خطر: خروج عام"
            is_actionable = False
            action_class = "avoid"

        stop_price = setup['stop'] if not np.isnan(setup['stop']) else current_price * 0.97

        latest_data.append({
            'company_name': name,
            'symbol': symbol,
            'sector': current_sector,
            'reason': get_xai_reason(latest_row.iloc[0]),
            'current_price': round(current_price, 2),
            'sentiment': round(sentiment_val * 100, 1),
            'predicted_daily': round(p_daily * 100, 2),
            'predicted_weekly': round(p_weekly * 100, 2),
            'entry': round(current_price, 2),
            'target_daily': round(setup['target_daily'], 2) if setup['target_daily'] is not None else None,
            'target_weekly': round(setup['target_weekly'], 2) if setup['target_weekly'] is not None else None,
            'stop': round(stop_price, 2),
            'sharpe': round(float(latest_row['sharpe_ratio_rolling'].values[0]), 2) if 'sharpe_ratio_rolling' in latest_row.columns else 0,
            'is_actionable': is_actionable,
            'action': action,
            'action_class': action_class,
            'is_liquid': is_liquid,
            'avg_traded_value_20d': round(avg_traded_value, 0),
            'market_breadth': market_breadth
        })
    
    return sorted(
        latest_data,
        key=lambda x: (x['is_actionable'], x['predicted_daily'], x['predicted_weekly']),
        reverse=True
    )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    df, _, _, _, _ = load_system_assets()
    intelligence = get_market_intelligence()
    
    # تحليلات إضافية للواجهة
    sectors = {}
    if intelligence:
        pdf = pd.DataFrame(intelligence)
        sectors = pdf.groupby('sector')['predicted_daily'].mean().to_dict()
    top_3 = [stock for stock in intelligence if stock['is_actionable']][:3]
    heatmap_predictions = intelligence[:HEATMAP_LIMIT]

    data_date = df['date'].max().strftime('%Y-%m-%d') if df is not None else "N/A"
    
    backtest = None
    if os.path.exists(BACKTEST_PATH):
        bt_df = pd.read_csv(BACKTEST_PATH)
        backtest = {'return': round(float(bt_df['value'].iloc[-1] - 100.0), 2)}

    market_breadth = intelligence[0]['market_breadth'] if intelligence else 0.5

    return templates.TemplateResponse(
        request=request, name="index.html",
        context={
            "request": request, "predictions": intelligence,
            "top_3": top_3, "heatmap_predictions": heatmap_predictions,
            "sectors": sectors, "backtest": backtest, "data_date": data_date,
            "market_breadth": round(market_breadth * 100, 1),
            "coverage_count": len(intelligence),
            "last_update": datetime.now().strftime("%H:%M")
        }
    )

@app.get("/stock/{symbol}", response_class=HTMLResponse)
async def stock_detail(request: Request, symbol: str):
    intelligence = get_market_intelligence()
    stock = next((item for item in intelligence if item['symbol'] == symbol), None)
    
    if not stock:
        return JSONResponse(status_code=404, content={"message": "Stock not found"})
        
    # جلب متوسط القطاع للمقارنة
    sector_avg = np.mean([s['predicted_daily'] for s in intelligence if s['sector'] == stock['sector']])
    
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
async def update_all(background_tasks: BackgroundTasks):
    def run():
        try:
            for s in ["fetch_data.py", "preprocess.py", "train_model.py", "backtest.py"]:
                subprocess.run([sys.executable, os.path.join(ROOT_DIR, f"scripts/{s}")], check=True)
        except: pass
    background_tasks.add_task(run)
    return {"status": "جاري التحديث الشامل للأنظمة في الخلفية..."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
