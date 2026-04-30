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
BACKTEST_PATH = os.path.join(ROOT_DIR, "data/backtest_results.csv")

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
        return None, None, None, None
    try:
        m_daily = joblib.load(MODEL_DAILY_PATH)
        m_weekly = joblib.load(MODEL_WEEKLY_PATH)
        features = joblib.load(FEATURES_PATH)
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df['اسم الشركة'] = df['symbol'].map(ARABIC_NAMES).fillna(df['company name'])
        return df, m_daily, m_weekly, features
    except: return None, None, None, None

def get_xai_reason(row):
    reasons = []
    if row['rsi'] > 60: reasons.append("زخم سعري (RSI)")
    if row['sentiment'] > 0.7: reasons.append("أخبار إيجابية")
    if row['tv_signal'] >= 1: reasons.append("توصية TradingView")
    if row['oil_correlation'] > 0.6: reasons.append("دعم النفط")
    return " + ".join(reasons[:2]) if reasons else "نمط فني صاعد"

def get_market_intelligence():
    df, m_daily, m_weekly, features = load_system_assets()
    if df is None: return []

    latest_data = []
    for (symbol, name), group in df.groupby(['symbol', 'اسم الشركة']):
        group = group.sort_values('date')
        latest_row = group.iloc[-1:].copy()
        
        X = latest_row[features].fillna(0)
        p_daily = float(m_daily.predict(X)[0])
        p_weekly = float(m_weekly.predict(X)[0])
        
        # حماية ضد القيم المفقودة
        current_price = float(latest_row['close'].values[0]) if not pd.isna(latest_row['close'].values[0]) else 0
        atr = float(latest_row['atr'].values[0]) if 'atr' in latest_row.columns and not pd.isna(latest_row['atr'].values[0]) else 0
        sentiment_val = float(latest_row['sentiment'].values[0]) if 'sentiment' in latest_row.columns and not pd.isna(latest_row['sentiment'].values[0]) else 0.5
        current_sector = latest_row['sector'].values[0] if 'sector' in latest_row.columns else "عام"
        
        # حساب الأهداف السعرية
        target_daily = current_price * (1 + p_daily)
        target_weekly = current_price * (1 + p_weekly)
        stop_price = current_price - (1.5 * atr)
        
        # التأكد من أن الأرقام صالحة وليست nan
        target_daily = target_daily if not np.isnan(target_daily) else current_price * 1.01
        target_weekly = target_weekly if not np.isnan(target_weekly) else current_price * 1.03
        stop_price = stop_price if not np.isnan(stop_price) else current_price * 0.97

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
            'target_daily': round(target_daily, 2),
            'target_weekly': round(target_weekly, 2),
            'stop': round(stop_price, 2),
            'sharpe': round(float(latest_row['sharpe_ratio_rolling'].values[0]), 2) if 'sharpe_ratio_rolling' in latest_row.columns else 0
        })
    
    return sorted(latest_data, key=lambda x: x['predicted_daily'], reverse=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    df, _, _, _ = load_system_assets()
    intelligence = get_market_intelligence()
    
    # تحليلات إضافية للواجهة
    sectors = {}
    if intelligence:
        pdf = pd.DataFrame(intelligence)
        sectors = pdf.groupby('sector')['predicted_daily'].mean().to_dict()

    data_date = df['date'].max().strftime('%Y-%m-%d') if df is not None else "N/A"
    
    backtest = None
    if os.path.exists(BACKTEST_PATH):
        bt_df = pd.read_csv(BACKTEST_PATH)
        backtest = {'return': round(float(bt_df['value'].iloc[-1] - 100.0), 2)}

    return templates.TemplateResponse(
        request=request, name="index.html",
        context={
            "request": request, "predictions": intelligence, "top_3": intelligence[:3],
            "sectors": sectors, "backtest": backtest, "data_date": data_date,
            "last_update": datetime.now().strftime("%H:%M")
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
