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

app = FastAPI(title="TASI AI API", description="API لخدمة توقعات الأسهم السعودية")

# Paths - Robust Path Calculation
# This points to the directory where main.py resides (which is 'app')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This points to the project root (one level up from 'app')
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models/tasi_rf_model.joblib")
DATA_PATH = os.path.join(ROOT_DIR, "data/tasi_processed.csv")
FEATURES_PATH = os.path.join(ROOT_DIR, "models/feature_names.joblib")
BACKTEST_PATH = os.path.join(ROOT_DIR, "data/backtest_results.csv")

# Arabic Names Mapping
ARABIC_NAMES = {
    "1120.SR": "مصرف الراجحي",
    "1180.SR": "البنك الأهلي السعودي",
    "2222.SR": "أرامكو السعودية",
    "2010.SR": "سابك",
    "7010.SR": "إس تي سي (الاتصالات)",
    "1150.SR": "مصرف الإنماء",
    "2350.SR": "كيان السعودية",
    "2020.SR": "سابك للمغذيات الزراعية",
    "4003.SR": "إكسترا",
    "1010.SR": "بنك الرياض"
}

# Templates - reside inside 'app/templates'
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

def load_data_and_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print(f"DEBUG: Missing files - Model: {os.path.exists(MODEL_PATH)}, Data: {os.path.exists(DATA_PATH)}")
        return None, None, None
    
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        df = pd.read_csv(DATA_PATH)
        # Convert to datetime without timezone to match scripts
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df['اسم الشركة'] = df['symbol'].map(ARABIC_NAMES).fillna(df['company name'])
        return df, model, features
    except Exception as e:
        print(f"DEBUG: Error loading model/data: {e}")
        return None, None, None

def get_predictions_logic():
    df, model, features = load_data_and_model()
    if df is None:
        print("DEBUG: Dataframe is None")
        return []

    latest_data = []
    try:
        # Check if features match
        available_features = [f for f in features if f in df.columns]
        if len(available_features) != len(features):
            print(f"DEBUG: Feature mismatch. Expected {len(features)}, got {len(available_features)}")

        for (symbol, name), group in df.groupby(['symbol', 'اسم الشركة']):
            group = group.sort_values('date')
            latest_row = group.iloc[-1:].copy()
            
            X = latest_row[features].fillna(0)
            prediction = float(model.predict(X)[0])
            
            current_price = float(latest_row['close'].values[0])
            atr = float(latest_row['atr'].values[0]) if 'atr' in latest_row.columns else 0
            
            bb_upper = float(latest_row['bbu_20_2.0'].values[0]) if 'bbu_20_2.0' in latest_row.columns else current_price * 1.05
            bb_lower = float(latest_row['bbl_20_2.0'].values[0]) if 'bbl_20_2.0' in latest_row.columns else current_price * 0.95
            
            entry_point = round(current_price, 2)
            tp_target = current_price * (1 + max(0.01, prediction * 1.5))
            take_profit = round(max(tp_target, bb_upper), 2)
            stop_loss = round(min(bb_lower, current_price - (1.5 * atr)), 2)
            
            risk = max(0.01, entry_point - stop_loss)
            reward = take_profit - entry_point
            rr_ratio = round(reward / risk, 2)
            
            sharpe = 0
            if 'sharpe_ratio_rolling' in latest_row.columns:
                val = latest_row['sharpe_ratio_rolling'].values[0]
                sharpe = round(float(val), 2) if not pd.isna(val) else 0

            latest_data.append({
                'company_name': str(name),
                'symbol': str(symbol),
                'date': str(latest_row['date'].dt.strftime('%Y-%m-%d').values[0]),
                'current_price': round(current_price, 2),
                'entry_point': entry_point,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'rr_ratio': rr_ratio,
                'predicted_return': round(prediction * 100, 2),
                'sharpe': sharpe
            })
    except Exception as e:
        print(f"DEBUG: Error generating predictions: {e}")
    
    return sorted(latest_data, key=lambda x: x['predicted_return'], reverse=True)

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    df, _, _ = load_data_and_model()
    predictions = get_predictions_logic()
    
    data_date = "غير متوفر"
    if df is not None and not df.empty:
        data_date = df['date'].max().strftime('%Y-%m-%d')
    
    backtest_info = None
    if os.path.exists(BACKTEST_PATH):
        try:
            bt_df = pd.read_csv(BACKTEST_PATH)
            if not bt_df.empty:
                backtest_info = {
                    'total_return': round(float(bt_df['value'].iloc[-1] - 100.0), 2),
                    'history': bt_df[['date', 'value']].to_dict(orient='records')
                }
        except Exception as e:
            print(f"DEBUG: Error loading backtest: {e}")
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "predictions": predictions, 
            "top_3": predictions[:3],
            "backtest": backtest_info,
            "data_date": data_date,
            "last_update": datetime.now().strftime("%H:%M")
        }
    )

@app.get("/api/predictions")
async def get_predictions():
    return JSONResponse(content=get_predictions_logic())

@app.post("/api/update")
async def update_data(background_tasks: BackgroundTasks):
    def run_scripts():
        try:
            subprocess.run([sys.executable, "scripts/fetch_data.py"], check=True)
            subprocess.run([sys.executable, "scripts/preprocess.py"], check=True)
            subprocess.run([sys.executable, "scripts/train_model.py"], check=True)
            subprocess.run([sys.executable, "scripts/backtest.py"], check=True)
            print("DEBUG: Background Update complete")
        except Exception as e:
            print(f"DEBUG: Background Update failed: {e}")

    background_tasks.add_task(run_scripts)
    return {"status": "جاري تحديث البيانات في الخلفية..."}

@app.get("/api/stock/{symbol}")
async def get_stock_detail(symbol: str):
    df, _, _ = load_data_and_model()
    if df is None:
        return JSONResponse(content={"error": "Data not found"}, status_code=404)
    
    stock_df = df[df['symbol'] == symbol].tail(100)
    if stock_df.empty:
        return JSONResponse(content={"error": "Stock not found"}, status_code=404)
        
    return JSONResponse(content=stock_df.to_dict(orient='records'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
