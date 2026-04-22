import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import sys
from datetime import datetime
import subprocess

# Add project root to path for scripts
sys.path.append(os.getcwd())

# Page configuration
st.set_page_config(page_title="TASI Stock Analysis AI", layout="wide")

# Paths
MODEL_PATH = "models/tasi_rf_model.joblib"
DATA_PATH = "data/tasi_processed.csv"
FEATURES_PATH = "models/feature_names.joblib"

def run_update_scripts():
    with st.status("Updating market data...", expanded=True) as status:
        st.write("Fetching latest TASI data from Yahoo Finance...")
        subprocess.run([sys.executable, "scripts/fetch_data.py"], check=True)
        st.write("Processing technical indicators...")
        subprocess.run([sys.executable, "scripts/preprocess.py"], check=True)
        status.update(label="Market data updated successfully!", state="complete", expanded=False)
    st.cache_data.clear()

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    return None, None

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def get_latest_predictions(df, model, features):
    latest_data = []
    for (symbol, name), group in df.groupby(['Symbol', 'Company Name']):
        group = group.sort_values('Date')
        latest_row = group.iloc[-1:].copy()
        
        # We want to predict for the *next* day
        X = latest_row[features]
        preds = model.predict(X)
        prediction = preds[0] if len(preds) > 0 else 0
        
        current_price = latest_row['Close'].values[0]
        atr = latest_row['ATR'].values[0]
        support = latest_row['Support'].values[0]
        resistance = latest_row['Resistance'].values[0]
        bb_upper = latest_row['BB_Upper'].values[0]
        bb_lower = latest_row['BB_Lower'].values[0]
        adx = latest_row['ADX'].values[0]
        
        # --- WALL STREET LOGIC ---
        # Entry: Professional "Buy Zone" (Near 20 EMA or BB Lower)
        entry_point = round(current_price, 2)
        
        # Take Profit: Next Major Resistance or Upper BB
        take_profit = round(max(current_price * (1 + prediction * 1.5), bb_upper), 2)
        
        # Stop Loss: Below Support or 1.5 * ATR (Institutional Standard)
        stop_loss = round(min(support, current_price - (1.5 * atr)), 2)
        
        # Risk/Reward Ratio
        risk = entry_point - stop_loss
        reward = take_profit - entry_point
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        latest_data.append({
            'Company Name': name,
            'Symbol': symbol,
            'Date': latest_row['Date'].values[0],
            'Current Price': current_price,
            'Trend Strength (ADX)': round(adx, 1),
            'Entry Point': entry_point,
            'Take Profit': take_profit,
            'Stop Loss': stop_loss,
            'Risk/Reward': rr_ratio,
            'Predicted Return (%)': prediction * 100
        })
    
    return pd.DataFrame(latest_data).sort_values('Risk/Reward', ascending=False)

# UI Elements
st.title("🇸🇦 TASI AI Stock Prediction App")
st.markdown("This app uses a Random Forest model trained on TASI historical data to predict the top 3 stocks for the next trading day.")

# Sidebar
st.sidebar.header("Data Management")
st.sidebar.write("📡 **Source:** Yahoo Finance")
st.sidebar.warning("⚠️ Note: Yahoo Finance data for TASI may lag behind Investing.com by 2-3 days.")
if st.sidebar.button("🔄 Refresh Market Data", help="Fetch the latest data and update indicators"):
    run_update_scripts()

st.sidebar.divider()
st.sidebar.header("Stock Analysis Settings")

model, features = load_model()
df = load_data()

if df is not None and model is not None:
    # Use Name (Symbol) in selectbox
    df['Display Name'] = df['Company Name'] + " (" + df['Symbol'] + ")"
    selected_display_name = st.sidebar.selectbox("Select a stock for detailed view:", df['Display Name'].unique())
    
    # Safety check for empty filter
    selected_rows = df[df['Display Name'] == selected_display_name]
    if not selected_rows.empty:
        selected_stock = selected_rows['Symbol'].iloc[0]
        
        # Get latest data date
        last_date = df['Date'].max().strftime('%Y-%m-%d')
        st.sidebar.caption(f"Data updated as of: {last_date}")
        
        # Get latest predictions
        predictions_df = get_latest_predictions(df, model, features)
        
        if not predictions_df.empty:
            # Top 3 Picks by Risk/Reward
            st.header(f"🎯 Top 3 Institutional Picks for Tomorrow ({datetime.now().strftime('%Y-%m-%d')})")
            top_3 = predictions_df.head(3)
            
            cols = st.columns(3)
            for i, (_, row) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.metric(label=f"#{i+1} {row['Company Name']}", 
                              value=f"R:R {row['Risk/Reward']}", 
                              delta=f"{row['Predicted Return (%)']:.2f}% Predicted")
                    st.caption(f"🗓️ Price Date: {pd.to_datetime(row['Date']).strftime('%Y-%m-%d')}")
                    
                    # Trend Strength Color
                    adx = row['Trend Strength (ADX)']
                    trend_label = "Strong Trend" if adx > 25 else "Weak/Sideways"
                    st.write(f"**Trend:** {trend_label} (ADX: {adx})")
                    
                    st.write(f"**Entry:** {row['Entry Point']} SAR")
                    st.write(f"**Take Profit:** {row['Take Profit']} SAR")
                    st.write(f"**Stop Loss:** {row['Stop Loss']} SAR")
                    
                    # R:R Indicator
                    st.progress(min(1.0, row['Risk/Reward']/5), text="Profit Potential")

            # All Predictions Table
            st.header("📊 Quant Trading Signals")
            st.dataframe(predictions_df, use_container_width=True)
            
            # Stock History Chart
            st.header(f"📈 {selected_display_name} History")
            stock_df = df[df['Symbol'] == selected_stock].tail(100)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=stock_df['Date'],
                            open=stock_df['Open'],
                            high=stock_df['High'],
                            low=stock_df['Low'],
                            close=stock_df['Close'],
                            name='Price'))
            
            fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['SMA_20'], 
                                     line=dict(color='orange', width=1), name='SMA 20'))
            fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['SMA_50'], 
                                     line=dict(color='blue', width=1), name='SMA 50'))
            
            fig.update_layout(xaxis_rangeslider_visible=False, title=f"{selected_stock} Price Chart")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No predictions could be generated. Try refreshing market data.")
    else:
        st.warning("Please select a valid stock.")
    
else:
    st.error("Data or Model files not found. Please click 'Refresh Market Data' to initialize.")
    if st.button("Initialize Data"):
        run_update_scripts()
        st.rerun()

st.sidebar.info(f"Last session check: {datetime.now().strftime('%H:%M')}")
