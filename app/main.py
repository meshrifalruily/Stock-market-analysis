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

# RTL and Arabic Font Support (CSS Injection)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Tajawal', sans-serif;
        direction: RTL;
        text-align: right;
    }
    .stMetric {
        text-align: right !important;
    }
    div[data-testid="stSidebar"] {
        direction: RTL;
        text-align: right;
    }
    /* Fix for Plotly to also respect RTL if possible, though mostly it's internal */
    </style>
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="منصة تحليل الأسهم السعودية بالذكاء الاصطناعي", layout="wide")

# Paths
MODEL_PATH = "models/tasi_rf_model.joblib"
DATA_PATH = "data/tasi_processed.csv"
FEATURES_PATH = "models/feature_names.joblib"

# Map symbols to Arabic names
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

def run_update_scripts():
    with st.status("جاري تحديث البيانات والاستراتيجية...", expanded=True) as status:
        st.write("جاري جلب أحدث بيانات تاسي من ياهو فاينانس...")
        subprocess.run([sys.executable, "scripts/fetch_data.py"], check=True)
        st.write("جاري معالجة المؤشرات الفنية...")
        subprocess.run([sys.executable, "scripts/preprocess.py"], check=True)
        st.write("جاري إعادة تدريب نموذج الذكاء الاصطناعي...")
        subprocess.run([sys.executable, "scripts/train_model.py"], check=True)
        st.write("جاري تشغيل محاكاة الاختبار العكسي...")
        subprocess.run([sys.executable, "scripts/backtest.py"], check=True)
        status.update(label="تم تحديث النظام بنجاح!", state="complete", expanded=False)
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
        df['date'] = pd.to_datetime(df['date'], utc=True)
        # Apply Arabic names
        df['اسم الشركة'] = df['symbol'].map(ARABIC_NAMES).fillna(df['company name'])
        return df
    return None

def get_latest_predictions(df, model, features):
    latest_data = []
    for (symbol, name), group in df.groupby(['symbol', 'اسم الشركة']):
        group = group.sort_values('date')
        latest_row = group.iloc[-1:].copy()
        
        X = latest_row[features].fillna(0)
        preds = model.predict(X)
        prediction = preds[0] if len(preds) > 0 else 0
        
        current_price = latest_row['close'].values[0]
        atr = latest_row['atr'].values[0]
        
        bb_upper = latest_row['bbu_20_2.0'].values[0] if 'bbu_20_2.0' in latest_row.columns else current_price * 1.05
        bb_lower = latest_row['bbl_20_2.0'].values[0] if 'bbl_20_2.0' in latest_row.columns else current_price * 0.95
        
        entry_point = round(current_price, 2)
        tp_target = current_price * (1 + max(0.01, prediction * 1.5))
        take_profit = round(max(tp_target, bb_upper), 2)
        stop_loss = round(min(bb_lower, current_price - (1.5 * atr)), 2)
        
        risk = entry_point - stop_loss
        reward = take_profit - entry_point
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        latest_data.append({
            'اسم الشركة': name,
            'الرمز': symbol,
            'التاريخ': latest_row['date'].values[0],
            'السعر الحالي': current_price,
            'نقطة الدخول': entry_point,
            'هدف الربح': take_profit,
            'وقف الخسارة': stop_loss,
            'نسبة الربح/المخاطرة': rr_ratio,
            'العائد المتوقع (%)': prediction * 100,
            'نسبة شارب (متحرك)': round(latest_row['sharpe_ratio_rolling'].values[0], 2) if 'sharpe_ratio_rolling' in latest_row.columns else 0
        })
    
    return pd.DataFrame(latest_data).sort_values('العائد المتوقع (%)', ascending=False)

# UI Elements
st.title("🇸🇦 منصة تحليل الأسهم السعودية (TASI) بالذكاء الاصطناعي")
st.markdown("توقعات احترافية للسوق السعودي باستخدام الغابات العشوائية (Random Forest) والمؤشرات الفنية المتقدمة.")

# Sidebar
st.sidebar.header("إدارة البيانات")
st.sidebar.write("📡 **المصدر:** ياهو فاينانس (Yahoo Finance)")
if st.sidebar.button("🔄 تحديث البيانات وإعادة التدريب", help="جلب أحدث البيانات، تحديث المؤشرات، وإعادة تدريب النموذج"):
    run_update_scripts()

st.sidebar.divider()
st.sidebar.header("إعدادات التحليل")

model, features = load_model()
df = load_data()

if df is not None and model is not None:
    df['display_name'] = df['اسم الشركة'] + " (" + df['symbol'] + ")"
    selected_display_name = st.sidebar.selectbox("اختر شركة للعرض التفصيلي:", df['display_name'].unique())
    
    selected_rows = df[df['display_name'] == selected_display_name]
    if not selected_rows.empty:
        selected_symbol = selected_rows['symbol'].iloc[0]
        last_date_ts = df['date'].max()
        st.sidebar.caption(f"بيانات السوق محدثة حتى: {last_date_ts.strftime('%Y-%m-%d')}")
        
        predictions_df = get_latest_predictions(df, model, features)
        
        if not predictions_df.empty:
            # --- Strategy Performance Section ---
            if os.path.exists("data/backtest_results.csv"):
                st.divider()
                st.header("🏆 أداء الاستراتيجية (اختبار عكسي لآخر 6 أشهر)")
                backtest_df = pd.read_csv("data/backtest_results.csv")
                backtest_df['date'] = pd.to_datetime(backtest_df['date'])
                
                final_val = backtest_df['value'].iloc[-1]
                total_return = (final_val - 100.0)
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.metric("العائد التاريخي للاستراتيجية", f"{total_return:.2f}%")
                perf_col2.metric("رأس المال البدائي", "100.00 ر.س")
                perf_col3.metric("رأس المال النهائي", f"{final_val:.2f} ر.س")
                
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(x=backtest_df['date'], y=backtest_df['value'], 
                                              fill='tozeroy', name='منحنى نمو الاستراتيجية'))
                fig_perf.update_layout(title="نمو محفظة الاستراتيجية (آخر 6 أشهر)", height=300)
                st.plotly_chart(fig_perf, use_container_width=True)
                st.divider()

            # Top 3 Picks
            st.header(f"🎯 أفضل 3 شركات متوقع صعودها للجلسة القادمة")
            top_3 = predictions_df.head(3)
            
            cols = st.columns(3)
            for i, (_, row) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.metric(label=f"#{i+1} {row['اسم الشركة']}", 
                              value=f"{row['العائد المتوقع (%)']:.2f}%", 
                              delta=f"ربح/مخاطرة {row['نسبة الربح/المخاطرة']}")
                    st.caption(f"نسبة شارب: {row['نسبة شارب (متحرك)']}")
                    
                    st.write(f"**الدخول:** {row['نقطة الدخول']} ر.س")
                    st.write(f"**الهدف:** {row['هدف الربح']} ر.س")
                    st.write(f"**الوقف:** {row['وقف الخسارة']} ر.س")
                    
                    confidence = min(1.0, max(0.0, (row['نسبة شارب (متحرك)'] + 1) / 4))
                    st.progress(confidence, text="ثقة النموذج (معدلة حسب المخاطر)")

            # All Predictions Table
            st.header("📊 توقعات السوق ومقاييس المخاطر الكاملة")
            st.dataframe(predictions_df, use_container_width=True)
            
            # Stock History Chart
            st.header(f"📈 الرسم البياني الفني لشركة {selected_display_name}")
            stock_df = df[df['symbol'] == selected_symbol].tail(100)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=stock_df['date'],
                            open=stock_df['open'],
                            high=stock_df['high'],
                            low=stock_df['low'],
                            close=stock_df['close'],
                            name='السعر'))
            
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['sma_20'], 
                                     line=dict(color='orange', width=1), name='متوسط 20'))
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['sma_50'], 
                                     line=dict(color='blue', width=1), name='متوسط 50'))
            
            if 'bbu_20_2.0' in stock_df.columns:
                fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['bbu_20_2.0'], 
                                         line=dict(color='rgba(173, 216, 230, 0.4)', width=1), name='بولينجر علوي'))
                fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['bbl_20_2.0'], 
                                         line=dict(color='rgba(173, 216, 230, 0.4)', width=1), name='بولينجر سفلي',
                                         fill='tonexty'))
            
            fig.update_layout(xaxis_rangeslider_visible=False, title=f"تحليل سعر {selected_symbol}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("لم يتم توليد أي توقعات. حاول تحديث بيانات السوق.")
    else:
        st.warning("يرجى اختيار شركة صالحة.")
    
else:
    st.error("ملفات البيانات أو النموذج غير موجودة. يرجى الضغط على 'تحديث البيانات' للبدء.")
    if st.button("بدء تشغيل النظام"):
        run_update_scripts()
        st.rerun()

st.sidebar.info(f"آخر تحديث للجلسة: {datetime.now().strftime('%H:%M')}")
