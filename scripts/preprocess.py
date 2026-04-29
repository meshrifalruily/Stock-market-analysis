import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def calculate_advanced_metrics(df):
    """محاكاة Pine Script وحساب المؤشرات الفنية المتقدمة."""
    
    # توحيد أسماء الأعمدة (أحرف صغيرة)
    df_ta = df.copy()
    df_ta.columns = [c.lower() for c in df_ta.columns]
    
    # 1. مؤشرات الاتجاه (Trend)
    df['sma_20'] = ta.sma(df_ta['close'], length=20)
    df['sma_50'] = ta.sma(df_ta['close'], length=50)
    df['sma_200'] = ta.sma(df_ta['close'], length=200)
    df['ema_20'] = ta.ema(df_ta['close'], length=20)
    
    # 2. مؤشرات الزخم (Momentum - TradingView Style)
    df['rsi'] = ta.rsi(df_ta['close'], length=14)
    macd = ta.macd(df_ta['close'])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        
    # 3. قوة الاتجاه (ADX - المحرك الأساسي لتريدنج فيو)
    adx_df = ta.adx(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
    if adx_df is not None:
        df['tv_adx'] = adx_df['ADX_14']
    else:
        df['tv_adx'] = 0
    
    # 4. مؤشرات التقلب (Volatility)
    df['atr'] = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
    bbands = ta.bbands(df_ta['close'], length=20)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
    
    # 5. محاكاة إشارة TradingView (Pine Script Signal Logic)
    # الإشارة = 2 (شراء قوي), 1 (شراء), 0 (حياد), -1 (بيع), -2 (بيع قوي)
    df['tv_signal'] = 0
    
    # منطق الشراء: السعر فوق المتوسط + RSI صاعد + ADX قوي
    buy_cond = (df_ta['close'] > df['ema_20']) & (df['rsi'] > 50) & (df['tv_adx'] > 20)
    strong_buy = buy_cond & (df_ta['close'] > df['sma_50']) & (df['rsi'] > 60)
    
    df.loc[buy_cond, 'tv_signal'] = 1
    df.loc[strong_buy, 'tv_signal'] = 2
    
    # 6. مقاييس المخاطر
    df['daily_return'] = df_ta['close'].pct_change()
    df['sharpe_ratio_rolling'] = (df['daily_return'].rolling(window=20).mean() / 
                                  df['daily_return'].rolling(window=20).std()) * np.sqrt(252)
    
    # معامل بيتا (Beta)
    tasi_col = 'Macro_TASI_PROXY'
    if tasi_col in df.columns:
        df['market_return'] = df[tasi_col].pct_change()
        covariance = df['daily_return'].rolling(60).cov(df['market_return'])
        variance = df['market_return'].rolling(60).var()
        df['beta'] = covariance / variance
    
    # الارتباط مع النفط
    oil_col = 'Macro_BZ_F'
    if oil_col in df.columns:
        df['oil_return'] = df[oil_col].pct_change()
        df['oil_correlation'] = df['daily_return'].rolling(60).corr(df['oil_return'])

    # 7. الهدف: عائد اليوم التالي
    df['target_next_day_return'] = df['daily_return'].shift(-1)
    
    # تنظيف وتوحيد الأسماء
    df.columns = [c.lower() for c in df.columns]
    rename_dict = {
        'macd_12_26_9': 'macd',
        'macds_12_26_9': 'macd_signal',
        'macdh_12_26_9': 'macd_hist'
    }
    df = df.rename(columns=rename_dict)
    
    return df

def preprocess_all_data(combined_file_path, output_file_path):
    if not os.path.exists(combined_file_path):
        print(f"خطأ: {combined_file_path} غير موجود.")
        return
    
    df = pd.read_csv(combined_file_path)
    # إزالة المناطق الزمنية فوراً عند التحميل
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values(['Symbol', 'Date'])
    
    processed_dfs = []
    for symbol, group in df.groupby('Symbol'):
        print(f"جاري حساب محاكاة Pine Script لـ {symbol}...")
        processed_group = calculate_advanced_metrics(group.copy())
        processed_dfs.append(processed_group)
    
    final_df = pd.concat(processed_dfs)
    final_df.to_csv(output_file_path, index=False)
    print(f"تم حفظ البيانات المعالجة بالمحاكاة المتقدمة في {output_file_path}")

if __name__ == "__main__":
    preprocess_all_data("data/tasi_combined.csv", "data/tasi_processed.csv")
