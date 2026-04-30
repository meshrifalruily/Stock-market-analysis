import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def calculate_advanced_metrics(df):
    """محاكاة Pine Script وحساب المؤشرات الفنية والمقاييس الأسبوعية."""
    
    # توحيد أسماء الأعمدة (أحرف صغيرة)
    df_ta = df.copy()
    df_ta.columns = [c.lower() for c in df_ta.columns]
    
    # 1. مؤشرات الاتجاه (Trend)
    df['sma_20'] = ta.sma(df_ta['close'], length=20)
    df['sma_50'] = ta.sma(df_ta['close'], length=50)
    df['sma_200'] = ta.sma(df_ta['close'], length=200)
    df['ema_20'] = ta.ema(df_ta['close'], length=20)
    
    # 2. مؤشرات الزخم (Momentum)
    df['rsi'] = ta.rsi(df_ta['close'], length=14)
    macd = ta.macd(df_ta['close'])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        
    # 3. قوة الاتجاه (ADX)
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
    
    # 5. محاكاة إشارة TradingView
    df['tv_signal'] = 0
    buy_cond = (df_ta['close'] > df['ema_20']) & (df['rsi'] > 50) & (df['tv_adx'] > 20)
    strong_buy = buy_cond & (df_ta['close'] > df['sma_50']) & (df['rsi'] > 60)
    df.loc[buy_cond, 'tv_signal'] = 1
    df.loc[strong_buy, 'tv_signal'] = 2
    
    # 6. مقاييس المخاطر والأداء
    df['daily_return'] = df_ta['close'].pct_change()
    df['weekly_return_hist'] = df_ta['close'].pct_change(5) # العائد في الـ 5 أيام الماضية
    
    df['sharpe_ratio_rolling'] = (df['daily_return'].rolling(window=20).mean() / 
                                  df['daily_return'].rolling(window=20).std()) * np.sqrt(252)
    
    tasi_col = 'Macro_TASI_PROXY'
    if tasi_col in df.columns:
        df['market_return'] = df[tasi_col].pct_change()
        covariance = df['daily_return'].rolling(60).cov(df['market_return'])
        variance = df['market_return'].rolling(60).var()
        df['beta'] = covariance / variance
    
    oil_col = 'Macro_BZ_F'
    if oil_col in df.columns:
        df['oil_return'] = df[oil_col].pct_change()
        df['oil_correlation'] = df['daily_return'].rolling(60).corr(df['oil_return'])

    # 7. الأهداف المستقبلية (Targets)
    df['target_next_day_return'] = df['daily_return'].shift(-1)
    df['target_next_week_return'] = df_ta['close'].pct_change(5).shift(-5) # العائد المتوقع بعد أسبوع
    # تنظيف وتوحيد الأسماء
    df.columns = [c.lower() for c in df.columns]
    rename_dict = {
        'macd_12_26_9': 'macd',
        'macds_12_26_9': 'macd_signal',
        'macdh_12_26_9': 'macd_hist'
    }
    df = df.rename(columns=rename_dict)

    # ملء الفراغات الناتجة عن النوافذ الزمنية (Rolling Windows)
    df = df.ffill().bfill().fillna(0)

    # الاحتفاظ بالأعمدة النصية الهامة
    return df


def preprocess_all_data(combined_file_path, output_file_path):
    if not os.path.exists(combined_file_path):
        print(f"خطأ: {combined_file_path} غير موجود.")
        return
    
    df = pd.read_csv(combined_file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values(['Symbol', 'Date'])
    
    processed_dfs = []
    for symbol, group in df.groupby('Symbol'):
        print(f"جاري معالجة البيانات المتقدمة لـ {symbol}...")
        processed_group = calculate_advanced_metrics(group.copy())
        
        # الحفاظ على الأعمدة الوصفية
        processed_group['sector'] = group['Sector'].iloc[0] if 'Sector' in group.columns else "عام"
        processed_group['sentiment'] = group['Sentiment'].iloc[0] if 'Sentiment' in group.columns else 0.5
        
        processed_dfs.append(processed_group)
    
    final_df = pd.concat(processed_dfs)
    final_df.to_csv(output_file_path, index=False)
    print(f"تم حفظ البيانات المعالجة الشاملة في {output_file_path}")

if __name__ == "__main__":
    preprocess_all_data("data/tasi_combined.csv", "data/tasi_processed.csv")
