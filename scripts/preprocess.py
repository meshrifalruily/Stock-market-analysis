import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def calculate_advanced_metrics(df):
    """حساب المؤشرات الفنية والمقاييس المالية الاحترافية."""
    
    # توحيد أسماء الأعمدة لتناسب مكتبة pandas-ta (أحرف صغيرة)
    df_ta = df.copy()
    df_ta.columns = [c.lower() for c in df_ta.columns]
    
    # 1. مؤشرات الاتجاه
    df['sma_20'] = ta.sma(df_ta['close'], length=20)
    df['sma_50'] = ta.sma(df_ta['close'], length=50)
    df['sma_200'] = ta.sma(df_ta['close'], length=200)
    df['ema_20'] = ta.ema(df_ta['close'], length=20)
    
    # 2. مؤشرات الزخم
    df['rsi'] = ta.rsi(df_ta['close'], length=14)
    macd = ta.macd(df_ta['close'])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    
    # 3. مؤشرات التقلب
    df['atr'] = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
    bbands = ta.bbands(df_ta['close'], length=20)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
    
    # 4. مقاييس الأداء والمخاطر
    df['daily_return'] = df_ta['close'].pct_change()
    
    # نسبة شارب المتحركة (تقريبية سنوية)
    df['sharpe_ratio_rolling'] = (df['daily_return'].rolling(window=20).mean() / 
                                  df['daily_return'].rolling(window=20).std()) * np.sqrt(252)
    
    # معامل بيتا بالنسبة لبديل تاسي (الراجحي)
    tasi_col = 'Macro_TASI_PROXY'
    if tasi_col in df.columns:
        df['market_return'] = df[tasi_col].pct_change()
        covariance = df['daily_return'].rolling(60).cov(df['market_return'])
        variance = df['market_return'].rolling(60).var()
        df['beta'] = covariance / variance
    
    # الارتباط مع خام برنت
    oil_col = 'Macro_BZ_F'
    if oil_col in df.columns:
        df['oil_return'] = df[oil_col].pct_change()
        df['oil_correlation'] = df['daily_return'].rolling(60).corr(df['oil_return'])

    # 5. الهدف: عائد اليوم التالي
    df['target_next_day_return'] = df['daily_return'].shift(-1)
    
    # توحيد أسماء الأعمدة
    df.columns = [c.lower() for c in df.columns]
    rename_dict = {
        'macd_12_26_9': 'macd',
        'macds_12_26_9': 'macd_signal',
        'macdh_12_26_9': 'macd_hist'
    }
    df = df.rename(columns=rename_dict)
    
    # لم نعد نحذف الصفوف هنا للإبقاء على أحدث البيانات للتوقعات الحية
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
        print(f"جاري معالجة المقاييس المتقدمة لـ {symbol}...")
        processed_group = calculate_advanced_metrics(group.copy())
        processed_dfs.append(processed_group)
    
    final_df = pd.concat(processed_dfs)
    final_df.to_csv(output_file_path, index=False)
    print(f"تم حفظ البيانات المعالجة المتقدمة في {output_file_path}")

if __name__ == "__main__":
    preprocess_all_data("data/tasi_combined.csv", "data/tasi_processed.csv")
