"""تحديث دفعي لمشاعر الأخبار الحقيقية وتخزينها في كاش يقرأه التطبيق.

يجلب عناوين الأخبار العربية الحديثة لكل رمز سائل عبر Google News RSS، يحسب درجة
مشاعر فعلية، ويحفظها في data/sentiment_cache.json. يُشغَّل دورياً (cron / زر تحديث)
لأن الجلب الحيّ لكل الأسهم في كل طلب بطيء ويخضع لحدود المعدّل.

التطبيق يمكنه قراءة هذا الكاش لعرض مشاعر حقيقية بدل القيمة التقريبية الثابتة.
"""

import os
import sys
import json
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd

from scripts.news_fetcher import google_news_fetcher
from scripts.sentiment import score_headlines
from scripts.strategy_config import load_strategy_config
from scripts.strategy_rules import filter_main_market

DATA_PATH = os.path.join(ROOT_DIR, "data/tasi_processed.csv")
CACHE_PATH = os.path.join(ROOT_DIR, "data/sentiment_cache.json")


def update_sentiment_cache(limit=None):
    """يحسب مشاعر الأسهم السائلة ويحفظها في الكاش. limit يحدّ عدد الرموز."""
    if not os.path.exists(DATA_PATH):
        print(f"خطأ: {DATA_PATH} غير موجود.")
        return

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = filter_main_market(df)
    config = load_strategy_config()

    # آخر صف لكل رمز سائل (نركّز على القابلة للتداول).
    latest = df.sort_values("date").groupby("symbol").tail(1)
    if "avg_traded_value_20d" in latest.columns:
        latest = latest[latest["avg_traded_value_20d"].fillna(0) >= config["min_avg_traded_value"]]

    name_col = "company name arabic" if "company name arabic" in latest.columns else "company name"
    rows = list(latest.iterrows())
    if limit:
        rows = rows[:limit]

    cache = {"updated_at": datetime.now().isoformat(timespec="seconds"), "scores": {}}
    print(f"تحديث مشاعر {len(rows)} رمزاً سائلاً...")
    for i, (_, row) in enumerate(rows, 1):
        symbol = row["symbol"]
        name = row.get(name_col, "") or ""
        headlines = google_news_fetcher(symbol, name)
        score = score_headlines(headlines)
        cache["scores"][symbol] = {
            "sentiment": round(float(score), 3),
            "headline_count": len(headlines),
            "top_headline": headlines[0][:140] if headlines else "",
        }
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}...")

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"تم حفظ مشاعر {len(cache['scores'])} رمزاً في {CACHE_PATH}")
    return cache


if __name__ == "__main__":
    # حدّ افتراضي بسيط لتجنّب حدود المعدّل عند التجربة؛ مرّر رقماً لتغييره.
    arg_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    update_sentiment_cache(limit=arg_limit)
