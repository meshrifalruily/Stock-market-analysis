"""جلب أخبار عربية حقيقية لأسهم تاسي ووصلها بمحرّك المشاعر.

يستخدم Google News RSS (مجاني، بلا مفاتيح، بلا تبعيات خارجية) للبحث عن عناوين
الأخبار العربية لكل شركة، ثم يمرّرها إلى scripts/sentiment.py لحساب درجة مشاعر
حقيقية تحلّ محل القيمة التقريبية الثابتة.

الاستخدام:
    from scripts.news_fetcher import google_news_fetcher
    from scripts.sentiment import compute_symbol_sentiment
    score = compute_symbol_sentiment("2222.SR", "أرامكو السعودية",
                                     fetcher=google_news_fetcher)

حدود الأمانة المنهجية:
  - Google News RSS يعطي أخباراً *حديثة* فقط، فلا يمكن التحقق من IC تاريخياً
    دون أرشيف أخبار مؤرّخ (غير متاح مجاناً). لذا يُستخدم للإشارة الحيّة فقط.
  - الجلب يخضع لحدود المعدّل (rate limits)؛ نضيف cache وتهلة بين الطلبات.
"""

from __future__ import annotations

import os
import sys
import time
import ssl
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=ar&gl=SA&ceid=SA:ar"
MAX_HEADLINES = int(os.getenv("TASI_NEWS_MAX_HEADLINES", "8"))
REQUEST_TIMEOUT = int(os.getenv("TASI_NEWS_TIMEOUT", "15"))

# cache بسيط في الذاكرة لتجنّب تكرار الطلبات لنفس الرمز في الجلسة.
_news_cache: dict = {}


def _http_get(url: str) -> str:
    """جلب نص URL مع التعامل مع شهادات SSL على ماك."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TASI-AI/1.0)"}
    try:
        with urlopen(Request(url, headers=headers), timeout=REQUEST_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        # رجوع: سياق SSL غير موثّق (مشكلة شائعة على macOS).
        ctx = ssl._create_unverified_context()
        with urlopen(Request(url, headers=headers), timeout=REQUEST_TIMEOUT, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="ignore")


def fetch_news_headlines(query: str, limit: int = MAX_HEADLINES) -> List[str]:
    """يُرجع قائمة عناوين أخبار عربية لاستعلام معيّن عبر Google News RSS."""
    if query in _news_cache:
        return _news_cache[query]
    url = GOOGLE_NEWS_RSS.format(query=quote(query))
    headlines: List[str] = []
    try:
        xml = _http_get(url)
        root = ElementTree.fromstring(xml)
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                headlines.append(title_el.text.strip())
            if len(headlines) >= limit:
                break
    except Exception as exc:
        print(f"تعذّر جلب أخبار '{query}': {exc}")
    _news_cache[query] = headlines
    return headlines


def google_news_fetcher(symbol: str, company_name: str = "") -> List[str]:
    """fetcher متوافق مع sentiment.compute_symbol_sentiment.

    يبحث باسم الشركة العربي (أدقّ من الرمز للأخبار العربية)، ويُضيف "سهم"
    لتضييق النتائج على السياق المالي.
    """
    query = f"{company_name} سهم" if company_name else symbol.replace(".SR", "")
    time.sleep(0.3)  # تهلة بسيطة لاحترام حدود المعدّل
    return fetch_news_headlines(query)


def demo(symbols=None):
    """عرض توضيحي: يجلب أخبار بضع شركات ويحسب مشاعرها الفعلية."""
    from scripts.sentiment import compute_symbol_sentiment, score_headlines

    samples = symbols or [
        ("2222.SR", "أرامكو السعودية"),
        ("1120.SR", "مصرف الراجحي"),
        ("2010.SR", "سابك"),
    ]
    print("جلب أخبار حقيقية وحساب المشاعر (Google News RSS):\n")
    for symbol, name in samples:
        headlines = google_news_fetcher(symbol, name)
        score = score_headlines(headlines)
        print(f"▶ {name} ({symbol}) — مشاعر={score:.3f} | عناوين={len(headlines)}")
        for h in headlines[:3]:
            print(f"    • {h[:90]}")
        print()


if __name__ == "__main__":
    demo()
