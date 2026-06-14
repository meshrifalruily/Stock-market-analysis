"""تحليل المشاعر المالية العربية لأسهم تاسي.

المشكلة التي يعالجها هذا الملف: حقل ``sentiment`` في خط الأنابيب الحالي كان
قيمة ثابتة/تقريبية (≈0.5) لا تحمل إشارة حقيقية. هنا نوفّر:

  1. مُصنّف مشاعر معجمي (lexicon-based) يعمل دون اتصال أو مكتبات خارجية،
     مبني على معجم مالي عربي للكلمات الإيجابية/السلبية. يعطي إشارة حقيقية فوراً.
  2. خطّاف (hook) اختياري لنموذج محوّلات عربي (مثل CAMeLBERT) إن كان مثبتاً،
     لتحليل أدق عند توفره.
  3. واجهة ``fetch_headlines`` مجرّدة تُوصَل بمصدر أخبار حقيقي (أرقام/تداول/RSS)
     عبر دالة تُمرَّر من الخارج — دون تضمين أي مفاتيح أو روابط مزروعة.

الناتج: درجة مشاعر في النطاق [0, 1] حيث 0.5 = محايد، >0.5 إيجابي، <0.5 سلبي.
هذا يوافق التوقعات الحالية في preprocess.py و main.py.

ملاحظة أمانة منهجية: المُصنّف المعجمي تقريبي وليس بديلاً عن نموذج مُدرَّب على
بيانات مالية عربية موسومة. لكنه إشارة حقيقية متغيّرة بدل ثابت بلا معنى.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Optional

# ── معجم مالي عربي مبسّط (يمكن توسيعه) ──────────────────────────────────────
# كلمات تدل على زخم/أداء إيجابي.
POSITIVE_TERMS = {
    "ارتفاع", "ارتفع", "صعود", "صعد", "نمو", "ربح", "أرباح", "مكاسب", "تفوق",
    "قياسي", "توزيعات", "استحواذ", "ترقية", "إيجابي", "قوي", "تعافي", "انتعاش",
    "زيادة", "تحسن", "اختراق", "دعم", "صفقة", "عقد", "توسع", "تمديد", "فائض",
    "توصية شراء", "هدف أعلى", "أفضل من المتوقع", "نتائج قوية",
}
# كلمات تدل على ضعف/أداء سلبي.
NEGATIVE_TERMS = {
    "انخفاض", "انخفض", "هبوط", "هبط", "تراجع", "خسارة", "خسائر", "تراجعت",
    "ضعف", "سلبي", "تخفيض", "إنذار", "تحذير", "عجز", "ديون", "تعثر", "غرامة",
    "تحقيق", "إيقاف", "تعليق", "شطب", "خفض التصنيف", "أسوأ من المتوقع",
    "نتائج ضعيفة", "توصية بيع", "هدف أدنى", "مخاطر", "تباطؤ", "ضغوط",
}
# كلمات تضاعف الإشارة (مكثّفات).
INTENSIFIERS = {"حاد", "كبير", "قوي", "ضخم", "غير مسبوق", "مفاجئ"}

_AR_DIACRITICS = re.compile(r"[ً-ْ]")
_NON_AR = re.compile(r"[^؀-ۿ\s]")


def normalize_arabic(text: str) -> str:
    """تطبيع النص العربي: إزالة التشكيل وتوحيد الألف والياء والتاء المربوطة."""
    if not text:
        return ""
    text = _AR_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = _NON_AR.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalized_terms(terms: Iterable[str]) -> set:
    return {normalize_arabic(t) for t in terms}


_POS = _normalized_terms(POSITIVE_TERMS)
_NEG = _normalized_terms(NEGATIVE_TERMS)
_INT = _normalized_terms(INTENSIFIERS)


def score_text_lexicon(text: str) -> float:
    """درجة مشاعر معجمية لنص واحد في النطاق [0, 1] (0.5 = محايد)."""
    norm = normalize_arabic(text)
    if not norm:
        return 0.5

    pos = sum(1 for term in _POS if term and term in norm)
    neg = sum(1 for term in _NEG if term and term in norm)
    intensity = 1.0 + 0.5 * sum(1 for term in _INT if term and term in norm)

    if pos == 0 and neg == 0:
        return 0.5

    raw = (pos - neg) * intensity
    # ضغط لوجستي إلى [0, 1].
    return _sigmoid(raw)


def score_headlines(headlines: Iterable[str]) -> float:
    """متوسط درجة المشاعر لمجموعة عناوين. يُرجع 0.5 إذا لا عناوين."""
    scores = [score_text_lexicon(h) for h in headlines if h]
    if not scores:
        return 0.5
    return float(sum(scores) / len(scores))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


# ── خطّاف نموذج المحوّلات العربي (اختياري) ───────────────────────────────────
_transformer_pipeline = None


def _try_load_transformer():
    """تحميل نموذج CAMeLBERT للمشاعر إن كانت transformers مثبتة. وإلا None."""
    global _transformer_pipeline
    if _transformer_pipeline is not None:
        return _transformer_pipeline
    try:
        from transformers import pipeline
        _transformer_pipeline = pipeline(
            "sentiment-analysis",
            model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
        )
    except Exception:
        _transformer_pipeline = False  # علامة "حاولنا وفشلنا" لتجنّب إعادة المحاولة
    return _transformer_pipeline


def score_text(text: str, prefer_transformer: bool = False) -> float:
    """درجة مشاعر لنص: يستخدم المحوّلات إن طُلبت وتوفّرت، وإلا المعجم."""
    if prefer_transformer:
        pipe = _try_load_transformer()
        if pipe:
            try:
                out = pipe(text[:512])[0]
                label = out.get("label", "").lower()
                conf = float(out.get("score", 0.5))
                if "pos" in label:
                    return 0.5 + 0.5 * conf
                if "neg" in label:
                    return 0.5 - 0.5 * conf
                return 0.5
            except Exception:
                pass
    return score_text_lexicon(text)


# ── واجهة جلب الأخبار (تُوصَل بمصدر حقيقي من الخارج) ──────────────────────────
# نوع الدالة: (رمز السهم, اسم الشركة) -> قائمة عناوين.
HeadlineFetcher = Callable[[str, str], Iterable[str]]


def compute_symbol_sentiment(
    symbol: str,
    company_name: str = "",
    fetcher: Optional[HeadlineFetcher] = None,
    prefer_transformer: bool = False,
) -> float:
    """يحسب درجة مشاعر لسهم واحد من عناوينه الأخيرة.

    ``fetcher`` دالة يوفّرها المستخدم تُرجع عناوين الأخبار لرمز معيّن من مصدر
    حقيقي (مثل RSS لموقع أرقام). إذا لم تُمرَّر، يُرجع 0.5 (محايد) بأمان.
    لا نزرع أي مصدر أو مفتاح هنا — يبقى الوصل بيد المستخدم.
    """
    if fetcher is None:
        return 0.5
    try:
        headlines = list(fetcher(symbol, company_name))
    except Exception:
        return 0.5
    if prefer_transformer:
        scores = [score_text(h, prefer_transformer=True) for h in headlines if h]
        return float(sum(scores) / len(scores)) if scores else 0.5
    return score_headlines(headlines)


if __name__ == "__main__":
    # عرض توضيحي سريع.
    samples = [
        "أرامكو تسجّل أرباحاً قياسية وتوزيعات قوية أفضل من المتوقع",
        "تراجع حاد في سهم الشركة بعد تحذير من خسائر وديون متراكمة",
        "السهم يتداول دون تغيّر يُذكر اليوم",
    ]
    for s in samples:
        print(f"{score_text_lexicon(s):.3f}  |  {s}")
