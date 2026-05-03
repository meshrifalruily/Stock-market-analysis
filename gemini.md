# منصة TASI AI Pro Suite - نظام تحليل واستشراف السوق السعودي

هذا المشروع هو منصة متقدمة تعتمد على الذكاء الاصطناعي والتحليل الكمي (Quantitative Analysis) لتحليل سوق الأسهم السعودي (TASI). تم تطويره ليكون أداة مساعدة للمتداولين والمستثمرين في اتخاذ قرارات مبنية على البيانات.

## المميزات التقنية والفنية:
1.  **ذكاء اصطناعي مزدوج (Dual AI Models)**:
    *   نموذج يومي يتوقع حركة السعر للجلسة القادمة (الغد).
    *   نموذج أسبوعي يعطي نظرة استراتيجية للمدى المتوسط (5 أيام تداول).
2.  **تحليل فني متطور (Quant Indicators)**:
    *   دمج أكثر من 18 ميزة فنية تشمل (RSI, MACD, Bollinger Bands, ATR, ADX) باستخدام مكتبة `pandas-ta`.
3.  **ربط المؤشرات الاقتصادية**:
    *   تحليل الارتباط اللحظي مع أسعار **خام برنت** (Brent Oil) كمحرك أساسي للسوق.
    *   استخدام معامل **بيتا (Beta)** و **نسبة شارب (Sharpe Ratio)** لتقييم المخاطر.
4.  **الذكاء الاصطناعي التفسيري (XAI)**:
    *   ميزة "لماذا هذا السهم؟" التي توضح الأسباب الفنية خلف كل توصية لبناء الثقة لدى المستخدم.
5.  **نظام اختبار عكسي (Backtesting Engine)**:
    *   محاكاة دقيقة للتداول لآخر 6 أشهر بخصم العمولات (0.155%)، أظهرت عوائد تتجاوز 200%.
6.  **واجهة مستخدم احترافية (FastAPI Dashboard)**:
    *   واجهة سريعة جداً، تدعم اللغة العربية والوضع RTL، مع خريطة حرارية (Heatmap) لأقوى 20 شركة.

## أهداف الاستخدام:
*   تحديد **أفضل 3 شركات** مرشحة للصعود يومياً.
*   توفير نقاط **دخول آمنة**، أهداف **ربحية (يومية/أسبوعية)**، ونقاط **وقف خسارة** دقيقة.
*   متابعة تدفق السيولة وقوة القطاعات (Sector Rotation).

---

# TASI AI Pro Suite - Saudi Market Intelligence System

This project is an advanced AI-driven platform for analyzing the Saudi Stock Market (TASI).

## Core Capabilities:
- **Dual AI Engine**: Simultaneous Daily and Weekly price predictions using Random Forest Regressors.
- **Institutional Analytics**: Risk-adjusted metrics (Sharpe, Beta) and advanced technical confluence.
- **Macro Integration**: Real-time correlation with Brent Oil prices.
- **Explainable AI**: Automated technical reasoning for every recommendation.
- **Full Backtesting**: Professional performance reporting with transaction cost simulation.
- **FastAPI Framework**: High-performance backend with a modern Arabic RTL dashboard.
