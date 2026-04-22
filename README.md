# منصة التحليل المالي للأسهم السعودية (TASI) - FastAPI Edition

أداة احترافية مدعومة بالذكاء الاصطناعي وواجهة برمجة تطبيقات (API) لتحليل وتوقع حركة سوق الأسهم السعودي.

## المميزات الجديدة (FastAPI)
- **واجهة برمجية (API)**: نقاط نهاية JSON جاهزة للربط مع تطبيقات الجوال أو الويب.
- **لوحة تحكم فائقة السرعة**: واجهة Tailwind CSS بتصميم حديث ودعم كامل للعربية.
- **تحديثات غير متزامنة**: تحديث البيانات وإعادة التدريب في الخلفية.

## المتطلبات التقنية
```bash
python3 -m pip install fastapi uvicorn jinja2 aiofiles yfinance pandas numpy scikit-learn joblib pandas-ta quantstats
```

## تشغيل المشروع

### 1. تجهيز البيانات (لأول مرة)
```bash
python3 scripts/fetch_data.py && python3 scripts/preprocess.py && python3 scripts/train_model.py && python3 scripts/backtest.py
```

### 2. تشغيل خادم FastAPI
```bash
uvicorn app.main:app --reload
```
سيظهر الموقع على: `http://127.0.0.1:8000`

## نقاط النهاية (Endpoints)
- `GET /`: لوحة التحكم الرئيسية (HTML).
- `GET /api/predictions`: قائمة بجميع التوقعات (JSON).
- `GET /api/stock/{symbol}`: البيانات التاريخية لآخر 100 يوم لسهم معين.
- `POST /api/update`: بدء عملية تحديث شاملة للبيانات والنماذج في الخلفية.
