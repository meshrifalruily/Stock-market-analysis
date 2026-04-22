# TASI AI Stock Prediction App

This project uses AI to analyze the Saudi Stock Market (TASI). It predicts the top 3 companies likely to rise on the next trading day and provides entry, take-profit, and stop-loss points.

## Project Structure
- `data/`: Contains raw and processed stock data.
- `models/`: Contains the trained Random Forest model and feature definitions.
- `scripts/`:
    - `fetch_data.py`: Downloads historical data from Yahoo Finance.
    - `preprocess.py`: Calculates technical indicators (SMA, RSI, MACD, ATR).
    - `train_model.py`: Trains the Random Forest model to predict next-day returns.
- `app/`:
    - `main.py`: The Streamlit dashboard.

## How to Run

### 1. Install Dependencies
```bash
python3 -m pip install yfinance pandas numpy plotly streamlit scikit-learn joblib
```

### 2. Update Data and Train Model
Run these scripts in order to get the latest data and refresh the AI model:
```bash
python3 scripts/fetch_data.py
python3 scripts/preprocess.py
python3 scripts/train_model.py
```

### 3. Launch the App
```bash
streamlit run app/main.py
```

## AI Model & Logic
- **Model**: Random Forest Regressor.
- **Features**: Close price, Volume, SMAs (20, 50, 200), RSI, MACD, and ATR.
- **Prediction**: The model predicts the percentage return for the next trading day.
- **Entry Point**: Current closing price.
- **Take Profit**: Based on the predicted return.
- **Stop Loss**: Calculated as `Current Price - 2 * ATR` to allow for normal volatility.
