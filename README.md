# 🧠 Nunno Predictor — ML-Based Crypto Price Movement Forecasting

Nunno Predictor is a beginner-friendly, machine learning-powered tool that forecasts the **next candlestick direction (UP or DOWN)** in the crypto market. It fetches real-time price data from Binance, applies technical indicators, trains a Random Forest model, and delivers both predictions and beginner-level reasoning behind the forecast.

> 🔮 Built as a backend module for **Nunno AI** — your personal trading mentor, developed by [Mujtaba Kazmi](https://github.com/mujtaba-kazmi2006)

---

## 📌 What It Does

- ✅ Pulls historical OHLCV data for any crypto trading pair from **Binance**
- ✅ Computes advanced **technical indicators** using the `ta` library
- ✅ Trains a **Random Forest classifier** to predict the next candle direction
- ✅ Outputs:
  - 🔼 or 🔽 prediction
  - Confidence score
  - Accuracy on test data
  - **Human-readable reasoning** using indicator values (ideal for beginners)

---

## 🧪 Sample Run

```bash
$ python nunno_predictor.py
markdown
Copy
Edit
Select a token to analyze:
1. BTCUSDT
2. ETHUSDT
3. BNBUSDT
4. SOLUSDT
5. DOGEUSDT
6. Enter custom token (e.g., MATICUSDT)
Your choice: 1

Select a timeframe:
1. 1m
2. 5m
3. 15m
4. 1h
5. 4h
Your choice: 3

Fetching data for BTCUSDT (15m)...

Model Accuracy: 85.40%
Prediction for Next Candle: 🔼 UP
Confidence: 73.65%

📊 Nunno's Reasoning:
• RSI is oversold → potential bounce or rally  
• MACD is bullish → trend momentum is up  
• Price is above EMA → uptrend bias  
📈 Technical Indicators Used
Indicator	Purpose
RSI	Overbought/oversold momentum
EMA	Trend bias
MACD	Momentum confirmation
Stochastic RSI	Fast oscillator for reversals
Bollinger Bands	Volatility + squeeze breakouts
ATR	Price range volatility
ADX	Trend strength
Candle Body Size	Momentum from size of price move
