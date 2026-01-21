# 📈 Stock Analyzer Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stockanlayserpro-aryank.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)

A professional-grade stock market analysis and AI-powered price prediction system specifically designed for **BSE** (Bombay Stock Exchange) and **NSE** (National Stock Exchange) stocks.

## 🌟 Features

### 🇮🇳 Indian Market Focus

- **NSE Stocks**: Use `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`)
- **BSE Stocks**: Use `.BO` suffix (e.g., `RELIANCE.BO`, `TCS.BO`)
- Pre-loaded with 30+ NIFTY 50 stocks for quick selection
- INR (₹) currency formatting throughout

### 📊 Price Analysis

- Interactive candlestick charts with volume
- **Monthly returns heatmap** showing seasonal patterns
- Cumulative returns visualization
- Drawdown analysis

### 📉 Technical Indicators

- **RSI** (Relative Strength Index) with overbought/oversold signals
- **MACD** with bullish/bearish crossover detection
- **Bollinger Bands** with price channel visualization
- **Moving Averages** table (5, 10, 20, 50, 200-day)

### ⚠️ Advanced Risk Analysis

| Feature                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| **Risk Gauge**           | Overall risk score (0-100) with color-coded levels |
| **Volatility Cone**      | Current vs historical volatility across timeframes |
| **VaR Distribution**     | Value at Risk with 95% & 99% confidence thresholds |
| **Rolling Risk Metrics** | 20-day volatility, Sharpe ratio, max drawdown      |
| **Beta Analysis**        | Correlation and beta with NIFTY 50 benchmark       |

### 🔮 ML Forecasting (Experimental)

> Note: Stock price prediction is inherently uncertain. These models are for educational demonstration.

- **Prophet**: Facebook's time-series forecasting
- **XGBoost**: Gradient boosting with technical indicators
- Model comparison mode
- Confidence intervals for all predictions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aryankule-08-a11y/stock_anlayser_pro.git

# Navigate to the project directory
cd stock_analyser_pro

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python -m streamlit run app.py
```

Open browser: **http://localhost:8501**

## 📁 Project Structure

```
Stock_Analyzer/
├── app.py                    # Streamlit dashboard (BSE/NSE focused)
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── data/                     # Data layer
│   ├── fetcher.py           # Yahoo Finance API
│   └── preprocessor.py      # Data cleaning
├── features/                 # Technical indicators
│   └── indicators.py
├── analysis/                 # Analysis modules
│   ├── eda.py               # Visualizations
│   └── risk.py              # Risk metrics
├── models/                   # ML models
│   ├── prophet_model.py
│   ├── xgboost_model.py
│   └── lstm_model.py
└── forecasting/              # Prediction engine
    └── predictor.py
```

## 📈 Supported Stocks

### NSE (National Stock Exchange)

- RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS
- SBIN.NS, BHARTIARTL.NS, ITC.NS, KOTAKBANK.NS, LT.NS
- WIPRO.NS, ASIANPAINT.NS, MARUTI.NS, TATASTEEL.NS, SUNPHARMA.NS
- And 15+ more NIFTY 50 stocks

### BSE (Bombay Stock Exchange)

- RELIANCE.BO, TCS.BO, HDFCBANK.BO, INFY.BO, ICICIBANK.BO
- And more...

### Indices

- ^NSEI (NIFTY 50), ^BSESN (BSE SENSEX), ^NSEBANK (NIFTY Bank)

## ⚠️ Disclaimer

> **IMPORTANT**: This application is for educational and demonstration purposes only.
>
> - Predictions are NOT investment advice
> - Stock markets are inherently unpredictable
> - Past performance does not guarantee future results
> - Always consult a **SEBI registered financial advisor** before investing
> - The creators assume no liability for financial losses

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for Indian investors
</p>
