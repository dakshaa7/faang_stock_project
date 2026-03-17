# 📈 FAANG+ Stock Analysis & Prediction Dashboard

> End-to-end Financial Analytics System built with 
> Python, SQL, Machine Learning & Streamlit

---

## 🎯 Project Overview

Comprehensive analysis of 6 major tech stocks 
(AAPL, AMZN, GOOGL, META, MSFT, NVDA) covering 
10 years of historical data (2016–2026).

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core programming |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML Models |
| XGBoost | Advanced ML |
| SQLite & SQL | Database queries |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| yFinance | Live stock data |

---

## 📊 Project Structure
```
├── faang_eda.ipynb      # EDA + ML Notebook
├── app.py               # Streamlit Dashboard
├── faang_stock_prices.csv  # Dataset
├── model_lr.pkl         # Linear Regression Model
├── model_xgb.pkl        # XGBoost Model
└── scaler.pkl           # Feature Scaler
```

---

## 🔍 Key Features

### 1. Exploratory Data Analysis
- 10-year price trend analysis
- Volume pattern analysis
- Daily returns distribution
- Correlation heatmap
- Technical indicators (RSI, MACD, Bollinger Bands)

### 2. Machine Learning Models
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | $2.04 | $3.86 | 0.9991 |
| XGBoost | $2.32 | $4.39 | 0.9988 |

### 3. SQL Business Analysis
- Total returns by stock (2016-2026)
- Top 10 worst single-day crashes
- Top 10 best single-day gains
- Yearly average returns
- Volatility analysis
- $10,000 ROI calculator

### 4. Business Insights
- **NVDA** delivered 26,866% returns (2016-2026)
- $10,000 invested in NVDA in 2016 = **$2,696,603** today
- **AAPL & MSFT** = Best risk-adjusted returns
- **META** = Biggest crash (2022) + Biggest recovery (2023)

---

## 💰 $10,000 Investment ROI (2016-2026)

| Stock | Final Value | Total Profit |
|-------|-------------|--------------|
| NVDA | $2,696,603 | +$2,686,603 |
| AAPL | $138,900 | +$128,900 |
| MSFT | $125,801 | +$115,801 |
| GOOGL | $99,395 | +$89,395 |
| AMZN | $92,016 | +$82,016 |
| META | $89,341 | +$79,341 |

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/dakshaa7/faang-stock-analysis.git
cd faang-stock-analysis
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit plotly yfinance joblib
```

### 3. Run Dashboard
```bash
streamlit run app.py
```

---

## 📈 Dashboard Features

- **Tab 1** — Live stock price & candlestick chart
- **Tab 2** — Technical indicators (RSI, MACD, Bollinger Bands)
- **Tab 3** — ML predictions for next-day closing price
- **Tab 4** — Business insights & ROI analysis
- **Tab 5** — Model performance comparison

---

## 🎓 Author

**Daksha Saini**  
Data Analyst | Fintech Enthusiast  
[GitHub](https://github.com/dakshaa7)
