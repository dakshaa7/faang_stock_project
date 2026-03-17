# =============================================================
# FAANG+ STOCK ANALYSIS DASHBOARD
# Complete Financial Analytics & ML Prediction System
# Dataset: FAANG+ Historical Stock Prices (2016-2026)
# Author: Daksha saini
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title = "FAANG+ Stock Analysis Dashboard",
    page_icon  = "📈",
    layout     = "wide"
)

# ── Load ML Models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr     = joblib.load('model_lr.pkl')
    xgb    = joblib.load('model_xgb.pkl')
    scaler = joblib.load('scaler.pkl')
    return lr, xgb, scaler

model_lr, model_xgb, scaler = load_models()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"],
    index=0
)

period = st.sidebar.selectbox(
    "Select Period",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    "End-to-end Financial Analytics Dashboard "
    "built with Python, SQL, and Machine Learning. "
    "Analyzes 10 years of FAANG+ stock data."
)

# ── Fetch Live Data ───────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker, period):
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    return df

df = fetch_data(ticker, period)

# ── Header ────────────────────────────────────────────────────
st.title("📈 FAANG+ Stock Analysis Dashboard")
st.markdown(
    "**End-to-end Financial Analytics System** | "
    "Python • SQL • Machine Learning • Live Data"
)
st.divider()

# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stock Overview",
    "🔍 Technical Analysis",
    "🤖 ML Prediction",
    "💰 Business Insights",
    "📋 Model Performance"
])

# =============================================================
# TAB 1 — STOCK OVERVIEW
# =============================================================
with tab1:
    st.subheader(f"📈 {ticker} — Stock Overview")

    # ── Live Metrics ──────────────────────────────────────────
    latest     = df.iloc[-1]
    prev       = df.iloc[-2]
    price_diff = latest['Close'] - prev['Close']
    pct_change = (price_diff / prev['Close']) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price",
                f"${latest['Close']:.2f}",
                f"{price_diff:+.2f}")
    col2.metric("Day Change",
                f"{pct_change:+.2f}%")
    col3.metric("Volume",
                f"{latest['Volume']:,.0f}")
    col4.metric("52W High",
                f"${df['High'].max():.2f}")

    st.divider()

    # ── Candlestick + Volume ──────────────────────────────────
    st.subheader("Candlestick Chart + Volume")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df['Open'],
        high  = df['High'],
        low   = df['Low'],
        close = df['Close'],
        increasing_line_color = '#26a69a',
        decreasing_line_color = '#ef5350',
        name  = 'Price'
    ), row=1, col=1)

    colors_vol = [
        '#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i]
        else '#ef5350' for i in range(len(df))
    ]

    fig.add_trace(go.Bar(
        x            = df.index,
        y            = df['Volume'],
        marker_color = colors_vol,
        name         = 'Volume',
        opacity      = 0.7
    ), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible = False,
        height     = 550,
        margin     = dict(l=20, r=20, t=20, b=20),
        showlegend = False
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",   row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# =============================================================
# TAB 2 — TECHNICAL ANALYSIS
# =============================================================
with tab2:
    st.subheader(f"🔍 {ticker} — Technical Indicators")

    # ── Calculate Indicators ──────────────────────────────────
    # RSI
    delta  = df['Close'].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss
    rsi    = 100 - (100 / (1 + rs))

    # MACD
    ema12  = df['Close'].ewm(span=12).mean()
    ema26  = df['Close'].ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    # Bollinger Bands
    sma20  = df['Close'].rolling(20).mean()
    std20  = df['Close'].rolling(20).std()
    upper  = sma20 + (std20 * 2)
    lower  = sma20 - (std20 * 2)

    fig2, axes = plt.subplots(3, 1, figsize=(14, 12))

    # RSI Plot
    axes[0].plot(df.index, rsi,
                 color='purple', linewidth=1.5, label='RSI')
    axes[0].axhline(y=70, color='red',
                    linestyle='--', alpha=0.7, label='Overbought (70)')
    axes[0].axhline(y=30, color='green',
                    linestyle='--', alpha=0.7, label='Oversold (30)')
    axes[0].fill_between(df.index, 70, 100,
                         alpha=0.1, color='red')
    axes[0].fill_between(df.index, 0, 30,
                         alpha=0.1, color='green')
    axes[0].set_title(f'{ticker} — RSI (14)',
                      fontweight='bold')
    axes[0].set_ylabel('RSI')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # MACD Plot
    axes[1].plot(df.index, macd,
                 color='blue', linewidth=1.5, label='MACD')
    axes[1].plot(df.index, signal,
                 color='red', linewidth=1.5, label='Signal')
    axes[1].bar(df.index, macd - signal,
                color=['#26a69a' if v >= 0 else '#ef5350'
                       for v in (macd - signal)],
                alpha=0.5, label='Histogram')
    axes[1].axhline(y=0, color='black',
                    linewidth=0.8, linestyle='--')
    axes[1].set_title(f'{ticker} — MACD',
                      fontweight='bold')
    axes[1].set_ylabel('MACD')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Bollinger Bands
    axes[2].plot(df.index, df['Close'],
                 color='steelblue', linewidth=1.5,
                 label='Close Price')
    axes[2].plot(df.index, upper,
                 color='red', linewidth=1,
                 linestyle='--', label='Upper Band')
    axes[2].plot(df.index, sma20,
                 color='orange', linewidth=1,
                 label='SMA 20')
    axes[2].plot(df.index, lower,
                 color='green', linewidth=1,
                 linestyle='--', label='Lower Band')
    axes[2].fill_between(df.index, upper, lower,
                         alpha=0.1, color='gray')
    axes[2].set_title(f'{ticker} — Bollinger Bands',
                      fontweight='bold')
    axes[2].set_ylabel('Price ($)')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)

    # ── Current Signal Summary ────────────────────────────────
    st.divider()
    st.subheader("📊 Current Signal Summary")

    current_rsi  = rsi.iloc[-1]
    current_macd = macd.iloc[-1]
    current_sig  = signal.iloc[-1]

    c1, c2, c3 = st.columns(3)

    rsi_signal = ("🔴 Overbought" if current_rsi > 70
                  else "🟢 Oversold" if current_rsi < 30
                  else "🟡 Neutral")
    macd_signal = ("🟢 Bullish" if current_macd > current_sig
                   else "🔴 Bearish")

    c1.metric("RSI (14)",
              f"{current_rsi:.2f}", rsi_signal)
    c2.metric("MACD Signal", macd_signal)
    c3.metric("Price vs SMA20",
              f"${df['Close'].iloc[-1]:.2f}",
              f"SMA20: ${sma20.iloc[-1]:.2f}")

# =============================================================
# TAB 3 — ML PREDICTION
# =============================================================
with tab3:
    st.subheader(f"🤖 {ticker} — Next Day Price Prediction")

    # ── Feature Engineering ───────────────────────────────────
    close      = df['Close'].iloc[-1]
    close_lag1 = df['Close'].iloc[-2]
    close_lag2 = df['Close'].iloc[-3]
    close_lag3 = df['Close'].iloc[-4]
    volume     = df['Volume'].iloc[-1]
    price_range = df['High'].iloc[-1] - df['Low'].iloc[-1]

    delta2     = df['Close'].diff()
    gain2      = delta2.clip(lower=0).rolling(14).mean()
    loss2      = (-delta2.clip(upper=0)).rolling(14).mean()
    rs2        = gain2 / loss2
    rsi_14     = float(100 - (100 / (1 + rs2.iloc[-1])))

    ema12_v    = df['Close'].ewm(span=12).mean()
    ema26_v    = df['Close'].ewm(span=26).mean()
    macd_val   = float(ema12_v.iloc[-1] - ema26_v.iloc[-1])
    sma20_v    = df['Close'].rolling(20).mean()
    sma_diff   = float(close - sma20_v.iloc[-1])
    volatility = float(
        df['Close'].pct_change().rolling(7).std().iloc[-1]
    )

    input_data = pd.DataFrame([{
        'Close'        : close,
        'Volume'       : volume,
        'RSI_14'       : rsi_14,
        'MACD'         : macd_val,
        'Close_Lag1'   : close_lag1,
        'Close_Lag2'   : close_lag2,
        'Close_Lag3'   : close_lag3,
        'Price_Range'  : price_range,
        'SMA_Diff'     : sma_diff,
        'Volatility_7d': volatility
    }])

    # Predictions
    input_scaled = scaler.transform(input_data)
    pred_lr      = model_lr.predict(input_scaled)[0]
    pred_xgb     = model_xgb.predict(input_data)[0]
    avg_pred     = (pred_lr + pred_xgb) / 2

    margin      = abs(pred_lr - pred_xgb) / 2
    lower_bound = avg_pred - margin
    upper_bound = avg_pred + margin
    signal_dir  = "BUY 🟢" if avg_pred > close else "SELL 🔴"
    signal_pct  = ((avg_pred - close) / close) * 100

    # ── Results ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression",
                f"${pred_lr:.2f}",
                f"{pred_lr - close:+.2f}")
    col2.metric("XGBoost",
                f"${pred_xgb:.2f}",
                f"{pred_xgb - close:+.2f}")
    col3.metric("Signal",
                signal_dir,
                f"{signal_pct:+.2f}%")

    st.divider()
    st.markdown("#### 📊 Prediction Range")

    col1, col2, col3 = st.columns(3)
    col1.metric("Lower Bound", f"${lower_bound:.2f}")
    col2.metric("Avg Prediction", f"${avg_pred:.2f}")
    col3.metric("Upper Bound", f"${upper_bound:.2f}")

    # ── Prediction Chart ──────────────────────────────────────
    last_30      = df['Close'].tail(30)
    future_index = [last_30.index[-1] + pd.Timedelta(days=1)]

    fig3, ax = plt.subplots(figsize=(13, 5))
    ax.plot(last_30.index, last_30.values,
            color='steelblue', linewidth=2,
            label='Actual Price')
    ax.scatter(future_index, [avg_pred],
               color='tomato', s=120,
               zorder=5, label='Predicted Price')
    ax.errorbar(future_index, [avg_pred],
                yerr=[[avg_pred - lower_bound],
                      [upper_bound - avg_pred]],
                fmt='o', color='tomato',
                capsize=8, capthick=2, linewidth=2)
    ax.axhline(y=close, color='gray',
               linestyle='--', alpha=0.5,
               label=f"Today: ${close:.2f}")
    ax.set_title(
        f"{ticker} — Last 30 Days + Next Day Prediction",
        fontsize=13, fontweight='bold'
    )
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

# =============================================================
# TAB 4 — BUSINESS INSIGHTS
# =============================================================
with tab4:
    st.subheader("💰 Business Insights — 10 Year Analysis")

    insight_tab1, insight_tab2, insight_tab3 = st.tabs([
        "💵 ROI Analysis",
        "⚖️ Risk vs Reward",
        "📅 Yearly Performance"
    ])

    with insight_tab1:
        st.markdown("#### $10,000 Invested in 2016 → Value in 2026")

        tickers_roi = ['NVDA','AAPL','MSFT',
                        'GOOGL','AMZN','META']
        values      = [2696603, 138900, 125801,
                       99395, 92016, 89341]
        colors_roi  = ['#00C851','#2196F3','#2196F3',
                       '#FF9800','#FF9800','#FF9800']

        fig4, ax4 = plt.subplots(figsize=(12, 6))
        bars = ax4.bar(tickers_roi, values,
                       color=colors_roi,
                       edgecolor='white', linewidth=0.8)

        for bar, val in zip(bars, values):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20000,
                f'${val:,.0f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        ax4.axhline(y=10000, color='red',
                    linestyle='--', linewidth=2,
                    label='Initial Investment ($10,000)')
        ax4.set_title(
            '$10,000 Invested in 2016 → Value in 2026',
            fontsize=13, fontweight='bold'
        )
        ax4.set_ylabel('Final Value ($)')
        ax4.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, p: f'${x:,.0f}'
            )
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)

        # ROI Table
        roi_df = pd.DataFrame({
            'Stock'        : tickers_roi,
            'Initial ($10K)': ['$10,000'] * 6,
            'Final Value'  : [f'${v:,.0f}' for v in values],
            'Total Profit' : [f'${v-10000:,.0f}'
                              for v in values],
            'Return'       : ['26,866%','1,289%','1,158%',
                              '894%','820%','793%']
        })
        st.dataframe(roi_df,
                     hide_index=True,
                     use_container_width=True)

    with insight_tab2:
        st.markdown("#### Risk vs Reward — Which Stock Is Worth It?")

        tickers_rr = ['NVDA','META','AMZN',
                       'GOOGL','AAPL','MSFT']
        volatility = [2.75, 1.99, 1.77, 1.60, 1.56, 1.45]
        returns_rr = [26866, 793, 820, 894, 1289, 1158]
        colors_rr  = ['#FF5722','#E91E63','#FF9800',
                      '#2196F3','#4CAF50','#9C27B0']

        fig5, ax5 = plt.subplots(figsize=(10, 7))

        for i, t in enumerate(tickers_rr):
            ax5.scatter(volatility[i], returns_rr[i],
                        color=colors_rr[i], s=300,
                        zorder=5,
                        edgecolors='white', linewidth=2)
            ax5.annotate(t,
                         (volatility[i], returns_rr[i]),
                         textcoords="offset points",
                         xytext=(12, 5),
                         fontsize=12, fontweight='bold',
                         color=colors_rr[i])

        ax5.axvline(x=1.8, color='gray',
                    linestyle='--', alpha=0.5)
        ax5.axhline(y=1000, color='gray',
                    linestyle='--', alpha=0.5)
        ax5.set_title(
            'Risk vs Reward — FAANG+ (2016-2026)',
            fontsize=13, fontweight='bold'
        )
        ax5.set_xlabel('Avg Daily Volatility (%)')
        ax5.set_ylabel('Total Return (%)')
        ax5.grid(True, alpha=0.3)
        ax5.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)

    with insight_tab3:
        st.markdown("#### Year-by-Year Performance Heatmap")

        yearly_data = {
            'AAPL' : [0.10, 0.16,-0.01, 0.27, 0.16,
                      0.16,-0.14, 0.27, 0.08, 0.10],
            'AMZN' : [0.14, 0.19, 0.13, 0.09, 0.16,
                      0.17,-0.19, 0.12, 0.08, 0.14],
            'GOOGL': [0.04, 0.12, 0.01, 0.11, 0.12,
                      0.15,-0.12, 0.17, 0.09, 0.04],
            'META' : [0.04, 0.18,-0.09, 0.10, 0.18,
                      0.20,-0.31, 0.30, 0.12, 0.04],
            'MSFT' : [0.09, 0.14, 0.09, 0.14, 0.17,
                      0.19,-0.10, 0.22, 0.09, 0.09],
            'NVDA' : [0.60, 0.27,-0.10, 0.14, 0.17,
                      0.43,-0.19, 0.65, 0.19, 0.60],
        }
        years = [2016,2017,2018,2019,
                 2020,2021,2022,2023,2024,2025]

        heatmap_df = pd.DataFrame(yearly_data, index=years)

        fig6, ax6 = plt.subplots(figsize=(13, 6))
        sns.heatmap(
            heatmap_df.T,
            annot=True, fmt='.2f',
            cmap='RdYlGn', center=0,
            linewidths=0.5, linecolor='white',
            ax=ax6
        )
        ax6.set_title(
            'FAANG+ Yearly Performance (2016-2025)\n'
            'Green = Positive | Red = Negative',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        st.pyplot(fig6)

# =============================================================
# TAB 5 — MODEL PERFORMANCE
# =============================================================
with tab5:
    st.subheader("📋 ML Model Performance Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression MAE", "$2.04",
                "Lower is better ✅")
    col2.metric("XGBoost MAE", "$2.32")
    col3.metric("Best Model", "Linear Regression 🏆")

    st.divider()

    perf_df = pd.DataFrame({
        'Model'    : ['Linear Regression', 'XGBoost'],
        'MAE ($)'  : [2.04, 2.32],
        'RMSE ($)' : [3.86, 4.39],
        'R² Score' : [0.9991, 0.9988],
        'Winner'   : ['🏆', '']
    })
    st.dataframe(perf_df,
                 hide_index=True,
                 use_container_width=True)

    st.divider()
    st.markdown("#### 🔍 Key Findings")

    st.info(
        "**Why Linear Regression outperformed XGBoost?**\n\n"
        "The dataset is dominated by lag features (Close_Lag1, "
        "Close_Lag2, Close_Lag3) which create a near-linear "
        "relationship. Linear Regression handles this efficiently "
        "while XGBoost's complexity introduces slight overfitting."
    )

    st.success(
        "**Model Accuracy:** On average, predictions are "
        "within $2.04 of the actual next-day closing price — "
        "demonstrating strong predictive capability across "
        "all 6 FAANG+ stocks."
    )