import streamlit as st
import pandas as pd
import yfinance as yf
import pickle

st.title("📈 Stock Return Prediction App")

st.write("Enter a stock ticker to predict the next-day return.")

# User input
ticker = st.text_input(
    "Stock Ticker (Example: AAPL, TSLA, RELIANCE.NS)",
    "AAPL"
)

# Load trained model
with open("models/linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

if st.button("Predict"):

    # Download stock data
    data = yf.download(ticker, period="1y")

    if data.empty:
        st.error("Invalid ticker or no data available.")
    else:

        df = data.copy()

        # -------- Feature Engineering --------

        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        df["lag1"] = df["Close"].shift(1)
        df["lag2"] = df["Close"].shift(2)
        df["lag3"] = df["Close"].shift(3)

        df["return"] = df["Close"].pct_change()
        df["volatility"] = df["return"].rolling(10).std()

        # -------- RSI Calculation --------

        delta = df["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df = df.dropna()

        latest = df.iloc[-1]

        # -------- Model Input --------

        input_data = pd.DataFrame({
            "lag1":[latest["lag1"]],
            "lag2":[latest["lag2"]],
            "lag3":[latest["lag3"]],
            "MA10":[latest["MA10"]],
            "MA50":[latest["MA50"]],
            "volume":[latest["Volume"]],
            "volatility":[latest["volatility"]],
            "RSI":[latest["RSI"]]
        })

        # -------- Prediction --------

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction > 0:
            st.success(f"Predicted Return: {prediction:.4f} 📈 Bullish")
        else:
            st.warning(f"Predicted Return: {prediction:.4f} 📉 Bearish")

        # -------- Stock Chart --------

        st.subheader("Stock Price Chart (Last 1 Year)")
        st.line_chart(df["Close"])