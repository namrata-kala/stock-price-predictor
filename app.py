import streamlit as st
import yfinance as yf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Price Predictor")

stock = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):

    data = yf.download(stock, start="2015-01-01")

    st.subheader("Recent Stock Data")
    st.write(data.tail())

    st.line_chart(data['Close'])