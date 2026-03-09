import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.title("📈 Stock Price Predictor")

# load model and scaler
model = load_model("stock_model.keras")
scaler = pickle.load(open("scaler.pkl","rb"))

stock = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict Future Prices"):

    data = yf.download(stock, start="2015-01-01")

    st.subheader("Recent Stock Data")
    st.write(data.tail())

    st.line_chart(data['Close'])

    features = data[['Open','High','Low','Close','Volume']]

    scaled_data = scaler.fit_transform(features)

    last_60 = scaled_data[-60:]
    X_input = np.array([last_60])

    prediction = model.predict(X_input)

    pred_full = np.zeros((1,5))
    pred_full[:,3] = prediction

    predicted_price = scaler.inverse_transform(pred_full)[0][3]

    st.subheader("Predicted Next Day Price")
    st.write(f"${predicted_price:.2f}")