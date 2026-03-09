import streamlit as st
import yfinance as yf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Price Predictor")

# Load model safely
try:
    model = pickle.load(open("model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
except:
    st.error("Model or scaler file not found!")
    st.stop()

stock = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict Future Prices"):

    st.write("Downloading stock data...")

    data = yf.download(stock, start="2015-01-01")

    if data.empty:
        st.error("Invalid stock symbol!")
        st.stop()

    st.subheader("Recent Stock Data")
    st.write(data.tail())

    st.subheader("Stock Price Chart")
    st.line_chart(data['Close'])

    # select features
    features = data[['Open','High','Low','Close','Volume']]

    # scale
    scaled_data = scaler.fit_transform(features)

    # last 60 days
    last_60 = scaled_data[-60:]

    X_input = np.array([last_60])

    # prediction
    prediction = model.predict(X_input)

    # rebuild structure for inverse scaling
    pred_full = np.zeros((1,5))
    pred_full[:,3] = prediction

    predicted_price = scaler.inverse_transform(pred_full)[0][3]

    st.subheader("Predicted Next Day Price")
    st.success(f"${predicted_price:.2f}")