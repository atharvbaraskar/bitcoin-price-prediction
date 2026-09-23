import streamlit as st
import pandas as pd
import pickle
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="₿",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("₿ Bitcoin Price Prediction")

st.write(
    "Predict Bitcoin closing price using cryptocurrency "
    "market data from BTC, ETH, USDT and BNB."
)

st.divider()


# ============================================================
# USER INPUTS
# ============================================================

st.subheader("Enter Cryptocurrency Market Data")

btc_volume = st.number_input(
    "Bitcoin Volume",
    min_value=0.0,
    value=10000000000.0
)

eth_close = st.number_input(
    "Ethereum Closing Price (USD)",
    min_value=0.0,
    value=3000.0
)

eth_volume = st.number_input(
    "Ethereum Volume",
    min_value=0.0,
    value=1000000000.0
)

usdt_close = st.number_input(
    "Tether Closing Price (USD)",
    min_value=0.0,
    value=1.0
)

usdt_volume = st.number_input(
    "Tether Volume",
    min_value=0.0,
    value=50000000000.0
)

bnb_close = st.number_input(
    "BNB Closing Price (USD)",
    min_value=0.0,
    value=600.0
)

bnb_volume = st.number_input(
    "BNB Volume",
    min_value=0.0,
    value=500000000.0
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "Volume (BTC)": [btc_volume],

    "Close (ETH)": [eth_close],
    "Volume (ETH)": [eth_volume],

    "Close (USDT)": [usdt_close],
    "Volume (USDT)": [usdt_volume],

    "Close (BNB)": [bnb_close],
    "Volume (BNB)": [bnb_volume]
})


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Bitcoin Price",
    type="primary"
):

    # Scale input using saved scaler
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)

    predicted_price = prediction[0]

    st.success(
        f"Predicted Bitcoin Closing Price: "
        f"${predicted_price:,.2f}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Model: Random Forest Regressor"
)