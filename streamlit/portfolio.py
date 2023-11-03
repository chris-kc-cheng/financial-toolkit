import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)

import pandas as pd
import numpy as np
import streamlit as st
import toolkit as ftk

tickers = {
    "VFIAX": "US Large Cap",
    "VSMAX": "US Small Cap",
    "VTMGX": "Intl DM Equities",
    "VEMAX": "EM Equities",
    "VBTLX": "US IG Bonds",
    "VWEAX": "US HY Bonds",
    "VTABX": "DM IG Bonds",
    "VGSLX": "REITs",
    "IAU": "Gold",
}


@st.cache_data(ttl=3600)
def get_data():
    px = ftk.get_yahoo_bulk(tickers.keys()).rename(columns=tickers)
    return px.resample("M").last().pct_change().dropna().to_period("M")


if "data" not in st.session_state:
    st.session_state.data = get_data()

data = st.session_state.data

with st.sidebar:
    horizon = st.select_slider(
        "Sample period",
        options=data.index,
        value=[data.index[0], data.index[len(data) // 2]],
    )
    rfr = st.slider(
        "Risk-free rate", value=0.05, min_value=0.0, max_value=0.1, step=0.01
    )
    bounds = st.slider(
        "Bounds", value=(0.0, 1.0), min_value=-3.0, max_value=3.0, step=0.1
    )
    show = st.toggle("Show Efficient Frontier")

st.title("Portfolio Optimization")

assets = st.multiselect("Select asset classes", tickers.values(), tickers.values())


if len(assets) > 1:
    if horizon[1] > horizon[0] and horizon[1] < data.index[-1]:
        col1, col2 = st.columns(2)

        col1.header("In Sample")
        col1.write(f"From {horizon[0]} to {horizon[1]}")

        col2.header("Out of Sample")
        col2.write(f"From {data[horizon[1]:].index[1]} to {data.index[-1]}")

    else:
        st.header("Invalid sample period")
else:
    st.header("Please select at least two asset classes")
