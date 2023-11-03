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
    'VFIAX': 'US Large Cap',
    'VSMAX': 'US Small Cap',
    'VTMGX': 'Intl DM Equities',
    'VEMAX': 'EM Equities',
    'VBTLX': 'US IG Bonds',
    'VWEAX': 'US HY Bonds',
    'VTABX': 'DM IG Bonds',
    'VGSLX': 'REITs',
    'IAU': 'Gold',
    'BTC-USD': 'Bitcoin'
}


@st.cache_data(ttl=3600)
def get_data():
    px = ftk.get_yahoo_bulk(tickers.keys()).rename(columns=tickers)
    return px.resample("M").last().pct_change().dropna().to_period("M")


if "data" not in st.session_state:
    st.session_state.data = get_data()

# DataFrame of asset class returns
data = st.session_state.data

with st.sidebar:
    horizon = st.select_slider(
        "Sample period",
        options=data.index,
        value=[data.index[0], data.index[len(data) // 2]],
    )
    rfr = st.slider(
        "Risk-free rate", value=0., min_value=0.0, max_value=0.1, step=0.01
    )
    bounds = st.slider(
        "Bounds", value=(0.05, 0.25), min_value=-1.0, max_value=2.0, step=0.05
    )

st.title("Portfolio Optimization")

assets = st.multiselect("Select asset classes", tickers.values(), list(tickers.values())[0:-1])
st.write(f"From {horizon[0]} to {horizon[1]}")

# Subset of assets and horizon
returns = data[assets]
begin = horizon[0]
end = horizon[-1]
cov = ftk.covariance(returns[begin:end], annualize=True)
er = ftk.compound_return(returns[begin:end], annualize=True)

wtgs = pd.DataFrame({'Equal Weight': ftk.equal_weight(er),                     
                     'Inverse Volatility': ftk.inverse_vol(cov),
                     f'Max. Sharpe ({bounds[0] * 100}-{bounds[1] * 100}%)': ftk.max_sharpe(er, cov, min=bounds[0], max=bounds[1]),
                     'Max. Sharpe (No shorting)': ftk.max_sharpe(er, cov, rfr=rfr, min=0), 
                     'Max. Sharpe (Unconstrained)': ftk.max_sharpe(er, cov, rfr=rfr),
                     'Min. Volatility (Unconstrained)': ftk.min_vol(cov),
                     'Risk Parity': ftk.risk_parity(cov),
                    }, index=cov.index)

if len(assets) > 1:
    if horizon[1] > horizon[0]:
        col1, col2 = st.columns(2)

        col1.header("Weights")
        col1.bar_chart(wtgs.T)

        col2.header("Risk Contribution")
        col2.bar_chart(ftk.risk_contribution(wtgs, cov))

        col1, col2 = st.columns(2)
        col1.header("Asset Class Returns")
        col1.scatter_chart(pd.concat([
            ftk.compound_return(returns, annualize=True),
            ftk.volatility(returns, annualize=True)],
            axis=1, keys=['Return', 'Volatility']).reset_index(),            
            y='Return', x='Volatility', color='index'
            )

        col2.header("Portfolio Returns")
        col2.scatter_chart(pd.concat([
            ftk.portfolio_return(wtgs, er),
            ftk.portfolio_volatility(wtgs, cov)],
            axis=1, keys=['Return', 'Volatility']).reset_index(),            
            y='Return', x='Volatility', color='index'
            )
    else:
        st.header("Invalid sample period")
else:
    st.header("Please select at least two asset classes")
