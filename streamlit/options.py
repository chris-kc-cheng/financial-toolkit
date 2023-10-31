"""_summary_
"""
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)

import json
import pandas as pd
import numpy as np
import streamlit as st
import toolkit as ftk

def bound(num):
    if abs(num) > 1e8:
        return 'Unlimited'
    else:
        return f'{num:.2f}'


strategy = 'Custom Option Strategy'
strategies = json.load(open('streamlit/data/option_strategy.json'))

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(strategies[0])

with st.sidebar:
    
    option = st.selectbox(
        'Select option strategy',
        [s['name'] for s in strategies])
    
    for s in strategies:
        if option == s['name']:
            strategy = s['name']
            st.session_state.data = pd.DataFrame(s['instruments'])
    
    vol = st.slider('Volatility', min_value=0.01, max_value=1., value=0.2, format='%f')
    time = st.slider('Time to expiration', min_value=0.01, max_value=1., value=0.25, format='%f')
    rate = st.slider('Risk-free rate', min_value=0., max_value=1., value=0.05, format='%f')
    dvd = st.slider('Dividend yield', min_value=0., max_value=1., value=0., format='%f')

    entry = st.slider('Entry Point', min_value=0, max_value=100, value=50, format='%f')

st.title(strategy)
edited_df = st.data_editor(st.session_state.data,
                           num_rows='dynamic',
                           column_config={
                               'name': st.column_config.TextColumn('Name'),
                               'instrument': st.column_config.SelectboxColumn('Type', options=['Call', 'Put', 'Stock', 'Debt']),
                               'strike': st.column_config.NumberColumn('Strike/Face', min_value=0.1),
                               'qty': st.column_config.NumberColumn('Quantity'),
                           },
                           column_order=['name', 'instrument', 'strike', 'qty'],
                           use_container_width=True,
                           hide_index=True,)

strikes = pd.concat([edited_df['strike'], pd.Series([1e-10, 1e10])])

spot  = np.linspace(0, 100, 100)
value = []
delta = []
gamma = []
vega  = []
theta = []

payoffs = np.zeros_like(strikes)
premium = 0

for _, row in edited_df.iterrows():
    if row.instrument:
        qty = row.qty

        # Use nested if-else as Streamlit use Python 3.9 while match/case was introduced in Python 3.10
        if row.instrument == 'Call':
            c = ftk.EuropeanCall(None, row.strike)
            value.append(pd.Series(qty * c.price(spot, rate, time, vol, dvd), name=row['name']))
            delta.append(pd.Series(qty * c.delta(spot, rate, time, vol, dvd), name=row['name']))
            gamma.append(pd.Series(qty * c.gamma(spot, rate, time, vol, dvd), name=row['name']))
            vega .append(pd.Series(qty * c.vega (spot, rate, time, vol, dvd), name=row['name']))
            theta.append(pd.Series(qty * c.theta(spot, rate, time, vol, dvd), name=row['name']))
            premium += qty * c.price(entry, rate, time, vol, dvd)
            payoffs += qty * c.moneyness(strikes)
        elif row.instrument == 'Put':
            p = ftk.EuropeanPut(None, row.strike)
            value.append(pd.Series(qty * p.price(spot, rate, time, vol, dvd), name=row['name']))
            delta.append(pd.Series(qty * p.delta(spot, rate, time, vol, dvd), name=row['name']))
            gamma.append(pd.Series(qty * p.gamma(spot, rate, time, vol, dvd), name=row['name']))
            vega .append(pd.Series(qty * p.vega (spot, rate, time, vol, dvd), name=row['name']))
            theta.append(pd.Series(qty * p.theta(spot, rate, time, vol, dvd), name=row['name']))
            premium += qty * p.price(entry, rate, time, vol, dvd)
            payoffs += qty * p.moneyness(strikes)
        elif row.instrument == 'Stock':
            value.append(pd.Series(qty * spot * np.exp(-dvd * time), name=row['name']))
            delta.append(pd.Series(qty * np.ones_like(spot) * np.exp(-dvd * time), name=row['name']))
            theta.append(pd.Series(qty * dvd * spot * np.exp(-dvd * time), name=row['name']))
            payoffs += qty * (strikes - entry) + entry * (1 - np.exp(-dvd * time))
        elif row.instrument == 'Debt':
            value.append(pd.Series(qty * np.ones_like(spot) * row.strike * np.exp(-rate * time), name=row['name']))                
            theta.append(pd.Series(qty * np.ones_like(spot) * rate * row.strike * np.exp(-rate * time), name=row['name']))
            payoffs += qty * row.strike * (1 - np.exp(-rate * time))

value_df = pd.DataFrame(value).T
value_df[strategy] = value_df.sum(axis=1)
delta_df = pd.DataFrame(delta).T
delta_df[strategy] = delta_df.sum(axis=1)
gamma_df = pd.DataFrame(gamma).T
gamma_df[strategy] = gamma_df.sum(axis=1)
vega_df = pd.DataFrame(vega).T
vega_df[strategy] = vega_df.sum(axis=1)
theta_df = pd.DataFrame(theta).T
theta_df[strategy] = theta_df.sum(axis=1)

payoffs -= premium
col1, col2, col3 = st.columns(3)
col1.metric("Premium Paid", f'{premium:.2f}')
col2.metric("Maximum Gain", bound(payoffs.max()))
col3.metric("Maximum Loss", bound(payoffs.min()))

st.header('Price')
st.line_chart(value_df)

col1, col2 = st.columns(2)
col1.header('Delta')
col1.line_chart(delta_df)

col1.header('Theta')
col1.line_chart(theta_df)

col2.header('Gamma')
col2.line_chart(gamma_df)

col2.header('Vega')
col2.line_chart(vega_df)

st.markdown(open('streamlit/data/signature.md').read())