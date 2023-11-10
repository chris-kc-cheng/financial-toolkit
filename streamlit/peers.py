import pandas as pd
import streamlit as st
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)
import toolkit as ftk

@st.cache_data(ttl=3600)
def get_price(tickers):
    return ftk.get_yahoo_bulk(tickers, period='20Y')

# Pre-process the data
params = st.experimental_get_query_params()

st.title('Peer Group Analysis')
if 'fund' not in params or 'benchmark' not in params or len(params['fund']) < 1 and len(params['benchmark']) != 1:
    st.error('ERROR: No fund or benchmark specified. Please try the following URL.')    
    st.experimental_set_query_params(
        fund=['PRCOX', 'GQEFX', 'STSEX', 'NUESX', 'VTCLX', 'CAPEX', 'USBOX', 'VPMCX', 'JDEAX', 'DFUSX', 'GALLX'],
        benchmark='^SP500TR'
    )
    st.rerun()
    #st.stop()


tickers = params['fund']
tickers.extend(params['benchmark'])

price = get_price(tickers)
periods = price.resample('M').last().to_period()

with st.sidebar:
    horizon = st.select_slider(
        'Sample period',
        options=periods.index,
        value=[periods.index[-60], periods.index[-1]]
    )
    rfr_annualized = st.slider(
        'Risk-free rate (%)', value=2., min_value=0.0, max_value=10., step=0.1
    )

# Process the data
rtn = ftk.price_to_return(price.resample('M').last())[horizon[0] : horizon[1]]
rtn['RF'] = (1 + rfr_annualized / 100) ** (1 / 12) - 1 # M
rtn = rtn.dropna(axis=1)

funds = rtn.iloc[:, :-2]
benchmark = rtn.iloc[:, -2]
rfr = rtn.iloc[:, -1]

dropped = [x for x in tickers if x not in list(rtn.columns)]

# Charts and Tables
if len(dropped) > 0:
    st.warning(f'WARNING: **{len(dropped)}** fund(s) were dropped due to short track record - **{", ".join(dropped)}**')

st.line_chart(ftk.return_to_price(rtn.iloc[:, :-1]))

summary = pd.DataFrame(ftk.summary(funds, benchmark, rfr))
st.write(summary)

st.markdown(open('streamlit/data/signature.md').read())
