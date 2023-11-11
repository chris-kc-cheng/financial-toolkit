import calendar
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

def get_rolling(df, annualize):    
    n = len(df) // 12
    df2 = df.T.copy()
    for i in range(1, n + 1):
        df2[f'{i}Y'] = df.T.iloc[:, -i * 12:].apply(lambda x: ftk.compound_return(x, annualize), axis=1)
    return (df2.iloc[:, -n:]).style.format('{0:.2%}').highlight_max(color='lightgreen')

def get_table(df, period):
    return (df.resample(period).aggregate(ftk.compound_return).T).style.format('{0:,.2%}').highlight_max(color='lightgreen')

def format_table(s):
    tbl = s.groupby([(s.index.year), (s.index.month)]).sum()
    tbl = tbl.unstack(level=1).sort_index(ascending=False)
    tbl.columns = [calendar.month_abbr[m] for m in range(1, 13)]
    tbl['YTD'] = tbl.agg(ftk.compound_return, axis=1)
    return tbl.style.format('{0:.2%}')    

# Pre-process the data
params = st.experimental_get_query_params()
url = "/?fund=PRCOX&fund=GQEFX&fund=STSEX&fund=NUESX&fund=VTCLX&fund=CAPEX&fund=USBOX&fund=VPMCX&fund=JDEAX&fund=DFUSX&fund=GALLX&benchmark=%5ESP500TR"

st.title('Peer Group Analysis')
if 'fund' not in params or 'benchmark' not in params or len(params['fund']) < 1 and len(params['benchmark']) != 1:
    st.error('ERROR: No fund or benchmark specified. Please try the following URL.')    
    st.markdown(f"""
    Click the following link and replace the tickers with your funds and benchmark.
                
    > <a href="{url}" target="_self">{url}</a>
    """, unsafe_allow_html=True)
    st.stop()


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
fund_n_bm = rtn.iloc[:, :-1]

dropped = [x for x in tickers if x not in list(rtn.columns)]

summary = pd.DataFrame(ftk.summary(funds, benchmark, rfr)).iloc[:, 2:]

# Charts and Tables
if len(dropped) > 0:
    st.warning(f'WARNING: **{len(dropped)}** fund(s) were dropped due to short track record - **{", ".join(dropped)}**')

st.line_chart(ftk.return_to_price(fund_n_bm))

col1, col2 = st.columns(2)
col1.scatter_chart(summary.reset_index(), x='Annualized Volatility', y='Annualized Return', color='index')
col2.scatter_chart(summary.reset_index(), x='Annualized Tracking Error', y='Annualized Active Return', color='index')

st.header('Performance')
category_tabs = st.tabs(['Rolling Period', 'By Year', 'By Quarter', 'By Month', 'By Fund'])

with category_tabs[0]:
    annualize = st.toggle('Annualize', value=True)
    st.dataframe(get_rolling(fund_n_bm, annualize), use_container_width=False)

with category_tabs[1]:
    st.dataframe(get_table(fund_n_bm, 'Y'))

with category_tabs[2]:
    st.dataframe(get_table(fund_n_bm, 'Q'))

with category_tabs[3]:
    st.dataframe(get_table(fund_n_bm, 'M'))

with category_tabs[4]:
    tabs = st.tabs(list(fund_n_bm.columns))
    for i, tab in enumerate(tabs):
        tab.write(format_table(fund_n_bm.iloc[:, i]))

st.header('Risk')
st.write(summary.style.highlight_max(color='lightgreen'))

st.markdown(open('streamlit/data/signature.md').read())
