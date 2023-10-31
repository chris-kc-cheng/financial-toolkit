import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)

import streamlit as st
import pandas as pd
import toolkit as ftk

@st.cache_data
def get_datasets():
    return ftk.get_famafrench_datasets()

@st.cache_data(ttl=3600)
def get_factors(dataset, mom):
    return ftk.get_famafrench_factors(dataset, mom)

@st.cache_data(ttl=60)
def get_price(ticker):
    return ftk.price_to_return(ftk.get_yahoo(ticker))

with st.sidebar:
    dataset = st.selectbox(
        'Select a factor',
        options = get_datasets(),
        format_func = lambda x: x.replace('_', ' '),
        index = 23)    
    
    mom = st.toggle('Add momentum factor')

portfolio = get_price('ARKK')
factors = get_factors(dataset, mom)

if ftk.periodicity(portfolio) > ftk.periodicity(factors):
    print(ftk.periodicity(portfolio), ftk.periodicity(factors))
    portfolio = portfolio.resample(factors.index.freqstr).aggregate(ftk.compound_return)

merged = pd.merge(portfolio, factors, left_index=True, right_index=True)
portfolio = merged.iloc[:, 0]
factors = merged.iloc[:, 1:]
betas = ftk.beta(portfolio, factors)

attribution = betas * factors
explained = attribution.sum(axis=1)
combined = pd.concat([portfolio, explained], axis=1)
combined.columns = ['Portfolio', 'Factors']

k = (attribution.T * ftk.carino(portfolio, 0)).T / ftk.carino(ftk.compound_return(portfolio), 0)
contribution = k.sum().sort_values(ascending=False)

table = pd.concat([betas, contribution], axis=1)
table.columns = ['Beta', 'Contribution']
table = table.rename(index={'Mkt-RF': 'Market returns above risk-free rate',
'HML': 'High minus low (HML)',
'RF' : 'Risk-Free Rate',
'CMA': 'Conservative minus aggressive (CMA)',
'WML': 'Winners minus losers (WML)',
'SMB': 'Small minus big (SMB)',
'RMW': 'Robust minus weak (RMW)'})

# Total return
st.write(ftk.rsquared(portfolio, factors))
# Unexplained


st.write(table.sort_values('Beta', ascending=False))
st.line_chart(ftk.return_to_price(combined))
