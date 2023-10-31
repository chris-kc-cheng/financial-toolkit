import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)

import re
import streamlit as st
import pandas as pd
import pandas_datareader as pdr
import toolkit as ftk

@st.cache_data
def get_datasets():
    pattern =  re.compile(r'^\w+_Factors\w*$')
    full = pdr.famafrench.get_available_datasets()
    return sorted([x for x in full if pattern.match(x)])

@st.cache_data(ttl=3600)
def get_factors(dataset):
    basic = pdr.get_data_famafrench(dataset)[0]
    momentum = pdr.get_data_famafrench(re.sub(r'[35]_Factors', 'Mom_Factor', dataset))[0]
    return basic, momentum

with st.sidebar:
    dataset = st.selectbox(
        'Select a factor',
        options = get_datasets(),
        format_func = lambda x: x.replace('_', ' '),
        index = 23)
    
    mom = st.toggle('Add momentum factor')

portfolio = ftk.price_to_return(ftk.get_yahoo('ARKK'))
factors = ftk.get_famafrench_factors(dataset, mom)

if ftk.periodicity(portfolio) > ftk.periodicity(factors):
    portfolio = portfolio.resample(factors.index.freqstr).aggregate(ftk.compound_return)

merged = pd.merge(portfolio, factors, left_index=True, right_index=True)
portfolio = merged.iloc[:, 0]
factors = merged.iloc[:, 1:]
betas = ftk.beta(portfolio, factors) #.sort_values() doesn't matter

explained = (betas * factors).sum(axis=1)
combined = pd.concat([ftk.return_to_price(portfolio), ftk.return_to_price(explained)], axis=1)
combined.columns = ['Portfolio', 'Explained by Betas']

st.write(betas)
st.line_chart(data=combined)

st.write(ftk.rsquared(portfolio, factors))

st.json(combined.to_json(), expanded=False)
print(combined)