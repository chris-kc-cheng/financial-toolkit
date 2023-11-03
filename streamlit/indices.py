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

@st.cache_data(ttl=3600)
def get_data():
    indices = json.load(open('streamlit/data/indices.json'))


# https://flagsapi.com/{x}/flat/64.png as backup (no flag for Europe/ASEAN)
def get_flag(code):
    return f'https://flagpedia.net/data/{"org" if len(code) > 2 else "flags"}/w320/{code.lower()}.png'

with st.sidebar:
    
    option = st.selectbox(
        'Select base currency',
        ['CAD', 'USD', 'EUR'])
    
    countries = ['US', 'CA', 'EU', 'JP', 'CN', 'KR', 'ASEAN']
    
    df = pd.DataFrame(map(lambda x: {'c': get_flag(x)}, countries))

st.dataframe(df,
                 column_config={
                    "c": st.column_config.ImageColumn('')
                 })
    
st.markdown(open('streamlit/data/signature.md').read())