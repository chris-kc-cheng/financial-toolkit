import streamlit as st
import pandas as pd
from io import StringIO

with st.sidebar:
    csv = st.file_uploader("Choose a file")

if csv is not None:
    
    dataframe = pd.read_csv(csv)
    st.write(dataframe)

st.markdown(open('streamlit/data/signature.md').read())