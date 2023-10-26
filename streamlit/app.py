"""_summary_
"""
import os
import sys
import streamlit as st

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)
import toolkit as ftk

st.title('🌞')
st.slider('Slider')
st.write(ftk.hello(234))