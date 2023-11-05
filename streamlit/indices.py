"""_summary_
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)
import toolkit as ftk


@st.cache_data(ttl=3600)
def get_data():
    data = pd.read_csv('streamlit/data/indices.csv', index_col=0)
    px = ftk.get_yahoo_bulk(data.index, '2y')
    return data, px


# https://flagsapi.com/{x}/flat/64.png as backup (no flag for Europe/ASEAN)
def get_flag(code):
    if isinstance(code, str):
        return f'https://flagpedia.net/data/{"org" if code == "EU" or code == "ASEAN" else "flags"}/w320/{code.lower()}.png'
    else:
        return ''


data, px = get_data()

with st.sidebar:

    date = st.select_slider(
        'Select a date',
        options=px.index,
        value=px.index[-1],
        format_func=lambda d: d.strftime('%Y-%m-%d'))

    currencies = ['Local']
    currencies.extend(sorted(set(data.Currency.dropna())))

    base = st.selectbox(
        'Base currency',
        options=currencies,
        index=0,
    )

# Adjust the prices
def getFX(c):
    return px.loc[:, f'{data.loc[c].Currency}=X']

usd = pd.Series(np.ones(len(px)), index=px.index)
px['USD=X'] = usd
if base != 'Local':

    foreign_fx = px.apply(lambda column: getFX(column.name))
    domestic_fx = px.loc[:, f'{base}=X']
    
    adj_fx = px.filter(regex='=X$', axis=1).apply(lambda x: ftk.convertFX(domestic_fx, x, usd))
    adj_px = px.filter(regex='.*(?<!=X)$', axis=1).apply(lambda x: ftk.convertFX(x, foreign_fx[x.name], domestic_fx))

    adjusted = pd.concat([adj_fx, adj_px], axis=1)

else:
    adjusted = px

# Prepare the table
begin_m = date - pd.offsets.MonthBegin()
begin_q = date - pd.offsets.QuarterBegin(startingMonth=1) # Default is 3
begin_y = date - pd.offsets.YearBegin()

table = pd.DataFrame({
    'MTD': ftk.compound_return(adjusted.pct_change(fill_method=None)[begin_m:date].to_period('D')) * 100,
    'QTD': ftk.compound_return(adjusted.pct_change(fill_method=None)[begin_q:date].to_period('D')) * 100,
    'YTD': ftk.compound_return(adjusted.pct_change(fill_method=None)[begin_y:date].to_period('D')) * 100,
    'Last': adjusted[:date].stack().groupby(level=1).last(),
    'As of': adjusted[:date].aggregate(pd.Series.last_valid_index),
    'chart': pd.Series(adjusted.ffill()[:date].iloc[-20:].T.values.tolist(), index=adjusted.columns)
})
table = pd.concat([data, table], axis=1)
table['Country'] = table['Country'].apply(lambda c: get_flag(c))
table['As of'] = table['As of'].apply(
    lambda d: None if pd.isnull(d) else d.strftime('%b %d'))

groups = ['America', 'Asia', 'EMEA', 'Currency']
horizons = ['MTD', 'QTD', 'YTD']

# Tabs
tabs = st.tabs(groups)

for i, group in enumerate(groups):
    with tabs[i]:
        t = table[table.Group == group]

        htabs = st.tabs(horizons)
        for j, horizon in enumerate(horizons):
            with htabs[j]:
                c = (alt.Chart(t)
                    .mark_bar()
                    .encode(
                        alt.X('Name', sort='-y'),
                        alt.Y(horizon),
                        alt.Color('Name')
                    )
                )
                st.altair_chart(c, use_container_width=True)

        st.dataframe(t,
                    height=1000,
                    hide_index=True,
                    column_order=['Country', 'Name', 'MTD',
                                'QTD', 'YTD', 'Last', 'As of', 'chart'],
                    column_config={
                        'Country': st.column_config.ImageColumn(''),
                        'MTD': st.column_config.NumberColumn(format='%.2f'),
                        'QTD': st.column_config.NumberColumn(format='%.2f'),
                        'YTD': st.column_config.NumberColumn(format='%.2f'),
                        'Last': st.column_config.NumberColumn(format='%.2f'),
                        'chart': st.column_config.LineChartColumn('Last 20 Days'),                        
                    },
                    use_container_width=True)

with st.expander('Data', expanded=False):
    st.write(adjusted)

st.markdown(open('streamlit/data/signature.md').read())
