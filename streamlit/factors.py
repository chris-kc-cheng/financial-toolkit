import pandas as pd
import streamlit as st
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.append(parent)
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


# Return (portfolio, factors, rfr)
def resample(portfolio, factors):
    if ftk.periodicity(portfolio) > ftk.periodicity(factors):
        portfolio = portfolio.resample(
            factors.index.freqstr).aggregate(ftk.compound_return)

    merged = pd.merge(portfolio, factors, left_index=True, right_index=True)
    return merged.iloc[:, 0], merged.iloc[:, 1:-1], merged.iloc[:, -1]

@st.cache_data(ttl=60)
def get_bestfit(portfolio):

    def analyse(portfolio, model):        
        portfolio, factors, rfr = resample(portfolio, get_factors(model, mom))
        return ftk.rsquared(portfolio - rfr, factors, adjusted=True)

    models = get_datasets()
    return pd.Series([analyse(portfolio, model) for model in models], index=models).sort_values(ascending=False)

if 'price' not in st.session_state:
    st.session_state.price = None

with st.sidebar:

    with st.form("my_form"):
        ticker = st.text_input('Ticker', 'ARKK')
        submitted = st.form_submit_button("Search")
        if submitted:
            try:
                st.session_state.price = get_price(ticker)
            except:
                st.error('Invalid ticker')

    dataset = st.selectbox(
        'Select a factor',
        options=get_datasets(),
        format_func=lambda x: x.replace('_', ' '),
        index=23)

    mom = st.toggle('Add momentum factor')

portfolio = st.session_state.price

st.title('Fama–French Factor Model')

if portfolio is not None:
    st.header(portfolio.name)

    if st.button('Check model of best fit'):
        best = get_bestfit(portfolio)
        st.info(f'The model of best fit is {best.index[0]} with adjusted R-squared of {best.iloc[0]:.2%}')

    factors = get_factors(dataset, mom)

    portfolio, factors, rfr = resample(portfolio, factors)
    betas = ftk.beta(portfolio - rfr, factors)

    attribution = pd.concat([betas * factors, rfr], axis=1)
    explained = attribution.sum(axis=1)
    combined = pd.concat([portfolio, explained], axis=1)
    combined.columns = ['Portfolio', 'Factors']

    total_return = ftk.compound_return(portfolio)
    k = (attribution.T * ftk.carino(portfolio, 0)).T / \
        ftk.carino(total_return, 0)
    contribution = k.sum().sort_values(ascending=False)

    summary = pd.DataFrame({'Beta': {'Unexplained': None, 'Total': None},
                            'Contribution': {'Unexplained': total_return - contribution.sum(), 'Total': total_return}})

    table = pd.concat([betas, contribution], axis=1)
    table.columns = ['Beta', 'Contribution']
    table = pd.concat([table, summary])
    table['Contribution'] = table['Contribution'] * 100

    table = table.rename(index={'Mkt-RF': 'Market returns above risk-free rate (Mkt-RF)',
                                'HML': 'High minus low (HML)',
                                'RF': 'Risk-Free Rate (RF)',
                                'CMA': 'Conservative minus aggressive (CMA)',
                                'WML': 'Winners minus losers (WML)',
                                'SMB': 'Small minus big (SMB)',
                                'RMW': 'Robust minus weak (RMW)'})
    table = table.sort_values('Beta', ascending=False)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Portfolio Ann. Return',
                f'{ftk.compound_return(portfolio, annualize=True):.2%}')
    col2.metric('Factor Ann. Return',
                f'{ftk.compound_return(explained, annualize=True):.2%}')
    col3.metric('R-Squared', f'{ftk.rsquared(portfolio - rfr, factors):.2%}')
    col4.metric('Adj. R-Squared',
                f'{ftk.rsquared(portfolio - rfr, factors, adjusted=True):.2%}')

    st.dataframe(table,
                 column_config={
                     "Beta": st.column_config.NumberColumn(
                         "Beta",
                         format='%.2f'
                     ),
                     "Contribution": st.column_config.NumberColumn(
                         "Contribution (%)",
                         format='%.1f'
                     ),
                 },)
    st.line_chart(ftk.return_to_price(combined))
else:
    st.write(
        'Please search a ticker on the sidebar e.g. `SPY`, `QQQ`, `ARKK`, `BRK-B`')

st.markdown(open('streamlit/data/signature.md').read())
