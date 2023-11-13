# Financial Toolkit 💹

This is an open-source Python 🐍 library that implements the most commonly used functions that are frequently used by an investment professional.

This is still in the early stage of development but I will keep adding new features over time. Here are some of the features implemented so far:

- Getting financial data
- Portfolio optimization
- Peer group analysis
- Factor analysis
- Option pricing

### Installation
```sh
pip install fintoolkit
```

### Documentation

The API doc is hosted on [GitHub Pages](https://chris-kc-cheng.github.io/financial-toolkit/toolkit.html).

## Related Repositories

### Jupyter Notebooks 📔

Jupyter is browser-based shell that allows user to interact with Python scripts.

In this repository, I have developed some Jupyter notebooks showing the most common use cases in investment management that utilizes the Financial Toolkit.

The source code is hosted on GitHub at: https://github.com/chris-kc-cheng/ftk-notebook

No Python? No problem. With Binder, you may interact with the Jupyter notebooks in an executable environment even without Python installed in your computer.

Binder: https://mybinder.org/v2/gh/chris-kc-cheng/ftk-notebook/HEAD

## Streamlit Apps 👑

Streamlit is a framework that turns Python scripts into interactive web apps. 

In this repository, I have developed some web apps showing the most common use cases in investment management that utilizes the Financial Toolkit.

| App | Description |
|-----|-------------|
| [Index Montior](https://ftk-indices.streamlit.app/) | Tables and charts of asset class returns measured in different currency terms and time horizons.
| [Factor Analysis](https://ftk-factors.streamlit.app/) | Analyzing factor loading of a portfolio using Fama-French model. |
| [Portfolio Optimization](https://ftk-portfolio-optimization.streamlit.app/) | Comparing risk reward and risk contribution of various weighting schemes. |
| [Peer Group Analysis](https://ftk-peers.streamlit.app/) | Comparing fund performance and risk measures against benchmark and peer group. |
| [Option Pricing & Greeks](https://ftk-options.streamlit.app/) | Visualizing payoff and Greeks of various option strategies. |

The source code is hosted on GitHub at: https://github.com/chris-kc-cheng/ftk-streamlit.