import numpy as np

def itm(otype='c', strike=0, reduced=1, underlying=0, **kwargs):
    if otype == 'c':
        return np.maximum(0, underlying * reduced - strike)
    else:
        return np.maximum(0, strike - underlying * reduced)

def otm(otype='c', strike=0, reduced=1, underlying=0, **kwargs):
    return itm(otype, strike=underlying * reduced, underlying=strike, reduced=1)

def margin(otype='c', quantity=1, expiration=1, strike=0, reduced=1, multiplier=100, otc=False, broad=False, leverage=1, value=0, underlying=0, **kwargs):
    if quantity > 0:
        if otc and expiration > 9/12:
            intrinsic = itm(otype=otype, strike=strike, underlying=underlying, reduced=reduced)
            return quantity * multiplier * (0.75 * intrinsic + value - intrinsic)
        else:
            return quantity * multiplier * value * (0.75 if expiration > 9/12 else 1)
    else:
        pct = 0.15 if broad else 0.2
        out_of_money = otm(otype=otype, strike=strike, underlying=underlying, reduced=reduced)
        return -quantity * multiplier * np.maximum(
            value + pct * leverage * underlying * reduced - out_of_money,
            value + 0.1 * leverage * (underlying * reduced if otype == 'c' else strike))

def max_loss(portfolio):
    strikes = portfolio.strike * portfolio.reduced
    return -min(0, portfolio.apply(lambda row: row.quantity * row.multiplier * itm(**row, underlying=strikes), axis=1).sum().min())

def margin_strategy(portfolio, underlying=0, eligible=False):
    long = portfolio[(portfolio.quantity > 0) & (portfolio.otype != 'u')]
    short = portfolio[(portfolio.quantity < 0) & (portfolio.otype != 'u')]
    under = portfolio[portfolio.otype == 'u']
    s = portfolio.sort_values(['strike', 'otype'])
    
    if eligible\
            and len(s) == 4 and s.expiration.nunique() == 1 and s.strike.nunique() == 2\
            and s.iloc[0].quantity == -s.iloc[1].quantity and s.iloc[0].strike == s.iloc[1].strike and s.iloc[0].otype != s.iloc[1].otype\
            and s.iloc[2].quantity == -s.iloc[3].quantity and s.iloc[2].strike == s.iloc[3].strike and s.iloc[2].otype != s.iloc[3].otype:
        margin_value = 0.5 * (s.strike.iloc[-1] - s.strike.iloc[0]) * s.multiplier[0]
    elif len(under) > 0:
        # Have underlying
        if len(long) == 1 and len(short) == 0:
            # Exchange maintenance margin can be higher than 30%
            margin_value = long.iloc[0].quantity * long.iloc[0].multiplier * np.minimum((0.1 if long.iloc[0].otype == 'p' else 1.1) * long.iloc[0].strike + otm(**long.iloc[0], underlying=underlying), (0.25 if long.iloc[0].otype == 'p' else 1.3) * underlying)        
        elif len(long) == 1 and len(short) == 1:
            if long.iloc[0].strike == short.iloc[0].strike:                
                margin_value = long.iloc[0].quantity * long.iloc[0].multiplier * ((0.1 if under.iloc[0].quantity > 0 else 1.1) * long.iloc[0].strike + (itm(**short.iloc[0], underlying=underlying) if short.iloc[0].otype == 'p' else 0))
            else:
                margin_value = long.iloc[0].quantity * long.iloc[0].multiplier * np.minimum(0.1 * long.iloc[0].strike + otm(**long.iloc[0], underlying=underlying), 0.25 * short.iloc[0].strike)
        else:            
            exposure = (under.quantity * under.value).sum()
            margin_value = exposure * (0.5 if exposure > 0 else -1.5)
    elif short.expiration.min() > long.expiration.max():
        # Short expires after long, recursive
        margin_value = margin_strategy(long) + short.apply(lambda row: margin(**row, underlying=underlying), axis=1).sum()
    elif len(short) == 2 and len(long) == 0 and short.iloc[0].otype != short.iloc[1].otype:
        margins = short.apply(lambda row: margin(**row, underlying=underlying), axis=1)
        smaller = short.iloc[1 - margins.idxmax()]
        margin_value = margins.max() - smaller.quantity * smaller.multiplier * smaller.value
    else:
        # Base Case
        margin_value = max_loss(portfolio) + (long.quantity * long.multiplier * long.value).sum()
    return margin_value * np.ones_like(underlying)