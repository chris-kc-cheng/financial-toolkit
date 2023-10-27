"""_summary_
"""
from abc import ABC, abstractmethod
import toolkit as ftk

class Instrument:
    def __init__(self, name=None):
        self.name = name if name else 'Unnamed'
    
    def __repr__(self):
        return self.name

class Equity(Instrument):
    pass

class Derivative(Instrument):
    def __init__(self, underlying: Instrument, name: str = None):
        self.underlying = underlying
        super().__init__(name)

    def underlying(self):
        return self.underlying

class Option(Derivative):
    def __init__(self, underlying: Instrument, strike: float, name: str = None):
        self.underlying = underlying
        self.strike = strike
        super().__init__(name)
    
    def greeks(self, spot, rate, time, vol, dvd):
        return {'Delta': self.delta(spot, rate, time, vol, dvd),
                'Gamma': self.gamma(spot, rate, time, vol, dvd),
                'Vega' : self.vega (spot, rate, time, vol, dvd),
                'Theta': self.theta(spot, rate, time, vol, dvd),
                'Rho'  : self.rho  (spot, rate, time, vol, dvd)}
    
    @abstractmethod
    def price(self, spot, rate, time, vol, dvd):
        pass

    @abstractmethod
    def delta(self, spot, rate, time, vol, dvd):
        pass

    @abstractmethod
    def gamma(self, spot, rate, time, vol, dvd):
        pass

    @abstractmethod
    def vega(self, spot, rate, time, vol, dvd):
        pass

    @abstractmethod
    def theta(self, spot, rate, time, vol, dvd):
        pass

    @abstractmethod
    def rho(self, spot, rate, time, vol, dvd):
        pass

class EuropeanOption(Option):
    def vega(self, spot, rate, time, vol, dvd):
        return ftk.vega(self.strike, spot, rate, time, vol, dvd)

    def gamma(self, spot, rate, time, vol, dvd):
        return ftk.gamma(self.strike, spot, rate, time, vol, dvd)

class AmericanOption(Option):
    pass

class EuropeanCall(EuropeanOption):

    def price(self, spot, rate, time, vol, dvd):
        return ftk.price_call(self.strike, spot, rate, time, vol, dvd)

    def delta(self, spot, rate, time, vol, dvd):
        return ftk.delta_call(self.strike, spot, rate, time, vol, dvd)

    def theta(self, spot, rate, time, vol, dvd):
        return ftk.theta_call(self.strike, spot, rate, time, vol, dvd)

    def rho(self, spot, rate, time, vol, dvd):
        return ftk.rho_call(self.strike, spot, rate, time, vol, dvd)

class EuropeanPut(EuropeanOption):

    def price(self, spot, rate, time, vol, dvd):
        return ftk.price_put(self.strike, spot, rate, time, vol, dvd)
    
    def delta(self, spot, rate, time, vol, dvd):
        return ftk.delta_put(self.strike, spot, rate, time, vol, dvd)
    
    def theta(self, spot, rate, time, vol, dvd):
        return ftk.theta_put(self.strike, spot, rate, time, vol, dvd)

    def rho(self, spot, rate, time, vol, dvd):
        return ftk.rho_put(self.strike, spot, rate, time, vol, dvd)

class AmericanCall(AmericanOption):
    pass

class AmericanPut(AmericanOption):
    pass