"""_summary_
"""

class Instrument:
    def __init__(self, name=None):
        self.name = name if name else 'Unnamed'
    
    def __repr__(self):
        return self.name
    
    @abstractmethod
    def 

class Equity(Instrument):
    pass

class Derivative(Instrument):
    def __init__(self, underlying: Instrument, name: str = None):
        self.underlying = underlying
        super().__init__(name)

    def underlying(self):
        return self.underlying

class Option(Derivative):
    def __init__(self, underlying: Instrument, strike: float = 1, name: str = None):
        self.underlying = underlying
        super().__init__(name)

class EuropeanOption(Option):
    pass

class AmericanOption(Option):
    pass

class EuropeanCall(EuropeanOption):
    pass

class EuropeanPut(EuropeanOption):
    pass

class AmericanCall(AmericanOption):
    pass

class AmericanPut(AmericanOption):
    pass