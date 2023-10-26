"""_summary_
"""

class Instrument:
    def __init__(self, name):
        print('Instrument init')
        self.name = name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

class Equity(Instrument):
    pass

class Derivative(Instrument):
    def __init__(self, name):
        super().__init__(name)

class Option(Derivative):
    pass

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