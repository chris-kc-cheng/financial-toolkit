class Instrument:
    pass

class Equity(Instrument):
    pass

class Derivative(Instrument):
    pass

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

class AmericanCall(AmericanOption):
    pass