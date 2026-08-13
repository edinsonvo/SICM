from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Equilibrium:

    output: float

    interest_rate: float

    inflation: float

    unemployment: float

    employment: float

    exchange_rate: float
