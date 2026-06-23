from dataclasses import dataclass


@dataclass
class EquilibriumResult:

    Y: float

    r: float

    inflation: float

    employment: float

    unemployment: float

    exchange_rate: float

    nx: float

    model: str

    notes: str = ""
