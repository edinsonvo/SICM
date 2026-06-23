from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict


class EconomyType(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


class ExchangeRateRegime(str, Enum):
    FIXED = "fixed"
    FLEXIBLE = "flexible"


class CapitalMobility(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PERFECT = "perfect"


@dataclass
class EconomyConfig:

    economy_type: EconomyType = EconomyType.CLOSED

    exchange_rate_regime: ExchangeRateRegime = (
        ExchangeRateRegime.FLEXIBLE
    )

    capital_mobility: CapitalMobility = (
        CapitalMobility.MEDIUM
    )

    # Consumo
    C0: float = 50.0
    c: float = 0.80

    # Inversión
    I0: float = 100.0
    b: float = 5.0

    # Mercado monetario
    k: float = 0.50
    h: float = 10.0

    # Gobierno
    G: float = 150.0
    T: float = 120.0

    # Dinero
    M: float = 500.0
    P: float = 1.0

    # Producción
    A: float = 1.0
    K: float = 100.0
    alpha: float = 0.33

    # Sector externo
    NX0: float = 0.0

    def validate(self):

        if not (0 < self.c < 1):
            raise ValueError(
                "La propensión marginal a consumir debe estar entre 0 y 1"
            )

        if self.P <= 0:
            raise ValueError(
                "El nivel de precios debe ser positivo"
            )

        if self.K <= 0:
            raise ValueError(
                "El capital debe ser positivo"
            )

        if not (0 < self.alpha < 1):
            raise ValueError(
                "alpha debe estar entre 0 y 1"
            )

    def to_dict(self) -> Dict:
        return asdict(self)
