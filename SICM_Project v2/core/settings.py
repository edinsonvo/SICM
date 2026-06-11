from dataclasses import dataclass
from enum import Enum

class EconomyType(Enum):
    CLOSED = "Cerrada"
    OPEN = "Abierta"

class ExchangeRate(Enum):
    FIXED = "Fijo"
    FLEXIBLE = "Flexible"

class CapitalMobility(Enum):
    NONE = "Nula"
    PARTIAL = "Parcial"
    PERFECT = "Perfecta"

class Horizon(Enum):
    STATIC = "Estático"
    DYNAMIC = "Dinámico"

@dataclass
class Settings:
    economy: EconomyType = EconomyType.CLOSED
    exchange_rate: ExchangeRate = ExchangeRate.FIXED
    capital_mobility: CapitalMobility = CapitalMobility.PERFECT
    horizon: Horizon = Horizon.STATIC
    
    def validate(self):
        """Validar consistencia de la configuración"""
        if self.economy == EconomyType.CLOSED:
            self.exchange_rate = ExchangeRate.FIXED
        return self
    
    def to_dict(self):
        return {
            "economy": self.economy.value,
            "exchange_rate": self.exchange_rate.value,
            "capital_mobility": self.capital_mobility.value,
            "horizon": self.horizon.value
        }
