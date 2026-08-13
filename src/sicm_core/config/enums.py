from enum import Enum


class ShockType(Enum):
    """Tipos de choques macroeconómicos"""
    SUPPLY = "supply"
    DEMAND = "demand"
    MONETARY = "monetary"
    FISCAL = "fiscal"
