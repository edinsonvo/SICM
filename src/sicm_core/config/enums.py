from enum import Enum


class ShockType(Enum):
    """Tipos de choques macroeconómicos"""
    SUPPLY = "supply"
    DEMAND = "demand"
    MONETARY = "monetary"
    FISCAL = "fiscal"


class ModelType(Enum):
    """Tipos de modelos macroeconómicos"""
    IS_LM = "is_lm"
    AD_AS = "ad_as"
    MUNDELL_FLEMING = "mundell_fleming"
    SOLOW = "solow"
    DSGE = "dsge"


class EconomyType(Enum):
    """Tipos de economía"""
    CLOSED = "closed"
    OPEN = "open"
