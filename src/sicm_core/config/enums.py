from enum import Enum

class ModelType(str, Enum):
    ISLM = "islm"
    CLASSICAL = "classical"
    MUNDELL_FLEMING = "mundell_fleming"
    KEYNESIAN = "keynesian"

class ShockType(str, Enum):
    FISCAL = "fiscal"
    MONETARY = "monetary"
    SUPPLY = "supply"
    EXTERNAL = "external"

class EconomyType(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
