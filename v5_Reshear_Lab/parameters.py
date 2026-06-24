"""
SICM v5 Research Lab — Motor de Parámetros
============================================
EconomyConfig: Configuración central de la economía simulada.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class TipoEconomia(Enum):
    CERRADA = "cerrada"
    ABIERTA = "abierta"


class RegimenCambiario(Enum):
    FIJO = "fijo"
    FLEXIBLE = "flexible"


class MovilidadCapital(Enum):
    PERFECTA = "perfecta"
    IMPERFECTA = "imperfecta"
    NULA = "nula"


@dataclass
class ParametrosConsumo:
    """Parámetros de la función de consumo: C = c0 + c1*(Y-T)"""
    c0: float = 100.0          # Consumo autónomo
    c1: float = 0.75           # Propensión marginal a consumir
    T: float = 100.0           # Impuestos (lump-sum)

    def consumo(self, Y: float) -> float:
        return self.c0 + self.c1 * max(0, Y - self.T)


@dataclass
class ParametrosInversion:
    """Parámetros de la función de inversión: I = I0 - b*r"""
    I0: float = 150.0          # Inversión autónoma
    b: float = 20.0            # Sensibilidad al tipo de interés

    def inversion(self, r: float) -> float:
        return max(0, self.I0 - self.b * r)


@dataclass
class ParametrosDinero:
    """Parámetros del mercado de dinero: M/P = k*Y - h*r"""
    M: float = 500.0           # Oferta monetaria nominal
    P: float = 1.0             # Nivel de precios
    k: float = 0.5             # Sensibilidad demanda dinero a ingreso
    h: float = 10.0            # Sensibilidad demanda dinero a interés

    def demanda_dinero(self, Y: float, r: float) -> float:
        return self.k * Y - self.h * r

    def oferta_dinero_real(self) -> float:
        return self.M / self.P


@dataclass
class ParametrosProduccion:
    """Parámetros de producción: Y = A * K^α * L^(1-α)"""
    A: float = 1.0             # Productividad total de factores
    K: float = 100.0           # Stock de capital
    alpha: float = 0.33        # Elasticidad del capital
    L_bar: float = 100.0       # Oferta de trabajo

    def produccion(self, L: float) -> float:
        return self.A * (self.K ** self.alpha) * (L ** (1 - self.alpha))

    def productividad_marginal_trabajo(self, L: float) -> float:
        return (1 - self.alpha) * self.A * (self.K ** self.alpha) * (L ** (-self.alpha))


@dataclass
class ParametrosSectorExterno:
    """Parámetros del sector externo"""
    X0: float = 50.0           # Exportaciones autónomas
    m: float = 0.2             # Propensión marginal a importar
    E: float = 1.0             # Tipo de cambio nominal
    P_f: float = 1.0           # Nivel de precios extranjero
    r_f: float = 0.03          # Tipo de interés extranjero

    def exportaciones(self, Y: float) -> float:
        return self.X0

    def importaciones(self, Y: float) -> float:
        return self.m * Y

    def tipo_cambio_real(self, P_dom: float) -> float:
        return (self.E * P_dom) / self.P_f


@dataclass
class EconomyConfig:
    """
    Configuración central de la economía.
    Todo el sistema depende de esta clase.
    """
    tipo_economia: TipoEconomia = TipoEconomia.CERRADA
    regimen_cambiario: RegimenCambiario = RegimenCambiario.FLEXIBLE
    movilidad_capital: MovilidadCapital = MovilidadCapital.PERFECTA

    # Sub-configuraciones
    consumo: ParametrosConsumo = field(default_factory=ParametrosConsumo)
    inversion: ParametrosInversion = field(default_factory=ParametrosInversion)
    dinero: ParametrosDinero = field(default_factory=ParametrosDinero)
    produccion: ParametrosProduccion = field(default_factory=ParametrosProduccion)
    externo: ParametrosSectorExterno = field(default_factory=ParametrosSectorExterno)

    # Gasto público
    G: float = 200.0

    # Parámetros adicionales
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tipo_economia": self.tipo_economia.value,
            "regimen_cambiario": self.regimen_cambiario.value,
            "movilidad_capital": self.movilidad_capital.value,
            "G": self.G,
            "consumo": {
                "c0": self.consumo.c0,
                "c1": self.consumo.c1,
                "T": self.consumo.T
            },
            "inversion": {
                "I0": self.inversion.I0,
                "b": self.inversion.b
            },
            "dinero": {
                "M": self.dinero.M,
                "P": self.dinero.P,
                "k": self.dinero.k,
                "h": self.dinero.h
            },
            "produccion": {
                "A": self.produccion.A,
                "K": self.produccion.K,
                "alpha": self.produccion.alpha,
                "L_bar": self.produccion.L_bar
            },
            "externo": {
                "X0": self.externo.X0,
                "m": self.externo.m,
                "E": self.externo.E,
                "P_f": self.externo.P_f,
                "r_f": self.externo.r_f
            }
        }
