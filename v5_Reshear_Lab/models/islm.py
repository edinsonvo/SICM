"""
SICM v5 Research Lab — Modelo IS-LM
====================================
Economía cerrada keynesiana. El más utilizado en docencia.
"""
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
from ..core.parameters import EconomyConfig
from ..core.equilibrium import EquilibriumSolver, EquilibriumResult


@dataclass
class ISLMData:
    """Datos para graficar IS-LM"""
    Y_range: np.ndarray
    IS_curve: np.ndarray
    LM_curve: np.ndarray
    equilibrium: EquilibriumResult

    def get_intersections(self) -> Tuple[float, float]:
        """Devuelve (Y_eq, r_eq)"""
        return self.equilibrium.Y, self.equilibrium.r


class ISLMModel:
    """
    Modelo IS-LM para economía cerrada.

    IS: Y = C(Y-T) + I(r) + G
    LM: M/P = L(Y, r)
    """

    def __init__(self, config: EconomyConfig):
        self.config = config
        self.solver = EquilibriumSolver(config)
        self.result = None

    def solve(self) -> EquilibriumResult:
        """Calcula el equilibrio IS-LM"""
        self.result = self.solver.solve_islm_cerrado()
        return self.result

    def generate_curves(self, Y_min: float = 0, Y_max: float = 2000, n_points: int = 100) -> ISLMData:
        """
        Genera las curvas IS y LM para visualización.

        Returns:
            ISLMData con rangos de Y y valores de r para cada curva
        """
        if self.result is None:
            self.solve()

        c = self.config.consumo
        inv = self.config.inversion
        d = self.config.dinero
        G = self.config.G

        Y_range = np.linspace(Y_min, Y_max, n_points)

        # Curva IS: r = (c0 + I0 + G - c1*T - (1-c1)*Y) / b
        # Despejando r desde la ecuación IS
        IS_curve = np.zeros_like(Y_range)
        for i, Y in enumerate(Y_range):
            numer = c.c0 + inv.I0 + G - c.c1 * c.T - (1 - c.c1) * Y
            IS_curve[i] = numer / inv.b if inv.b != 0 else 0

        # Curva LM: r = (k*Y - M/P) / h
        LM_curve = np.zeros_like(Y_range)
        for i, Y in enumerate(Y_range):
            LM_curve[i] = (d.k * Y - d.M / d.P) / d.h if d.h != 0 else 0

        return ISLMData(
            Y_range=Y_range,
            IS_curve=IS_curve,
            LM_curve=LM_curve,
            equilibrium=self.result
        )

    def comparative_statics(self, param: str, delta: float) -> Dict:
        """
        Análisis de estática comparada.

        Args:
            param: Parámetro a variar ("G", "T", "M", "c0", "I0", "b")
            delta: Cambio en el parámetro

        Returns:
            Diccionario con cambios en Y y r
        """
        base = self.solve()

        # Crear configuración modificada
        import copy
        new_config = copy.deepcopy(self.config)

        if param == "G":
            new_config.G += delta
        elif param == "T":
            new_config.consumo.T += delta
        elif param == "M":
            new_config.dinero.M += delta
        elif param == "c0":
            new_config.consumo.c0 += delta
        elif param == "I0":
            new_config.inversion.I0 += delta
        elif param == "b":
            new_config.inversion.b += delta
        elif param == "k":
            new_config.dinero.k += delta
        elif param == "h":
            new_config.dinero.h += delta
        else:
            raise ValueError(f"Parámetro no soportado: {param}")

        new_solver = EquilibriumSolver(new_config)
        new_result = new_solver.solve_islm_cerrado()

        return {
            "delta_Y": new_result.Y - base.Y,
            "delta_r": new_result.r - base.r,
            "delta_C": new_result.C - base.C,
            "delta_I": new_result.I - base.I,
            "multiplicador": (new_result.Y - base.Y) / delta if delta != 0 else 0,
            "base": base.to_dict(),
            "nuevo": new_result.to_dict()
        }

    def policy_mix(self, delta_G: float, delta_M: float) -> Dict:
        """
        Analiza una combinación de políticas fiscal y monetaria.

        Args:
            delta_G: Cambio en gasto público
            delta_M: Cambio en oferta monetaria

        Returns:
            Resultado de la política mixta
        """
        base = self.solve()

        import copy
        new_config = copy.deepcopy(self.config)
        new_config.G += delta_G
        new_config.dinero.M += delta_M

        new_solver = EquilibriumSolver(new_config)
        new_result = new_solver.solve_islm_cerrado()

        return {
            "delta_G": delta_G,
            "delta_M": delta_M,
            "delta_Y": new_result.Y - base.Y,
            "delta_r": new_result.r - base.r,
            "efecto_Y": "expansivo" if new_result.Y > base.Y else "contraccionista",
            "efecto_r": "sube" if new_result.r > base.r else "baja",
            "base": base.to_dict(),
            "nuevo": new_result.to_dict()
        }
