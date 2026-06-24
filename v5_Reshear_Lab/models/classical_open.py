"""
SICM v5 Research Lab — Modelo Clásico Abierto
==============================================
Añade: Tipo de cambio real, Exportaciones netas.
"""
import numpy as np
from typing import Dict
from dataclasses import dataclass
from ..core.parameters import EconomyConfig
from ..core.equilibrium import EquilibriumSolver, EquilibriumResult


@dataclass
class ClassicalOpenData:
    """Datos para graficar modelo clásico abierto"""
    Y_range: np.ndarray
    oferta_agregada: np.ndarray
    demanda_agregada: np.ndarray
    q_range: np.ndarray
    exportaciones: np.ndarray
    importaciones: np.ndarray
    equilibrium: EquilibriumResult


class ClassicalOpenModel:
    """
    Modelo Clásico Abierto.

    Añade sector externo con tipo de cambio real.
    """

    def __init__(self, config: EconomyConfig):
        self.config = config
        self.solver = EquilibriumSolver(config)
        self.result = None

    def solve(self) -> EquilibriumResult:
        """Calcula el equilibrio clásico abierto"""
        self.result = self.solver.solve_classical_open()
        return self.result

    def generate_curves(self, Y_min: float = 0, Y_max: float = 2000,
                       q_min: float = 0.5, q_max: float = 2.0,
                       n_points: int = 100) -> ClassicalOpenData:
        """Genera las curvas para visualización"""
        if self.result is None:
            self.solve()

        p = self.config.produccion
        c = self.config.consumo
        inv = self.config.inversion
        ext = self.config.externo
        G = self.config.G

        # Mercado de bienes (OA-DA)
        Y_range = np.linspace(Y_min, Y_max, n_points)
        oferta_agregada = np.full_like(Y_range, self.result.Y)

        demanda_agregada = np.zeros_like(Y_range)
        for i, Y in enumerate(Y_range):
            Ahorro = Y - c.T - c.consumo(Y)
            r_eq = (inv.I0 - (Ahorro - ext.X0 + ext.m * Y)) / inv.b if inv.b != 0 else 0
            demanda_agregada[i] = c.consumo(Y) + inv.inversion(r_eq) + G + ext.X0 - ext.m * Y

        # Sector externo
        q_range = np.linspace(q_min, q_max, n_points)
        # Exportaciones: X = X0 - v*(q - q_bar) [simplificado]
        exportaciones = ext.X0 * np.ones_like(q_range)  # Simplificado
        importaciones = ext.m * self.result.Y * np.ones_like(q_range)

        return ClassicalOpenData(
            Y_range=Y_range,
            oferta_agregada=oferta_agregada,
            demanda_agregada=demanda_agregada,
            q_range=q_range,
            exportaciones=exportaciones,
            importaciones=importaciones,
            equilibrium=self.result
        )

    def real_exchange_rate_analysis(self, delta_E: float) -> Dict:
        """
        Analiza el efecto de un cambio en el tipo de cambio nominal.

        Args:
            delta_E: Cambio en tipo de cambio nominal

        Returns:
            Dict con análisis del tipo de cambio real
        """
        base = self.solve()

        import copy
        new_config = copy.deepcopy(self.config)
        new_config.externo.E += delta_E

        new_solver = EquilibriumSolver(new_config)
        new_result = new_solver.solve_classical_open()

        q_base = (base.E * base.P) / self.config.externo.P_f
        q_new = (new_result.E * new_result.P) / self.config.externo.P_f

        return {
            "delta_E": delta_E,
            "delta_E_porcentual": ((new_result.E - base.E) / base.E) * 100,
            "q_base": q_base,
            "q_nuevo": q_new,
            "delta_q": q_new - q_base,
            "delta_P": new_result.P - base.P,
            "efecto_P": "Aumenta" if new_result.P > base.P else "Disminuye",
            "explicacion": "En el modelo clásico, una devaluación nominal (E↑) aumenta P proporcionalmente, dejando q sin cambio",
            "base": base.to_dict(),
            "nuevo": new_result.to_dict()
        }
