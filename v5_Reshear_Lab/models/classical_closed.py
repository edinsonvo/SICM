"""
SICM v5 Research Lab — Modelo Clásico Cerrado
===============================================
Mercados: Trabajo, Producción, Fondos prestables.
Precios flexibles, pleno empleo.
"""
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from ..core.parameters import EconomyConfig
from ..core.equilibrium import EquilibriumSolver, EquilibriumResult


@dataclass
class ClassicalClosedData:
    """Datos para graficar modelo clásico cerrado"""
    L_range: np.ndarray
    demanda_trabajo: np.ndarray
    oferta_trabajo: np.ndarray
    Y_range: np.ndarray
    oferta_agregada: np.ndarray
    demanda_agregada: np.ndarray
    r_range: np.ndarray
    ahorro: np.ndarray
    inversion: np.ndarray
    equilibrium: EquilibriumResult


class ClassicalClosedModel:
    """
    Modelo Clásico Cerrado.

    Mercado de trabajo: Ns = Nd (pleno empleo)
    Producción: Y = F(K, L)
    Mercado de fondos: S(r) = I(r)
    Dinero: M = P*L(Y, r) (neutro)
    """

    def __init__(self, config: EconomyConfig):
        self.config = config
        self.solver = EquilibriumSolver(config)
        self.result = None

    def solve(self) -> EquilibriumResult:
        """Calcula el equilibrio clásico cerrado"""
        self.result = self.solver.solve_classical_closed()
        return self.result

    def generate_curves(self, L_min: float = 0, L_max: float = 200, 
                       Y_min: float = 0, Y_max: float = 2000,
                       r_min: float = 0, r_max: float = 0.2,
                       n_points: int = 100) -> ClassicalClosedData:
        """Genera las curvas de los 3 mercados para visualización"""
        if self.result is None:
            self.solve()

        p = self.config.produccion
        c = self.config.consumo
        inv = self.config.inversion
        d = self.config.dinero
        G = self.config.G

        # Mercado de trabajo
        L_range = np.linspace(L_min, L_max, n_points)
        # Demanda de trabajo: PML = W/P => W/P = (1-α)*A*(K/L)^α
        demanda_trabajo = p.productividad_marginal_trabajo(L_range)
        # Oferta de trabajo: vertical en L_bar (clásico)
        oferta_trabajo = np.full_like(L_range, p.L_bar)

        # Mercado de bienes (OA-DA)
        Y_range = np.linspace(Y_min, Y_max, n_points)
        # OA: vertical en Y* (pleno empleo)
        oferta_agregada = np.full_like(Y_range, self.result.Y)
        # DA: Y = C + I + G (con r endógeno)
        demanda_agregada = np.zeros_like(Y_range)
        for i, Y in enumerate(Y_range):
            # r que equilibra S=I para ese Y
            Ahorro = Y - c.T - c.consumo(Y)
            r_eq = (inv.I0 - Ahorro) / inv.b if inv.b != 0 else 0
            demanda_agregada[i] = c.consumo(Y) + inv.inversion(r_eq) + G

        # Mercado de fondos prestables
        r_range = np.linspace(r_min, r_max, n_points)
        ahorro = np.zeros_like(r_range)
        inversion = np.zeros_like(r_range)
        for i, r in enumerate(r_range):
            # Para cada r, calculamos Y de pleno empleo y luego S
            Y_fe = p.produccion(p.L_bar)
            ahorro[i] = Y_fe - c.T - c.consumo(Y_fe)
            inversion[i] = inv.inversion(r)

        return ClassicalClosedData(
            L_range=L_range,
            demanda_trabajo=demanda_trabajo,
            oferta_trabajo=oferta_trabajo,
            Y_range=Y_range,
            oferta_agregada=oferta_agregada,
            demanda_agregada=demanda_agregada,
            r_range=r_range,
            ahorro=ahorro,
            inversion=inversion,
            equilibrium=self.result
        )

    def neutrality_of_money(self, delta_M: float) -> Dict:
        """
        Demuestra la neutralidad del dinero.

        Args:
            delta_M: Cambio en oferta monetaria

        Returns:
            Dict mostrando que Y y r no cambian, solo P
        """
        base = self.solve()

        import copy
        new_config = copy.deepcopy(self.config)
        new_config.dinero.M += delta_M

        new_solver = EquilibriumSolver(new_config)
        new_result = new_solver.solve_classical_closed()

        return {
            "delta_M": delta_M,
            "delta_Y": new_result.Y - base.Y,
            "delta_r": new_result.r - base.r,
            "delta_P": new_result.P - base.P,
            "delta_P_porcentual": ((new_result.P - base.P) / base.P) * 100,
            "neutralidad": "Mantiene" if abs(new_result.Y - base.Y) < 0.01 else "No mantiene",
            "explicacion": "En el modelo clásico, el dinero es neutral: solo afecta precios, no variables reales",
            "base": base.to_dict(),
            "nuevo": new_result.to_dict()
        }
