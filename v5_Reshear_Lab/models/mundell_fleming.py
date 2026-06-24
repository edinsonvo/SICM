"""
SICM v5 Research Lab — Modelo Mundell-Fleming
==============================================
Economía abierta con tipo de cambio fijo o flexible.
"""
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from ..core.parameters import EconomyConfig, RegimenCambiario, MovilidadCapital
from ..core.equilibrium import EquilibriumSolver, EquilibriumResult


@dataclass
class MundellFlemingData:
    """Datos para graficar Mundell-Fleming"""
    Y_range: np.ndarray
    IS_curve: np.ndarray
    LM_curve: np.ndarray
    BP_curve: np.ndarray
    equilibrium: EquilibriumResult

    def get_intersections(self) -> Tuple[float, float, float]:
        """Devuelve (Y_eq, r_eq, E_eq)"""
        return self.equilibrium.Y, self.equilibrium.r, self.equilibrium.E


class MundellFlemingModel:
    """
    Modelo Mundell-Fleming para economía abierta.

    IS*: Y = C + I + G + NX
    LM: M/P = L(Y, r)
    BP: r = r_f (con movilidad perfecta) o NX + KA = 0
    """

    def __init__(self, config: EconomyConfig):
        self.config = config
        self.solver = EquilibriumSolver(config)
        self.result = None

    def solve(self) -> EquilibriumResult:
        """Calcula el equilibrio Mundell-Fleming"""
        self.result = self.solver.solve_mundell_fleming()
        return self.result

    def generate_curves(self, Y_min: float = 0, Y_max: float = 2000, n_points: int = 100) -> MundellFlemingData:
        """Genera las curvas IS*, LM y BP para visualización"""
        if self.result is None:
            self.solve()

        c = self.config.consumo
        inv = self.config.inversion
        d = self.config.dinero
        ext = self.config.externo
        G = self.config.G

        Y_range = np.linspace(Y_min, Y_max, n_points)

        # IS*: r = (c0 + I0 + G - c1*T + X0 - (1-c1+m)*Y) / b
        IS_curve = np.zeros_like(Y_range)
        for i, Y in enumerate(Y_range):
            numer = c.c0 + inv.I0 + G - c.c1 * c.T + ext.X0 - (1 - c.c1 + ext.m) * Y
            IS_curve[i] = numer / inv.b if inv.b != 0 else 0

        # LM: r = (k*Y - M/P) / h
        LM_curve = np.zeros_like(Y_range)
        for i, Y in enumerate(Y_range):
            LM_curve[i] = (d.k * Y - d.M / d.P) / d.h if d.h != 0 else 0

        # BP: depende del régimen y movilidad
        BP_curve = np.zeros_like(Y_range)
        if self.config.movilidad_capital == MovilidadCapital.PERFECTA:
            # BP es horizontal en r = r_f
            BP_curve[:] = ext.r_f
        elif self.config.movilidad_capital == MovilidadCapital.NULA:
            # BP coincide con NX=0: Y = X0/m
            Y_nx0 = ext.X0 / ext.m if ext.m != 0 else Y_max
            BP_curve[:] = np.nan  # Vertical
        else:
            # BP con pendiente positiva
            for i, Y in enumerate(Y_range):
                # NX = X0 - m*Y, KA = z*(r - r_f)
                # BP: NX + KA = 0 => r = r_f - NX/z
                z = 50  # Sensibilidad de flujos de capital
                BP_curve[i] = ext.r_f - (ext.X0 - ext.m * Y) / z

        return MundellFlemingData(
            Y_range=Y_range,
            IS_curve=IS_curve,
            LM_curve=LM_curve,
            BP_curve=BP_curve,
            equilibrium=self.result
        )

    def trilemma_analysis(self) -> Dict:
        """
        Analiza la trílema de Mundell-Fleming.

        Returns:
            Dict con los tres objetivos y cuáles son compatibles
        """
        fijo = self.config.regimen_cambiario == RegimenCambiario.FIJO
        movil = self.config.movilidad_capital == MovilidadCapital.PERFECTA

        # Con tipo fijo + movilidad perfecta => política monetaria NO independiente
        # Con tipo flexible + movilidad perfecta => política monetaria independiente
        # Con tipo fijo + sin movilidad => política monetaria independiente

        return {
            "tipo_cambio_fijo": fijo,
            "movilidad_capital_perfecta": movil,
            "politica_monetaria_independiente": not (fijo and movil),
            "trilema": "Solo pueden elegirse 2 de 3",
            "compatibles": [
                "Tipo fijo + Movilidad perfecta → Sin política monetaria independiente",
                "Tipo flexible + Movilidad perfecta → Con política monetaria independiente",
                "Tipo fijo + Sin movilidad → Con política monetaria independiente"
            ]
        }

    def policy_effectiveness(self, policy: str) -> Dict:
        """
        Analiza la efectividad de políticas según el régimen.

        Args:
            policy: "fiscal" o "monetaria"

        Returns:
            Dict con análisis de efectividad
        """
        fijo = self.config.regimen_cambiario == RegimenCambiario.FIJO
        movil = self.config.movilidad_capital == MovilidadCapital.PERFECTA

        if policy == "fiscal":
            if fijo and movil:
                efectividad = "máxima"
                razon = "Con tipo fijo y movilidad perfecta, la política fiscal es muy efectiva (no hay crowding-out)"
            elif not fijo and movil:
                efectividad = "nula"
                razon = "Con tipo flexible y movilidad perfecta, la expansión fiscal solo aprecia la moneda (crowding-out total vía tipo de cambio)"
            else:
                efectividad = "parcial"
                razon = "Efectividad intermedia dependiendo del grado de movilidad"
        else:  # monetaria
            if not fijo and movil:
                efectividad = "máxima"
                razon = "Con tipo flexible y movilidad perfecta, la política monetaria es muy efectiva"
            elif fijo and movil:
                efectividad = "nula"
                razon = "Con tipo fijo y movilidad perfecta, la política monetaria es inefectiva (se ajusta vía reservas)"
            else:
                efectividad = "parcial"
                razon = "Efectividad intermedia dependiendo del grado de movilidad"

        return {
            "politica": policy,
            "efectividad": efectividad,
            "razon": razon,
            "regimen": "fijo" if fijo else "flexible",
            "movilidad": "perfecta" if movil else "imperfecta/nula"
        }
