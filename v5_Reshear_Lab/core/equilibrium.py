"""
SICM v5 Research Lab — Solucionador de Equilibrio
=================================================
EquilibriumSolver: Calcula PIB, Inflación, Interés, Empleo, Tipo de cambio.
El verdadero corazón del sistema.
"""
import numpy as np
from scipy.optimize import fsolve, brentq
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from .parameters import EconomyConfig, TipoEconomia, RegimenCambiario, MovilidadCapital


@dataclass
class EquilibriumResult:
    """Resultado del cálculo de equilibrio"""
    Y: float          # Producto (PIB)
    r: float          # Tipo de interés
    P: float          # Nivel de precios
    L: float          # Empleo
    E: float          # Tipo de cambio
    C: float          # Consumo
    I: float          # Inversión
    G: float          # Gasto público
    NX: float         # Exportaciones netas
    M_s: float        # Oferta monetaria real
    M_d: float        # Demanda de dinero

    # Indicadores derivados
    desempleo: float = 0.0
    inflacion: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "PIB (Y)": round(self.Y, 2),
            "Interés (r)": round(self.r, 4),
            "Precios (P)": round(self.P, 4),
            "Empleo (L)": round(self.L, 2),
            "Tipo de cambio (E)": round(self.E, 4),
            "Consumo (C)": round(self.C, 2),
            "Inversión (I)": round(self.I, 2),
            "Gasto público (G)": round(self.G, 2),
            "Exportaciones netas (NX)": round(self.NX, 2),
            "Desempleo (%)": round(self.desempleo, 2),
            "Inflación (%)": round(self.inflacion, 2)
        }


class EquilibriumSolver:
    """
    Solucionador de equilibrio macroeconómico.
    Soporta múltiples modelos: IS-LM, Mundell-Fleming, Clásico.
    """

    def __init__(self, config: EconomyConfig):
        self.config = config

    # ============================================================
    # IS-LM CERRADO (Fase 2.1)
    # ============================================================
    def solve_islm_cerrado(self) -> EquilibriumResult:
        """
        Resuelve el modelo IS-LM para economía cerrada.

        IS: Y = C + I + G
        LM: M/P = kY - hr
        """
        c = self.config.consumo
        inv = self.config.inversion
        d = self.config.dinero
        G = self.config.G

        # Función IS: Y = c0 + c1*(Y-T) + I0 - b*r + G
        # Reorganizando: Y*(1-c1) = c0 - c1*T + I0 - b*r + G
        # Y = (c0 - c1*T + I0 + G - b*r) / (1-c1)

        # Función LM: M/P = k*Y - h*r
        # r = (k*Y - M/P) / h

        # Sustituyendo LM en IS:
        # Y = (c0 - c1*T + I0 + G - b*(k*Y - M/P)/h) / (1-c1)
        # Y*(1-c1) = c0 - c1*T + I0 + G - (b*k*Y)/h + (b*M)/(P*h)
        # Y*[(1-c1) + b*k/h] = c0 - c1*T + I0 + G + (b*M)/(P*h)

        denom = (1 - c.c1) + (inv.b * d.k) / d.h
        numer = c.c0 - c.c1 * c.T + inv.I0 + G + (inv.b * d.M) / (d.P * d.h)

        Y_eq = numer / denom
        r_eq = (d.k * Y_eq - d.M / d.P) / d.h

        # Calcular componentes
        C_eq = c.consumo(Y_eq)
        I_eq = inv.inversion(r_eq)
        M_s = d.oferta_dinero_real()
        M_d = d.demanda_dinero(Y_eq, r_eq)

        # Empleo (aproximación Keynesiana: demanda determina empleo)
        L_eq = min(Y_eq / self.config.produccion.A, self.config.produccion.L_bar)
        desempleo = max(0, (self.config.produccion.L_bar - L_eq) / self.config.produccion.L_bar * 100)

        return EquilibriumResult(
            Y=Y_eq,
            r=r_eq,
            P=d.P,
            L=L_eq,
            E=self.config.externo.E,
            C=C_eq,
            I=I_eq,
            G=G,
            NX=0.0,
            M_s=M_s,
            M_d=M_d,
            desempleo=desempleo,
            inflacion=0.0
        )

    # ============================================================
    # MUNDELL-FLEMING (Fase 2.2)
    # ============================================================
    def solve_mundell_fleming(self) -> EquilibriumResult:
        """
        Resuelve el modelo Mundell-Fleming para economía abierta.

        Soporta:
        - Tipo de cambio fijo vs flexible
        - Movilidad perfecta/imperfecta/nula de capital
        """
        c = self.config.consumo
        inv = self.config.inversion
        d = self.config.dinero
        ext = self.config.externo
        G = self.config.G

        if self.config.regimen_cambiario == RegimenCambiario.FLEXIBLE:
            return self._solve_mf_flexible(c, inv, d, ext, G)
        else:
            return self._solve_mf_fijo(c, inv, d, ext, G)

    def _solve_mf_flexible(self, c, inv, d, ext, G):
        """Mundell-Fleming con tipo de cambio flexible"""

        if self.config.movilidad_capital == MovilidadCapital.PERFECTA:
            # Con movilidad perfecta: r = r_f (paridad descubierta)
            r_eq = ext.r_f

            # IS*: Y = C + I + G + NX
            # NX = X0 - m*Y - n*q (donde q = tipo de cambio real)
            # En equilibrio: Y = (c0 - c1*T + I0 - b*r_f + G + X0) / (1 - c1 + m)

            denom = 1 - c.c1 + ext.m
            numer = c.c0 - c.c1 * c.T + inv.I0 - inv.b * r_eq + G + ext.X0
            Y_eq = numer / denom

            # Tipo de cambio de equilibrio desde LM
            # M/P = k*Y - h*r => con r = r_f
            # E se ajusta para que se cumpla
            # Pero en MF flexible con movilidad perfecta, E se determina por BP
            # Simplificación: E se ajusta para que NX = 0 en equilibrio
            NX_eq = ext.X0 - ext.m * Y_eq
            # Ajuste del tipo de cambio para equilibrar BP
            E_eq = ext.P_f / d.P  # Tipo de cambio de paridad

        elif self.config.movilidad_capital == MovilidadCapital.NULA:
            # Sin movilidad: LM determina Y, IS* determina E
            # LM: M/P = k*Y - h*r => Y = (M/P + h*r)/k
            # Pero r es endógeno, necesitamos resolver simultáneo

            def sistema(vars):
                Y, r, E = vars
                # IS: Y = c0 + c1*(Y-T) + I0 - b*r + G + X0 - m*Y
                f1 = Y - (c.c0 + c.c1*(Y-c.T) + inv.I0 - inv.b*r + G + ext.X0 - ext.m*Y)
                # LM: M/P = k*Y - h*r
                f2 = d.M/d.P - (d.k*Y - d.h*r)
                # BP: NX = 0 (sin movilidad, balanza comercial = 0)
                # q = E*P/P_f, NX = X0 - m*Y - v*q
                # Simplificación: BP se equilibra con E
                f3 = E - ext.E  # E se ajusta libremente
                return [f1, f2, f3]

            sol = fsolve(sistema, [1000, 0.05, 1.0])
            Y_eq, r_eq, E_eq = sol
            NX_eq = ext.X0 - ext.m * Y_eq

        else:  # Movilidad imperfecta
            # Sistema completo
            def sistema(vars):
                Y, r, E = vars
                f1 = Y - (c.c0 + c.c1*(Y-c.T) + inv.I0 - inv.b*r + G + ext.X0 - ext.m*Y)
                f2 = d.M/d.P - (d.k*Y - d.h*r)
                # BP: NX + KA = 0, KA depende de (r - r_f)
                KA = 50 * (r - ext.r_f)  # Flujos de capital
                f3 = (ext.X0 - ext.m*Y) + KA
                return [f1, f2, f3]

            sol = fsolve(sistema, [1000, 0.05, 1.0])
            Y_eq, r_eq, E_eq = sol
            NX_eq = ext.X0 - ext.m * Y_eq

        C_eq = c.consumo(Y_eq)
        I_eq = inv.inversion(r_eq)
        M_s = d.oferta_dinero_real()
        M_d = d.demanda_dinero(Y_eq, r_eq)
        L_eq = min(Y_eq / self.config.produccion.A, self.config.produccion.L_bar)
        desempleo = max(0, (self.config.produccion.L_bar - L_eq) / self.config.produccion.L_bar * 100)

        return EquilibriumResult(
            Y=Y_eq, r=r_eq, P=d.P, L=L_eq, E=E_eq,
            C=C_eq, I=I_eq, G=G, NX=NX_eq,
            M_s=M_s, M_d=M_d, desempleo=desempleo, inflacion=0.0
        )

    def _solve_mf_fijo(self, c, inv, d, ext, G):
        """Mundell-Fleming con tipo de cambio fijo"""
        # Con tipo de cambio fijo, E está dado
        E_eq = ext.E

        if self.config.movilidad_capital == MovilidadCapital.PERFECTA:
            # r = r_f (paridad de intereses)
            r_eq = ext.r_f
            # Y desde IS con E fijo
            denom = 1 - c.c1 + ext.m
            numer = c.c0 - c.c1 * c.T + inv.I0 - inv.b * r_eq + G + ext.X0
            Y_eq = numer / denom
            # M se ajusta endógenamente (política monetaria no es independiente)
            # M/P = k*Y - h*r => M = P*(k*Y - h*r)
            M_implicita = d.P * (d.k * Y_eq - d.h * r_eq)
        else:
            # Sistema con E fijo
            def sistema(vars):
                Y, r = vars
                f1 = Y - (c.c0 + c.c1*(Y-c.T) + inv.I0 - inv.b*r + G + ext.X0 - ext.m*Y)
                f2 = d.M/d.P - (d.k*Y - d.h*r)
                return [f1, f2]

            sol = fsolve(sistema, [1000, 0.05])
            Y_eq, r_eq = sol
            M_implicita = d.M

        C_eq = c.consumo(Y_eq)
        I_eq = inv.inversion(r_eq)
        NX_eq = ext.X0 - ext.m * Y_eq
        M_s = M_implicita / d.P
        M_d = d.demanda_dinero(Y_eq, r_eq)
        L_eq = min(Y_eq / self.config.produccion.A, self.config.produccion.L_bar)
        desempleo = max(0, (self.config.produccion.L_bar - L_eq) / self.config.produccion.L_bar * 100)

        return EquilibriumResult(
            Y=Y_eq, r=r_eq, P=d.P, L=L_eq, E=E_eq,
            C=C_eq, I=I_eq, G=G, NX=NX_eq,
            M_s=M_s, M_d=M_d, desempleo=desempleo, inflacion=0.0
        )

    # ============================================================
    # CLÁSICO CERRADO (Fase 2.3)
    # ============================================================
    def solve_classical_closed(self) -> EquilibriumResult:
        """
        Resuelve el modelo clásico cerrado.

        Mercados: Trabajo, Producción, Fondos prestables.
        Precios flexibles, pleno empleo.
        """
        p = self.config.produccion
        c = self.config.consumo
        inv = self.config.inversion
        d = self.config.dinero
        G = self.config.G

        # Mercado de trabajo: PML = W/P (salario real)
        # Oferta de trabajo = L_bar (pleno empleo)
        L_eq = p.L_bar

        # Producción de pleno empleo
        Y_eq = p.produccion(L_eq)

        # Mercado de fondos prestables: S = I
        # S = Y - T - C = Y - T - c0 - c1*(Y-T) = (1-c1)*(Y-T) - c0
        Ahorro = Y_eq - c.T - c.consumo(Y_eq)
        # I = I0 - b*r => r = (I0 - S)/b
        r_eq = (inv.I0 - Ahorro) / inv.b

        # Mercado de dinero (neutro): M determina P
        # M/P = k*Y - h*r => P = M / (k*Y - h*r)
        M_d_val = d.k * Y_eq - d.h * r_eq
        P_eq = d.M / M_d_val if M_d_val > 0 else 1.0

        C_eq = c.consumo(Y_eq)
        I_eq = inv.inversion(r_eq)
        M_s = d.M / P_eq
        M_d = d.demanda_dinero(Y_eq, r_eq)

        return EquilibriumResult(
            Y=Y_eq, r=r_eq, P=P_eq, L=L_eq, E=self.config.externo.E,
            C=C_eq, I=I_eq, G=G, NX=0.0,
            M_s=M_s, M_d=M_d, desempleo=0.0, inflacion=0.0
        )

    # ============================================================
    # CLÁSICO ABIERTO (Fase 2.4)
    # ============================================================
    def solve_classical_open(self) -> EquilibriumResult:
        """
        Resuelve el modelo clásico abierto.

        Añade: Tipo de cambio real, Exportaciones netas.
        """
        p = self.config.produccion
        c = self.config.consumo
        inv = self.config.inversion
        d = self.config.dinero
        ext = self.config.externo
        G = self.config.G

        # Pleno empleo
        L_eq = p.L_bar
        Y_eq = p.produccion(L_eq)

        # Tipo de cambio real de equilibrio
        q_eq = (ext.E * d.P) / ext.P_f

        # Exportaciones netas dependen de q
        # NX = X0 - m*Y - v*(q - q_bar)
        NX_eq = ext.X0 - ext.m * Y_eq

        # Mercado de fondos prestables: S + (T-G) = I + NX
        # O en apertura financiera: S - I = NX
        Ahorro = Y_eq - c.T - c.consumo(Y_eq)
        # r de equilibrio interno
        r_eq = (inv.I0 - (Ahorro - NX_eq)) / inv.b

        # Precios de equilibrio
        M_d_val = d.k * Y_eq - d.h * r_eq
        P_eq = d.M / M_d_val if M_d_val > 0 else 1.0

        C_eq = c.consumo(Y_eq)
        I_eq = inv.inversion(r_eq)
        M_s = d.M / P_eq
        M_d = d.demanda_dinero(Y_eq, r_eq)

        return EquilibriumResult(
            Y=Y_eq, r=r_eq, P=P_eq, L=L_eq, E=ext.E,
            C=C_eq, I=I_eq, G=G, NX=NX_eq,
            M_s=M_s, M_d=M_d, desempleo=0.0, inflacion=0.0
        )

    # ============================================================
    # MÉTODO UNIFICADO
    # ============================================================
    def solve(self, modelo: str = "auto") -> EquilibriumResult:
        """
        Resuelve el equilibrio según la configuración o modelo especificado.

        Args:
            modelo: "auto", "islm", "mundell_fleming", "classical_closed", "classical_open"
        """
        if modelo == "auto":
            if self.config.tipo_economia == TipoEconomia.CERRADA:
                return self.solve_islm_cerrado()
            else:
                return self.solve_mundell_fleming()
        elif modelo == "islm":
            return self.solve_islm_cerrado()
        elif modelo == "mundell_fleming":
            return self.solve_mundell_fleming()
        elif modelo == "classical_closed":
            return self.solve_classical_closed()
        elif modelo == "classical_open":
            return self.solve_classical_open()
        else:
            raise ValueError(f"Modelo no soportado: {modelo}")
