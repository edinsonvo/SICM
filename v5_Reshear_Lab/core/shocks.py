"""
SICM v5 Research Lab — Motor de Choques
=======================================
ShockEngine: Genera, aplica y narra choques económicos.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from .parameters import EconomyConfig, ParametrosConsumo, ParametrosInversion, ParametrosDinero


class TipoShock(Enum):
    FISCAL = "fiscal"
    MONETARIO = "monetario"
    OFERTA = "oferta"
    EXTERNO = "externo"


@dataclass
class Shock:
    """Representación de un choque económico"""
    tipo: TipoShock
    magnitud: float
    descripcion: str
    parametro_afectado: str
    duracion: int = 1  # Periodos

    def __repr__(self):
        return f"Shock({self.tipo.value}: {self.descripcion}, magnitud={self.magnitud})"


class ShockEngine:
    """
    Motor de choques económicos.
    Responsable de:
    - Desplazar curvas
    - Recalcular equilibrio
    - Generar narrativa económica
    """

    def __init__(self, config: EconomyConfig):
        self.config_base = config
        self.historial_choques: List[Shock] = []

    def aplicar_choque_fiscal(self, delta_G: float = 0.0, delta_T: float = 0.0, 
                               delta_c0: float = 0.0) -> tuple[EconomyConfig, Shock]:
        """
        Aplica un choque fiscal.

        Args:
            delta_G: Cambio en gasto público
            delta_T: Cambio en impuestos
            delta_c0: Cambio en consumo autónomo
        """
        nueva_config = self._clonar_config()

        # Aplicar cambios
        nueva_config.G += delta_G
        nueva_config.consumo.T += delta_T
        nueva_config.consumo.c0 += delta_c0

        # Determinar narrativa
        if delta_G > 0:
            desc = f"Expansión fiscal: G aumenta {delta_G}"
            efecto_IS = "derecha"
        elif delta_G < 0:
            desc = f"Contracción fiscal: G disminuye {abs(delta_G)}"
            efecto_IS = "izquierda"
        elif delta_T > 0:
            desc = f"Contracción fiscal: T aumenta {delta_T}"
            efecto_IS = "izquierda"
        elif delta_T < 0:
            desc = f"Expansión fiscal: T disminuye {abs(delta_T)}"
            efecto_IS = "derecha"
        else:
            desc = "Cambio en consumo autónomo"
            efecto_IS = "derecha" if delta_c0 > 0 else "izquierda"

        shock = Shock(
            tipo=TipoShock.FISCAL,
            magnitud=max(abs(delta_G), abs(delta_T), abs(delta_c0)),
            descripcion=desc,
            parametro_afectado=f"G={nueva_config.G}, T={nueva_config.consumo.T}",
            duracion=1
        )

        self.historial_choques.append(shock)

        narrativa = self._generar_narrativa_fiscal(delta_G, delta_T, efecto_IS)

        return nueva_config, shock, narrativa

    def aplicar_choque_monetario(self, delta_M: float = 0.0, delta_P: float = 0.0) -> tuple[EconomyConfig, Shock, str]:
        """
        Aplica un choque monetario.

        Args:
            delta_M: Cambio en oferta monetaria
            delta_P: Cambio en nivel de precios
        """
        nueva_config = self._clonar_config()

        nueva_config.dinero.M += delta_M
        nueva_config.dinero.P += delta_P

        if delta_M > 0:
            desc = f"Expansión monetaria: M aumenta {delta_M}"
            efecto_LM = "abajo/derecha"
        elif delta_M < 0:
            desc = f"Contracción monetaria: M disminuye {abs(delta_M)}"
            efecto_LM = "arriba/izquierda"
        else:
            desc = "Cambio en nivel de precios"
            efecto_LM = "indeterminado"

        shock = Shock(
            tipo=TipoShock.MONETARIO,
            magnitud=max(abs(delta_M), abs(delta_P)),
            descripcion=desc,
            parametro_afectado=f"M={nueva_config.dinero.M}, P={nueva_config.dinero.P}",
            duracion=1
        )

        self.historial_choques.append(shock)

        narrativa = self._generar_narrativa_monetario(delta_M, efecto_LM)

        return nueva_config, shock, narrativa

    def aplicar_choque_oferta(self, delta_A: float = 0.0, delta_K: float = 0.0) -> tuple[EconomyConfig, Shock, str]:
        """
        Aplica un choque de oferta.

        Args:
            delta_A: Cambio en productividad
            delta_K: Cambio en stock de capital
        """
        nueva_config = self._clonar_config()

        nueva_config.produccion.A += delta_A
        nueva_config.produccion.K += delta_K

        if delta_A > 0:
            desc = f"Shock positivo de oferta: A aumenta {delta_A}"
        elif delta_A < 0:
            desc = f"Shock negativo de oferta: A disminuye {abs(delta_A)}"
        elif delta_K > 0:
            desc = f"Acumulación de capital: K aumenta {delta_K}"
        else:
            desc = f"Destrucción de capital: K disminuye {abs(delta_K)}"

        shock = Shock(
            tipo=TipoShock.OFERTA,
            magnitud=max(abs(delta_A), abs(delta_K)),
            descripcion=desc,
            parametro_afectado=f"A={nueva_config.produccion.A}, K={nueva_config.produccion.K}",
            duracion=1
        )

        self.historial_choques.append(shock)

        narrativa = self._generar_narrativa_oferta(delta_A, delta_K)

        return nueva_config, shock, narrativa

    def aplicar_choque_externo(self, delta_X0: float = 0.0, delta_E: float = 0.0,
                                delta_r_f: float = 0.0) -> tuple[EconomyConfig, Shock, str]:
        """
        Aplica un choque externo.

        Args:
            delta_X0: Cambio en exportaciones autónomas
            delta_E: Cambio en tipo de cambio nominal
            delta_r_f: Cambio en tipo de interés extranjero
        """
        nueva_config = self._clonar_config()

        nueva_config.externo.X0 += delta_X0
        nueva_config.externo.E += delta_E
        nueva_config.externo.r_f += delta_r_f

        if delta_X0 > 0:
            desc = f"Aumento de exportaciones: X0 aumenta {delta_X0}"
        elif delta_X0 < 0:
            desc = f"Caída de exportaciones: X0 disminuye {abs(delta_X0)}"
        elif delta_E > 0:
            desc = f"Devaluación: E aumenta {delta_E}"
        elif delta_E < 0:
            desc = f"Revaluación: E disminuye {abs(delta_E)}"
        else:
            desc = f"Cambio en tasas internacionales: r_f = {nueva_config.externo.r_f}"

        shock = Shock(
            tipo=TipoShock.EXTERNO,
            magnitud=max(abs(delta_X0), abs(delta_E), abs(delta_r_f)),
            descripcion=desc,
            parametro_afectado=f"X0={nueva_config.externo.X0}, E={nueva_config.externo.E}, r_f={nueva_config.externo.r_f}",
            duracion=1
        )

        self.historial_choques.append(shock)

        narrativa = self._generar_narrativa_externo(delta_X0, delta_E, delta_r_f)

        return nueva_config, shock, narrativa

    def _clonar_config(self) -> EconomyConfig:
        """Crea una copia profunda de la configuración base"""
        import copy
        return copy.deepcopy(self.config_base)

    def _generar_narrativa_fiscal(self, delta_G, delta_T, efecto_IS) -> str:
        """Genera narrativa textual del choque fiscal"""
        pasos = []
        if delta_G != 0:
            pasos.append(f"G {'↑' if delta_G > 0 else '↓'} → DA se desplaza")
        if delta_T != 0:
            pasos.append(f"T {'↑' if delta_T > 0 else '↓'} → Yd {'↓' if delta_T > 0 else '↑'} → C {'↓' if delta_T > 0 else '↑'}")

        pasos.append(f"IS se desplaza hacia la {efecto_IS}")
        pasos.append("Nuevo equilibrio IS-LM")

        return " → ".join(pasos)

    def _generar_narrativa_monetario(self, delta_M, efecto_LM) -> str:
        """Genera narrativa textual del choque monetario"""
        if delta_M > 0:
            return f"M ↑ → (M/P) ↑ → r ↓ → I ↑ → Y ↑"
        elif delta_M < 0:
            return f"M ↓ → (M/P) ↓ → r ↑ → I ↓ → Y ↓"
        return "Cambio en precios → efecto sobre LM"

    def _generar_narrativa_oferta(self, delta_A, delta_K) -> str:
        """Genera narrativa textual del choque de oferta"""
        if delta_A > 0 or delta_K > 0:
            return "A/K ↑ → PML ↑ → Oferta laboral ↑ → Y* ↑ → OA se desplaza derecha → P ↓"
        return "A/K ↓ → PML ↓ → Oferta laboral ↓ → Y* ↓ → OA se desplaza izquierda → P ↑"

    def _generar_narrativa_externo(self, delta_X0, delta_E, delta_r_f) -> str:
        """Genera narrativa textual del choque externo"""
        if delta_X0 != 0:
            return f"X0 {'↑' if delta_X0 > 0 else '↓'} → NX {'↑' if delta_X0 > 0 else '↓'} → IS {'derecha' if delta_X0 > 0 else 'izquierda'}"
        elif delta_E != 0:
            return f"E {'↑' if delta_E > 0 else '↓'} → q {'↑' if delta_E > 0 else '↓'} → NX {'↑' if delta_E > 0 else '↓'}"
        return f"r_f {'↑' if delta_r_f > 0 else '↓'} → flujos de capital {'salida' if delta_r_f > delta_r_f else 'entrada'}"

    def obtener_historial(self) -> List[Shock]:
        """Devuelve el historial de choques aplicados"""
        return self.historial_choques.copy()

    def resetear(self):
        """Limpia el historial de choques"""
        self.historial_choques.clear()
