"""
SICM v5 Research Lab — Mecanismo de Transmisión
================================================
Panel explicativo automático del mecanismo de transmisión de choques.
"""
import plotly.graph_objects as go
from typing import List, Dict, Optional
from ..core.shocks import Shock, TipoShock
from ..core.parameters import EconomyConfig
from ..core.equilibrium import EquilibriumResult


class TransmissionMechanism:
    """
    Genera visualizaciones del mecanismo de transmisión de políticas económicas.

    Ejemplo:
    G ↑ → IS → derecha → Y ↑ → L(Y,r) ↑ → r ↑
    """

    def __init__(self, config: EconomyConfig):
        self.config = config

    def generate_fiscal_expansion(self, delta_G: float = 50) -> Dict:
        """
        Genera el mecanismo de transmisión de una expansión fiscal.

        Args:
            delta_G: Aumento en gasto público

        Returns:
            Dict con pasos y visualización
        """
        pasos = [
            {
                "nodo": "G ↑",
                "descripcion": f"El gobierno aumenta el gasto público en {delta_G}",
                "efecto": "Inicial",
                "color": "#2E86AB"
            },
            {
                "nodo": "DA ↑",
                "descripcion": "La demanda agregada aumenta directamente",
                "efecto": "G es componente de DA",
                "color": "#3A7D44"
            },
            {
                "nodo": "IS →",
                "descripcion": "La curva IS se desplaza hacia la derecha",
                "efecto": "Para cada r, Y es mayor",
                "color": "#A23B72"
            },
            {
                "nodo": "Y ↑",
                "descripcion": "El producto de equilibrio aumenta",
                "efecto": "Multiplicador keynesiano",
                "color": "#F18F01"
            },
            {
                "nodo": "L(Y,r) ↑",
                "descripcion": "La demanda de dinero aumenta (transacciones)",
                "efecto": "k·ΔY",
                "color": "#C73E1D"
            },
            {
                "nodo": "r ↑",
                "descripcion": "El tipo de interés sube (con M constante)",
                "efecto": "Crowding-out parcial",
                "color": "#3B1F2B"
            },
            {
                "nodo": "I ↓",
                "descripcion": "La inversión privada disminuye",
                "efecto": "b·Δr",
                "color": "#6B4C3B"
            },
            {
                "nodo": "Y ↓ (parcial)",
                "descripcion": "Efecto de crowding-out parcial",
                "efecto": "ΔY final < ΔY inicial",
                "color": "#8B5A2B"
            }
        ]

        return {
            "politica": "Expansión Fiscal",
            "magnitud": delta_G,
            "pasos": pasos,
            "conclusion": "La expansión fiscal aumenta Y pero con crowding-out parcial de I"
        }

    def generate_monetary_expansion(self, delta_M: float = 100) -> Dict:
        """
        Genera el mecanismo de transmisión de una expansión monetaria.

        Args:
            delta_M: Aumento en oferta monetaria

        Returns:
            Dict con pasos y visualización
        """
        pasos = [
            {
                "nodo": "M ↑",
                "descripcion": f"El banco central aumenta la oferta monetaria en {delta_M}",
                "efecto": "Inicial",
                "color": "#2E86AB"
            },
            {
                "nodo": "(M/P) ↑",
                "descripcion": "La oferta monetaria real aumenta",
                "efecto": "Si P es fijo a corto plazo",
                "color": "#3A7D44"
            },
            {
                "nodo": "LM ↓",
                "descripcion": "La curva LM se desplaza hacia abajo/derecha",
                "efecto": "Exceso de oferta de dinero",
                "color": "#A23B72"
            },
            {
                "nodo": "r ↓",
                "descripcion": "El tipo de interés baja",
                "efecto": "Para restablecer equilibrio en mercado de dinero",
                "color": "#F18F01"
            },
            {
                "nodo": "I ↑",
                "descripcion": "La inversión privada aumenta",
                "efecto": "b·Δr",
                "color": "#C73E1D"
            },
            {
                "nodo": "DA ↑",
                "descripcion": "La demanda agregada aumenta",
                "efecto": "I es componente de DA",
                "color": "#3B1F2B"
            },
            {
                "nodo": "Y ↑",
                "descripcion": "El producto de equilibrio aumenta",
                "efecto": "Multiplicador",
                "color": "#6B4C3B"
            },
            {
                "nodo": "L(Y,r) ↑",
                "descripcion": "La demanda de dinero aumenta",
                "efecto": "k·ΔY - h·Δr",
                "color": "#8B5A2B"
            }
        ]

        return {
            "politica": "Expansión Monetaria",
            "magnitud": delta_M,
            "pasos": pasos,
            "conclusion": "La expansión monetaria aumenta Y vía reducción de r y estimulo de I"
        }

    def generate_supply_shock(self, delta_A: float = -0.2) -> Dict:
        """
        Genera el mecanismo de transmisión de un choque de oferta negativo.

        Args:
            delta_A: Cambio en productividad (negativo = shock negativo)

        Returns:
            Dict con pasos y visualización
        """
        paso_signo = "↑" if delta_A > 0 else "↓"
        efecto = "positivo" if delta_A > 0 else "negativo"

        pasos = [
            {
                "nodo": f"A {paso_signo}",
                "descripcion": f"Shock de oferta {efecto}: productividad cambia en {delta_A}",
                "efecto": "Inicial",
                "color": "#2E86AB"
            },
            {
                "nodo": "PML ↓",
                "descripcion": "La productividad marginal del trabajo disminuye",
                "efecto": "Menor output por trabajador",
                "color": "#3A7D44"
            },
            {
                "nodo": "Nd ↓",
                "descripcion": "La demanda de trabajo se reduce",
                "efecto": "Las empresas contratan menos",
                "color": "#A23B72"
            },
            {
                "nodo": "Y* ↓",
                "descripcion": "El producto potencial disminuye",
                "efecto": "Menor capacidad productiva",
                "color": "#F18F01"
            },
            {
                "nodo": "OA ←",
                "descripcion": "La curva de oferta agregada se desplaza izquierda",
                "efecto": "Menor oferta a cada precio",
                "color": "#C73E1D"
            },
            {
                "nodo": "P ↑",
                "descripcion": "El nivel de precios sube",
                "efecto": "Estanflación",
                "color": "#3B1F2B"
            },
            {
                "nodo": "Y ↓",
                "descripcion": "El producto de equilibrio disminuye",
                "efecto": "Menor actividad económica",
                "color": "#6B4C3B"
            },
            {
                "nodo": "Desempleo ↑",
                "descripcion": "El desempleo aumenta",
                "efecto": "Estanflación: inflación + desempleo",
                "color": "#8B5A2B"
            }
        ]

        return {
            "politica": f"Shock de Oferta {efecto.title()}",
            "magnitud": delta_A,
            "pasos": pasos,
            "conclusion": "Los choques de oferta generan estanflación: menor Y con mayor P"
        }

    def plot(self, mechanism: Dict, width: int = 1000, height: int = 600) -> go.Figure:
        """
        Genera una visualización del mecanismo de transmisión.

        Args:
            mechanism: Dict generado por los métodos anteriores

        Returns:
            Figura de Plotly
        """
        pasos = mechanism["pasos"]
        n = len(pasos)

        fig = go.Figure()

        # Posiciones en zigzag
        x_pos = []
        y_pos = []
        for i in range(n):
            x_pos.append(i)
            y_pos.append(0.5 + (0.3 if i % 2 == 0 else -0.3))

        # Nodos
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            text=[p["nodo"] for p in pasos],
            textposition='top center',
            marker=dict(
                size=30,
                color=[p["color"] for p in pasos],
                line=dict(width=2, color='white')
            ),
            hovertext=[f"{p['nodo']}<br>{p['descripcion']}<br><i>{p['efecto']}</i>" for p in pasos],
            hoverinfo='text',
            showlegend=False
        ))

        # Flechas entre nodos
        for i in range(n - 1):
            fig.add_annotation(
                x=x_pos[i+1] - 0.15,
                y=y_pos[i+1],
                ax=x_pos[i] + 0.15,
                ay=y_pos[i],
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='gray'
            )

        # Título y conclusión
        fig.update_layout(
            title=dict(
                text=f"Mecanismo de Transmisión: {mechanism['politica']} (Δ={mechanism['magnitud']})",
                font=dict(size=18)
            ),
            annotations=[
                dict(
                    text=f"<b>Conclusión:</b> {mechanism['conclusion']}",
                    xref='paper', yref='paper',
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=14, color='#333'),
                    bgcolor='#f0f0f0',
                    borderpad=10
                )
            ],
            xaxis=dict(visible=False, range=[-0.5, n-0.5]),
            yaxis=dict(visible=False, range=[-0.5, 1.5]),
            width=width, height=height,
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=100)
        )

        return fig

    def get_text_summary(self, mechanism: Dict) -> str:
        """Genera un resumen textual del mecanismo"""
        lines = [
            f"=== {mechanism['politica']} (Δ={mechanism['magnitud']}) ===",
            ""
        ]
        for i, paso in enumerate(mechanism['pasos'], 1):
            lines.append(f"{i}. {paso['nodo']}: {paso['descripcion']}")
            lines.append(f"   → Efecto: {paso['efecto']}")
        lines.append("")
        lines.append(f"Conclusión: {mechanism['conclusion']}")
        return "
".join(lines)
