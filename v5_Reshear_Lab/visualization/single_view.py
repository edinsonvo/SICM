"""
SICM v5 Research Lab — Vista Simple
=====================================
Una sola gráfica para validación rápida.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict
from ..core.parameters import EconomyConfig
from ..models.islm import ISLMModel
from ..models.mundell_fleming import MundellFlemingModel
from ..models.classical_closed import ClassicalClosedModel
from ..models.classical_open import ClassicalOpenModel


class SingleView:
    """
    Vista simple de una sola gráfica.
    Permite validar rápidamente el modelo.
    """

    def __init__(self, config: EconomyConfig, modelo: str = "islm"):
        self.config = config
        self.modelo = modelo
        self.fig = None

    def plot(self, width: int = 800, height: int = 600) -> go.Figure:
        """Genera la gráfica simple según el modelo"""
        if self.modelo == "islm":
            return self._plot_islm(width, height)
        elif self.modelo == "mundell_fleming":
            return self._plot_mundell_fleming(width, height)
        elif self.modelo == "classical_closed":
            return self._plot_classical_closed(width, height)
        elif self.modelo == "classical_open":
            return self._plot_classical_open(width, height)
        else:
            raise ValueError(f"Modelo no soportado: {self.modelo}")

    def _plot_islm(self, width: int, height: int) -> go.Figure:
        """Gráfica IS-LM simple"""
        model = ISLMModel(self.config)
        data = model.generate_curves()
        eq = data.equilibrium

        fig = go.Figure()

        # Curva IS
        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.IS_curve,
            mode='lines', name='IS',
            line=dict(color='#2E86AB', width=3)
        ))

        # Curva LM
        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.LM_curve,
            mode='lines', name='LM',
            line=dict(color='#A23B72', width=3)
        ))

        # Punto de equilibrio
        fig.add_trace(go.Scatter(
            x=[eq.Y], y=[eq.r],
            mode='markers+text',
            name='Equilibrio',
            marker=dict(size=14, color='#F18F01', symbol='diamond'),
            text=[f"E<br>Y={eq.Y:.1f}<br>r={eq.r:.3f}"],
            textposition="top right"
        ))

        fig.update_layout(
            title='Modelo IS-LM: Economía Cerrada',
            xaxis_title='Producto (Y)',
            yaxis_title='Tipo de interés (r)',
            width=width, height=height,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig

    def _plot_mundell_fleming(self, width: int, height: int) -> go.Figure:
        """Gráfica Mundell-Fleming simple"""
        model = MundellFlemingModel(self.config)
        data = model.generate_curves()
        eq = data.equilibrium

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.IS_curve,
            mode='lines', name='IS*',
            line=dict(color='#2E86AB', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.LM_curve,
            mode='lines', name='LM',
            line=dict(color='#A23B72', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.BP_curve,
            mode='lines', name='BP',
            line=dict(color='#C73E1D', width=3, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=[eq.Y], y=[eq.r],
            mode='markers+text',
            name='Equilibrio',
            marker=dict(size=14, color='#F18F01', symbol='diamond'),
            text=[f"E<br>Y={eq.Y:.1f}<br>r={eq.r:.3f}<br>E={eq.E:.3f}"],
            textposition="top right"
        ))

        regime = "Fijo" if self.config.regimen_cambiario.value == "fijo" else "Flexible"
        movil = self.config.movilidad_capital.value

        fig.update_layout(
            title=f'Mundell-Fleming: TC {regime}, Movilidad {movil}',
            xaxis_title='Producto (Y)',
            yaxis_title='Tipo de interés (r)',
            width=width, height=height,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig

    def _plot_classical_closed(self, width: int, height: int) -> go.Figure:
        """Gráfica clásico cerrado simple (mercado de fondos)"""
        model = ClassicalClosedModel(self.config)
        data = model.generate_curves()
        eq = data.equilibrium

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.r_range, y=data.ahorro,
            mode='lines', name='Ahorro (S)',
            line=dict(color='#2E86AB', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=data.r_range, y=data.inversion,
            mode='lines', name='Inversión (I)',
            line=dict(color='#A23B72', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=[eq.r], y=[eq.I],
            mode='markers+text',
            name='Equilibrio',
            marker=dict(size=14, color='#F18F01', symbol='diamond'),
            text=[f"E<br>r={eq.r:.3f}<br>I=S={eq.I:.1f}"],
            textposition="top right"
        ))

        fig.update_layout(
            title='Modelo Clásico: Mercado de Fondos Prestables',
            xaxis_title='Tipo de interés (r)',
            yaxis_title='Ahorro / Inversión',
            width=width, height=height,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig

    def _plot_classical_open(self, width: int, height: int) -> go.Figure:
        """Gráfica clásico abierto simple (sector externo)"""
        model = ClassicalOpenModel(self.config)
        data = model.generate_curves()
        eq = data.equilibrium

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.q_range, y=data.exportaciones,
            mode='lines', name='Exportaciones (X)',
            line=dict(color='#2E86AB', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=data.q_range, y=data.importaciones,
            mode='lines', name='Importaciones (M)',
            line=dict(color='#A23B72', width=3)
        ))

        q_eq = (eq.E * eq.P) / self.config.externo.P_f
        fig.add_trace(go.Scatter(
            x=[q_eq], y=[eq.NX + data.importaciones[len(data.importaciones)//2]],
            mode='markers+text',
            name='Equilibrio',
            marker=dict(size=14, color='#F18F01', symbol='diamond'),
            text=[f"E<br>q={q_eq:.3f}<br>NX={eq.NX:.1f}"],
            textposition="top right"
        ))

        fig.update_layout(
            title='Modelo Clásico Abierto: Sector Externo',
            xaxis_title='Tipo de cambio real (q)',
            yaxis_title='Exportaciones / Importaciones',
            width=width, height=height,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig
