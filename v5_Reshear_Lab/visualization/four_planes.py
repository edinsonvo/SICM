"""
SICM v5 Research Lab — Vista de Cuatro Planos
==============================================
Visualización completa de los 4 planos del modelo.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, List
from ..core.parameters import EconomyConfig, TipoEconomia
from ..models.islm import ISLMModel
from ..models.mundell_fleming import MundellFlemingModel
from ..models.classical_closed import ClassicalClosedModel
from ..models.classical_open import ClassicalOpenModel


class FourPlanesView:
    """
    Vista de cuatro planos para análisis completo.

    Keynesiano:
    - IS-LM
    - DA-OA
    - Mercado de trabajo
    - Mecanismo de transmisión

    Clásico:
    - Mercado de trabajo
    - Producción
    - Fondos prestables
    - Sector externo (si es abierto)
    """

    def __init__(self, config: EconomyConfig):
        self.config = config

    def plot_keynesian(self, width: int = 1200, height: int = 900) -> go.Figure:
        """
        Cuatro planos para modelo Keynesiano.
        """
        model = ISLMModel(self.config)
        data = model.generate_curves()
        eq = data.equilibrium

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Plano 1: IS-LM',
                'Plano 2: Demanda Agregada - Oferta Agregada',
                'Plano 3: Mercado de Trabajo',
                'Plano 4: Mecanismo de Transmisión'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )

        # === PLANO 1: IS-LM ===
        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.IS_curve,
            mode='lines', name='IS', line=dict(color='#2E86AB', width=2.5)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.LM_curve,
            mode='lines', name='LM', line=dict(color='#A23B72', width=2.5)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[eq.Y], y=[eq.r],
            mode='markers', name='E',
            marker=dict(size=12, color='#F18F01', symbol='diamond')
        ), row=1, col=1)

        fig.update_xaxes(title_text='Y', row=1, col=1)
        fig.update_yaxes(title_text='r', row=1, col=1)

        # === PLANO 2: DA-OA (Keynesiano) ===
        Y_range = np.linspace(0, 2000, 100)
        # OA keynesiana: horizontal (precios fijos a corto plazo)
        OA_keynes = np.full_like(Y_range, eq.P)
        # DA: desde IS-LM con P variable
        DA = np.zeros_like(Y_range)
        for i, Y in enumerate(Y_range):
            # r desde IS
            c = self.config.consumo
            inv = self.config.inversion
            r_is = (c.c0 + inv.I0 + self.config.G - c.c1*c.T - (1-c.c1)*Y) / inv.b
            # P desde LM con ese r
            d = self.config.dinero
            if d.k*Y - d.h*r_is > 0:
                DA[i] = d.M / (d.k*Y - d.h*r_is)
            else:
                DA[i] = np.nan

        fig.add_trace(go.Scatter(
            x=Y_range, y=OA_keynes,
            mode='lines', name='OA (Keynes)', line=dict(color='#C73E1D', width=2.5)
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=Y_range, y=DA,
            mode='lines', name='DA', line=dict(color='#2E86AB', width=2.5)
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[eq.Y], y=[eq.P],
            mode='markers', name='E',
            marker=dict(size=12, color='#F18F01', symbol='diamond')
        ), row=1, col=2)

        fig.update_xaxes(title_text='Y', row=1, col=2)
        fig.update_yaxes(title_text='P', row=1, col=2)

        # === PLANO 3: Mercado de Trabajo (Keynesiano) ===
        L_range = np.linspace(0, 200, 100)
        p = self.config.produccion
        # Demanda de trabajo: PML
        PML = p.productividad_marginal_trabajo(L_range)
        # Oferta de trabajo (keynesiana: desempleo persistente)
        # Oferta efectiva = empleo efectivo
        Ns = np.full_like(L_range, eq.L)

        fig.add_trace(go.Scatter(
            x=L_range, y=PML,
            mode='lines', name='Nd (PML)', line=dict(color='#2E86AB', width=2.5)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=L_range, y=Ns,
            mode='lines', name='Ns efectiva', line=dict(color='#A23B72', width=2.5)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[eq.L], y=[PML[min(int(eq.L/2), len(PML)-1)]],
            mode='markers', name='E',
            marker=dict(size=12, color='#F18F01', symbol='diamond')
        ), row=2, col=1)

        fig.update_xaxes(title_text='L (Empleo)', row=2, col=1)
        fig.update_yaxes(title_text='W/P', row=2, col=1)

        # === PLANO 4: Mecanismo de Transmisión ===
        # Diagrama de flujo
        fig.add_trace(go.Scatter(
            x=[0.5, 1.5, 2.5, 3.5, 4.5],
            y=[0.5, 0.5, 0.5, 0.5, 0.5],
            mode='markers+text',
            text=['G↑', 'IS→', 'Y↑', 'L↑', 'r↑'],
            textposition='top center',
            marker=dict(size=20, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']),
            name='Transmisión'
        ), row=2, col=2)

        # Flechas
        for i in range(4):
            fig.add_annotation(
                x=i+0.75, y=0.5,
                ax=i+0.25, ay=0.5,
                xref='x4', yref='y4',
                axref='x4', ayref='y4',
                showarrow=True,
                arrowhead=2, arrowsize=1.5, arrowwidth=2,
                arrowcolor='gray'
            )

        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

        fig.update_layout(
            title_text='Modelo Keynesiano: Cuatro Planos',
            width=width, height=height,
            template='plotly_white',
            showlegend=False
        )

        return fig

    def plot_classical(self, width: int = 1200, height: int = 900) -> go.Figure:
        """
        Cuatro planos para modelo Clásico.
        """
        model = ClassicalClosedModel(self.config)
        data = model.generate_curves()
        eq = data.equilibrium

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Plano 1: Mercado de Trabajo',
                'Plano 2: Función de Producción',
                'Plano 3: Mercado de Fondos Prestables',
                'Plano 4: OA-DA'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )

        # === PLANO 1: Mercado de Trabajo ===
        fig.add_trace(go.Scatter(
            x=data.L_range, y=data.demanda_trabajo,
            mode='lines', name='Nd', line=dict(color='#2E86AB', width=2.5)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[eq.L], y=[data.demanda_trabajo[min(int(eq.L), len(data.demanda_trabajo)-1)]],
            mode='markers', name='E',
            marker=dict(size=12, color='#F18F01', symbol='diamond')
        ), row=1, col=1)

        fig.update_xaxes(title_text='L', row=1, col=1)
        fig.update_yaxes(title_text='W/P', row=1, col=1)

        # === PLANO 2: Función de Producción ===
        Y_prod = self.config.produccion.produccion(data.L_range)
        fig.add_trace(go.Scatter(
            x=data.L_range, y=Y_prod,
            mode='lines', name='Y=F(K,L)', line=dict(color='#A23B72', width=2.5)
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[eq.L], y=[eq.Y],
            mode='markers', name='E',
            marker=dict(size=12, color='#F18F01', symbol='diamond')
        ), row=1, col=2)

        fig.update_xaxes(title_text='L', row=1, col=2)
        fig.update_yaxes(title_text='Y', row=1, col=2)

        # === PLANO 3: Fondos Prestables ===
        fig.add_trace(go.Scatter(
            x=data.r_range, y=data.ahorro,
            mode='lines', name='S', line=dict(color='#2E86AB', width=2.5)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.r_range, y=data.inversion,
            mode='lines', name='I', line=dict(color='#C73E1D', width=2.5)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[eq.r], y=[eq.I],
            mode='markers', name='E',
            marker=dict(size=12, color='#F18F01', symbol='diamond')
        ), row=2, col=1)

        fig.update_xaxes(title_text='r', row=2, col=1)
        fig.update_yaxes(title_text='S, I', row=2, col=1)

        # === PLANO 4: OA-DA ===
        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.oferta_agregada,
            mode='lines', name='OA', line=dict(color='#C73E1D', width=2.5)
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=data.Y_range, y=data.demanda_agregada,
            mode='lines', name='DA', line=dict(color='#2E86AB', width=2.5)
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=[eq.Y], y=[eq.P],
            mode='markers', name='E',
            marker=dict(size=12, color='#F18F01', symbol='diamond')
        ), row=2, col=2)

        fig.update_xaxes(title_text='Y', row=2, col=2)
        fig.update_yaxes(title_text='P', row=2, col=2)

        fig.update_layout(
            title_text='Modelo Clásico: Cuatro Planos',
            width=width, height=height,
            template='plotly_white',
            showlegend=False
        )

        return fig

    def plot(self, width: int = 1200, height: int = 900) -> go.Figure:
        """Genera los 4 planos según el tipo de economía/configuración"""
        if self.config.tipo_economia == TipoEconomia.CERRADA:
            # Por defecto usamos keynesiano para cerrada
            return self.plot_keynesian(width, height)
        else:
            # Para abierta, podemos usar Mundell-Fleming o clásico
            return self.plot_classical(width, height)
