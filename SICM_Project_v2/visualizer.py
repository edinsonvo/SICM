"""Visualizaciones interactivas del SICM v2.0.

La clase :class:`Visualizer` genera figuras Plotly estáticas (sin
animaciones ni controles deslizantes automáticos) para los modelos
IS-LM, AD-AS y Mundell-Fleming, análisis de sensibilidad, series
temporales y el dashboard de indicadores.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import ISLMModel, ADASModel, MundellFlemingModel
from data_manager import VARIABLE_LABELS, VARIABLE_COLORS


class Visualizer:
    """Colección de métodos estáticos que construyen figuras Plotly."""

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    @staticmethod
    def _layout(title, x_title, y_title, height=500):
        """Configuración común de los gráficos (estilo limpio)."""
        return dict(
            title=dict(text=title, x=0.5),
            xaxis_title=x_title,
            yaxis_title=y_title,
            template="plotly_white",
            height=height,
            margin=dict(l=60, r=30, t=70, b=50),
            hovermode="x unified",
        )

    @staticmethod
    def _equilibrium_trace(eq, name, color="#2ca02c", symbol="star",
                           size=16, extra=None):
        """Trazo del punto de equilibrio (estrella verde por defecto)."""
        hovertemplate = f"{name}<br>%{{x:.2f}}, %{{y:.4f}}<extra></extra>"
        marker = dict(size=size, color=color, symbol=symbol,
                      line=dict(color="white", width=1))
        return go.Scatter(
            x=[eq["x"]], y=[eq["y"]], mode="markers",
            name=name, marker=marker, hovertemplate=hovertemplate,
        )

    @staticmethod
    def _y_range_around(center, factor=0.45, floor=1e-6):
        """Rango de Y centrado alrededor de un valor con ancho mínimo."""
        lo = center * (1 - factor)
        hi = center * (1 + factor)
        width = hi - lo
        if width < 40:
            mid = center
            lo = mid - 20
            hi = mid + 20
        if lo <= 0:
            lo = floor
        return np.linspace(lo, hi, 120)

    # ------------------------------------------------------------------
    # Modelo IS-LM
    # ------------------------------------------------------------------
    @staticmethod
    def plot_is_lm(model, after_model=None, title="Modelo IS-LM"):
        """Curvas IS y LM con puntos de equilibrio (inicial y tras choque)."""
        eq = model.solve()
        Y_range = Visualizer._y_range_around(eq["Y"])
        Y_is, r_is = model.get_is_curve(Y_range)
        Y_lm, r_lm = model.get_lm_curve(Y_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Y_is, y=r_is * 100, mode="lines", name="IS",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="IS<br>Y=%{x:.1f}, r=%{y:.2f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=Y_lm, y=r_lm * 100, mode="lines", name="LM",
            line=dict(color="#d62728", width=3),
            hovertemplate="LM<br>Y=%{x:.1f}, r=%{y:.2f}%<extra></extra>",
        ))
        fig.add_trace(Visualizer._equilibrium_trace(
            {"x": eq["Y"], "y": eq["r"] * 100},
            f"Equilibrio inicial<br>Y*={eq['Y']:.1f}, r*={eq['r']*100:.2f}%",
            color="#2ca02c", symbol="star"))

        if after_model is not None:
            aeq = after_model.solve()
            aY, ar = after_model.get_is_curve(Y_range)
            aY2, ar2 = after_model.get_lm_curve(Y_range)
            fig.add_trace(go.Scatter(
                x=aY, y=ar * 100, mode="lines", name="IS' (tras choque)",
                line=dict(color="#1f77b4", width=2, dash="dash")))
            fig.add_trace(go.Scatter(
                x=aY2, y=ar2 * 100, mode="lines", name="LM' (tras choque)",
                line=dict(color="#d62728", width=2, dash="dash")))
            fig.add_trace(Visualizer._equilibrium_trace(
                {"x": aeq["Y"], "y": aeq["r"] * 100},
                f"Nuevo equilibrio<br>Y*={aeq['Y']:.1f}, r*={aeq['r']*100:.2f}%",
                color="#ff7f0e", symbol="star"))

        # Región visible con un pequeño margen alrededor del equilibrio
        y_max = max(float(np.nanmax(r_is) * 100), float(np.nanmax(r_lm) * 100),
                    1.0) * 1.15
        fig.update_layout(**Visualizer._layout(
            title, "Producción (Y)", "Tasa de interés (r, %)"))
        fig.update_yaxes(range=[0, y_max])
        fig.update_xaxes(range=[float(np.nanmin(Y_range)),
                                float(np.nanmax(Y_range))])
        return fig

    # ------------------------------------------------------------------
    # Modelo AD-AS
    # ------------------------------------------------------------------
    @staticmethod
    def plot_ad_as(model, after_model=None, title="Modelo AD-AS"):
        """Curvas AD, SRAS y LRAS con equilibrios de corto y largo plazo."""
        eq = model.solve()
        lr = model.long_run_equilibrium()
        Y_range = np.linspace(max(0.5 * eq["Yn"], 1.0),
                              min(1.8 * eq["Yn"], 4.0 * eq["Yn"]), 120)
        Y_ad, P_ad = model.get_ad_curve(Y_range)
        Y_sr, P_sr = model.get_sras_curve(Y_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Y_ad, y=P_ad, mode="lines", name="AD",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="AD<br>Y=%{x:.1f}, P=%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=Y_sr, y=P_sr, mode="lines", name="SRAS",
            line=dict(color="#d62728", width=3),
            hovertemplate="SRAS<br>Y=%{x:.1f}, P=%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[eq["Yn"], eq["Yn"]], y=[0, max(float(P_ad.max()) * 1.05, 1.0)],
            mode="lines", name="LRAS (Yₙ)",
            line=dict(color="#2ca02c", width=3, dash="dot")))

        fig.add_trace(Visualizer._equilibrium_trace(
            {"x": eq["Y"], "y": eq["P"]},
            f"Equilibrio corto plazo<br>Y={eq['Y']:.1f}, P={eq['P']:.2f}",
            color="#2ca02c", symbol="star"))
        fig.add_trace(Visualizer._equilibrium_trace(
            {"x": lr["Y"], "y": lr["P"]},
            f"Equilibrio largo plazo<br>Yₙ={lr['Y']:.1f}, P*={lr['P']:.2f}",
            color="#9467bd", symbol="diamond", size=13))

        if after_model is not None:
            aeq = after_model.solve()
            aY, aP = after_model.get_ad_curve(Y_range)
            aY2, aP2 = after_model.get_sras_curve(Y_range)
            fig.add_trace(go.Scatter(
                x=aY, y=aP, mode="lines", name="AD' (tras choque)",
                line=dict(color="#1f77b4", width=2, dash="dash")))
            fig.add_trace(go.Scatter(
                x=aY2, y=aP2, mode="lines", name="SRAS' (tras choque)",
                line=dict(color="#d62728", width=2, dash="dash")))
            fig.add_trace(Visualizer._equilibrium_trace(
                {"x": aeq["Y"], "y": aeq["P"]},
                f"Nuevo equilibrio<br>Y={aeq['Y']:.1f}, P={aeq['P']:.2f}",
                color="#ff7f0e", symbol="star"))

        fig.update_layout(**Visualizer._layout(
            title, "Producción (Y)", "Nivel de precios (P)"))
        fig.update_xaxes(range=[float(Y_range[0]), float(Y_range[-1])])
        return fig

    # ------------------------------------------------------------------
    # Modelo Mundell-Fleming
    # ------------------------------------------------------------------
    @staticmethod
    def plot_mundell_fleming(model, after_model=None,
                             title="Modelo Mundell-Fleming"):
        """Curvas IS*, LM* y BP en el plano (Y, e)."""
        eq = model.solve()
        Y_range = Visualizer._y_range_around(eq["Y"])
        Y_is, e_is = model.get_is_curve(Y_range)
        Y_lm, e_lm = model.get_lm_curve(Y_range)
        Y_bp, e_bp = model.get_bp_curve(Y_range)

        regime = model.params.get("regime", "Flexible")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Y_is, y=e_is, mode="lines", name="IS*",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="IS*<br>Y=%{x:.1f}, e=%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=Y_lm, y=e_lm, mode="lines", name="LM*",
            line=dict(color="#d62728", width=3),
            hovertemplate="LM*<br>Y=%{x:.1f}, e=%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=Y_bp, y=e_bp, mode="lines", name="BP",
            line=dict(color="#2ca02c", width=3, dash="dot")))

        if regime.lower() == "fijo":
            fig.add_trace(go.Scatter(
                x=Y_range, y=[eq["e_bar"]] * len(Y_range), mode="lines",
                name="Ancla (e_bar)",
                line=dict(color="#333333", width=3, dash="longdash")))

        fig.add_trace(Visualizer._equilibrium_trace(
            {"x": eq["Y"], "y": eq["e"]},
            f"Equilibrio<br>Y*={eq['Y']:.1f}, e*={eq['e']:.2f}",
            color="#2ca02c", symbol="star"))

        if after_model is not None:
            aeq = after_model.solve()
            aY, ae = after_model.get_is_curve(Y_range)
            aY2, ae2 = after_model.get_lm_curve(Y_range)
            fig.add_trace(go.Scatter(
                x=aY, y=ae, mode="lines", name="IS*' (tras choque)",
                line=dict(color="#1f77b4", width=2, dash="dash")))
            fig.add_trace(go.Scatter(
                x=aY2, y=ae2, mode="lines", name="LM*' (tras choque)",
                line=dict(color="#d62728", width=2, dash="dash")))
            fig.add_trace(Visualizer._equilibrium_trace(
                {"x": aeq["Y"], "y": aeq["e"]},
                f"Nuevo equilibrio<br>Y*={aeq['Y']:.1f}, e*={aeq['e']:.2f}",
                color="#ff7f0e", symbol="star"))

        fig.update_layout(**Visualizer._layout(
            title, "Producción (Y)", "Tipo de cambio (e)"))
        fig.update_xaxes(range=[float(np.nanmin(Y_range)),
                                float(np.nanmax(Y_range))])
        return fig

    # ------------------------------------------------------------------
    # Análisis de sensibilidad
    # ------------------------------------------------------------------
    @staticmethod
    def plot_sensitivity_analysis(x_values, y_values, x_label, y_label,
                                  title="Análisis de sensibilidad"):
        """Gráfico de líneas para resultados en función de un parámetro."""
        fig = go.Figure(go.Scatter(
            x=x_values, y=y_values, mode="lines+markers",
            name="Resultado", line=dict(color="#1f77b4", width=3),
            marker=dict(size=7)))
        fig.update_layout(**Visualizer._layout(title, x_label, y_label))
        return fig

    # ------------------------------------------------------------------
    # Series temporales
    # ------------------------------------------------------------------
    @staticmethod
    def plot_series_interactive(data, variables, title="Series temporales"):
        """Gráfico de líneas con múltiples variables y eje secundario.

        Parámetros
        ----------
        data : pandas.DataFrame
            Debe contener una columna ``fecha``.
        variables : list[str]
            Variables a graficar en el eje principal.
        title : str
            Título del gráfico.
        """
        fig = go.Figure()
        for var in variables:
            if var not in data.columns:
                continue
            fig.add_trace(go.Scatter(
                x=data["fecha"], y=data[var], mode="lines",
                name=VARIABLE_LABELS.get(var, var),
                line=dict(color=VARIABLE_COLORS.get(var, "#1f77b4"), width=2)))
        fig.update_layout(**Visualizer._layout(
            title, "Fecha", "Valor", height=480))
        return fig

    @staticmethod
    def plot_series_dual(data, primary, secondary):
        """Gráfico con eje secundario (y2) para dos series de distinta escala."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data[primary], mode="lines", name=primary,
            line=dict(color=VARIABLE_COLORS.get(primary, "#1f77b4"), width=2.5)))
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data[secondary], mode="lines", name=secondary,
            line=dict(color=VARIABLE_COLORS.get(secondary, "#d62728"),
                      width=2.5),
            yaxis="y2"))
        fig.update_layout(
            **Visualizer._layout(f"{primary} vs {secondary}", "Fecha",
                                 primary, height=480))
        fig.update_layout(yaxis2=dict(
            title=secondary, overlaying="y", side="right",
            showgrid=False))
        return fig

    @staticmethod
    def plot_macro_dashboard(data):
        """Dashboard 3x2 con las series macroeconómicas principales."""
        n = len(data)
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "PIB vs PIB potencial", "Inflación (%)",
                "Tasa de interés (%)", "Desempleo (%)",
                "Tipo de cambio", "Brecha del producto (%)"])

        # (1,1) PIB vs potencial
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["PIB"], name="PIB",
            line=dict(color="#1f77b4", width=2)), row=1, col=1)
        if "PIB_Potencial" in data.columns:
            fig.add_trace(go.Scatter(
                x=data["fecha"], y=data["PIB_Potencial"], name="PIB potencial",
                line=dict(color="#bcbd22", width=2, dash="dash")), row=1, col=1)

        # (1,2) Inflación
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["Inflacion"], name="Inflación",
            line=dict(color="#d62728", width=2)), row=1, col=2)

        # (2,1) Tasa de interés
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["Tasa_Interes"], name="Tasa de interés",
            line=dict(color="#2ca02c", width=2)), row=2, col=1)

        # (2,2) Desempleo
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["Desempleo"], name="Desempleo",
            line=dict(color="#ff7f0e", width=2)), row=2, col=2)

        # (3,1) Tipo de cambio
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["Tipo_Cambio"], name="Tipo de cambio",
            line=dict(color="#9467bd", width=2)), row=3, col=1)

        # (3,2) Brecha del producto
        from data_manager import DataManager
        gap = DataManager.output_gap(data)
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=gap, name="Brecha (%)",
            line=dict(color="#000000", width=2)), row=3, col=2)
        fig.add_hline(y=0, line=dict(color="#999999", dash="dot"),
                      row=3, col=2)

        fig.update_layout(
            template="plotly_white", height=720,
            title=dict(text="Dashboard macroeconómico", x=0.5),
            margin=dict(l=50, r=30, t=70, b=40),
            showlegend=False)
        return fig

    @staticmethod
    def plot_output_gap(data):
        """PIB, potencial y brecha del producto en un solo gráfico."""
        from data_manager import DataManager
        trend = DataManager.hp_trend(data, "PIB")
        gap = DataManager.output_gap(data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["PIB"], name="PIB",
            line=dict(color="#1f77b4", width=2.5)))
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend.values, name="PIB potencial (HP)",
            line=dict(color="#2ca02c", width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=gap, name="Brecha (%)",
            line=dict(color="#ff7f0e", width=2), yaxis="y2",
            fill="tozeroy"))
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1, yref="y2", y0=0, y1=0,
            line=dict(color="#999999", dash="dot"))
        fig.update_layout(
            **Visualizer._layout("Brecha del producto", "Fecha", "PIB", 520))
        fig.update_layout(yaxis2=dict(
            title="Brecha (%)", overlaying="y", side="right",
            showgrid=False))
        return fig

    @staticmethod
    def plot_taylor_rule(data, r_neutral=2.5, pi_target=3.5):
        """Inflación, tasa vigente y tasa recomendada por la regla de Taylor."""
        from data_manager import DataManager
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["Inflacion"], name="Inflación (%)",
            line=dict(color="#d62728", width=2)))
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=data["Tasa_Interes"], name="Tasa de interés (%)",
            line=dict(color="#2ca02c", width=2)))
        taylor = [DataManager.taylor_recommendation(
            data.iloc[: i + 1], r_neutral=r_neutral,
            pi_target=pi_target)[0] for i in range(len(data))]
        fig.add_trace(go.Scatter(
            x=data["fecha"], y=taylor, name="Regla de Taylor (%)",
            line=dict(color="#9467bd", width=2.5, dash="dot")))
        fig.update_layout(**Visualizer._layout(
            "Regla de Taylor", "Fecha", "Porcentaje (%)", 480))
        return fig

    @staticmethod
    def plot_business_cycle(data):
        """Ciclo económico (desviación del PIB respecto a la tendencia HP)."""
        from data_manager import DataManager
        cycle_df = DataManager.cyclical_analysis(data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycle_df["fecha"], y=cycle_df["Ciclo"], name="Ciclo",
            line=dict(color="#1f77b4", width=2.5), fill="tozeroy"))
        fig.add_hline(y=0, line=dict(color="#999999", dash="dot"))
        fig.update_layout(**Visualizer._layout(
            "Análisis del ciclo económico (filtro HP)", "Fecha",
            "Componente cíclico del PIB", 480))
        return fig
