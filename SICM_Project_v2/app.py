"""Aplicación principal del Simulador Integral de Choques Macroeconómicos (SICM) v2.0.

Ejecutar con:  streamlit run app.py

Estructura: navegación lateral + páginas para modelos IS-LM/AD-AS,
Mundell-Fleming, visualización de choques, series temporales, simulación
de políticas, escenarios, reportes PDF y dashboard de laboratorio.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from models import (ISLMModel, ADASModel, MundellFlemingModel, build_model,
                    SHOCK_CATALOG, SHOCK_KEYS, get_shock_mechanism)
from data_manager import DataManager, REQUIRED_COLUMNS
from visualizer import Visualizer
from policy import (simulate_policy, sensitivity_analysis,
                    policy_result_metrics, POLICY_NAMES)
from scenarios import ScenarioManager
from reports import generate_pdf_report

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SICM v2.0 · Simulador de Choques Macroeconómicos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")


def inject_css():
    """Inyecta estilos CSS ligeros para una interfaz más limpia."""
    st.markdown("""
    <style>
      .block-container { padding-top: 2rem; }
      .kpi-card {
        background: linear-gradient(135deg, #1f3a5f, #2c5a8f);
        border-radius: 12px; padding: 16px 18px; color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      }
      .kpi-card .kpi-label { font-size: 0.8rem; opacity: 0.85; }
      .kpi-card .kpi-value { font-size: 1.6rem; font-weight: 700; }
      .kpi-card .kpi-delta { font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
def init_session_state():
    """Inicializa las variables persistentes de la sesión."""
    defaults = {
        "data": None,               # DataFrame de series
        "data_source": "Sintético",
        "scenario_manager": None,
        "shock_before": None,       # equilibrio previo (choques)
        "shock_after": None,
        "shock_model": None,        # modelo usado en el módulo de choques
        "shock_params": None,       # parámetros usados en el último choque
        "shock_label": None,
        "shock_applied_magnitude": None,  # magnitud aplicada (último choque)
        "policy_result": None,       # último resultado de política
        "report_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.scenario_manager is None:
        st.session_state.scenario_manager = ScenarioManager()


# ---------------------------------------------------------------------------
# Parámetros por defecto (persistidos en el estado de sesión)
# ---------------------------------------------------------------------------
ISLM_PARAM_RANGES = {
    "C0": (20.0, 100.0, 50.0), "c": (0.3, 0.95, 0.75),
    "I0": (50.0, 150.0, 100.0), "b": (0.1, 0.8, 0.4),
    "G": (50.0, 200.0, 120.0), "T": (20.0, 200.0, 80.0),
    "M": (100.0, 400.0, 200.0), "k": (0.1, 1.5, 0.5),
    "h": (500.0, 3000.0, 1500.0),
}

ADAS_PARAM_RANGES = {
    "M": (100.0, 400.0, 200.0), "V": (3.0, 8.0, 5.0),
    "Yn": (80.0, 150.0, 100.0), "lambda": (0.01, 0.10, 0.05),
    "Pe_factor": (0.5, 2.0, 1.0),
}

MF_PARAM_RANGES = {
    "C0": (20.0, 100.0, 50.0), "c": (0.3, 0.95, 0.75),
    "I0": (50.0, 150.0, 100.0), "b": (0.1, 0.8, 0.4),
    "G": (50.0, 200.0, 120.0), "T": (20.0, 200.0, 80.0),
    "M": (100.0, 400.0, 200.0), "k": (0.1, 1.5, 0.5),
    "h": (500.0, 3000.0, 1500.0),
    "NX0": (0.0, 200.0, 30.0), "theta": (0.1, 2.0, 0.5),
}


def _param(key_prefix, label, ranges, help_text=""):
    """Crea un slider de parámetro persistido con key única."""
    lo, hi, default = ranges
    return st.slider(
        label, lo, hi, default, step=(hi - lo) / 200.0,
        key=f"{key_prefix}_{label.split(' ')[0].replace('(', '').replace(')', '')}",
        help=help_text)


def islm_params_from_state(prefix="islm"):
    """Reconstruye los parámetros IS-LM desde el estado de sesión."""
    params = {}
    for key, (lo, hi, default) in ISLM_PARAM_RANGES.items():
        params[key] = st.session_state.get(f"{prefix}_{key}", default)
    params["P"] = 1.0
    return params


def adas_params_from_state(prefix="adas"):
    """Reconstruye los parámetros AD-AS desde el estado de sesión."""
    params = {}
    for key, (lo, hi, default) in ADAS_PARAM_RANGES.items():
        params[key] = st.session_state.get(f"{prefix}_{key}", default)
    return params


def mf_params_from_state(prefix="mf"):
    """Reconstruye los parámetros Mundell-Fleming desde el estado de sesión."""
    params = {}
    for key, (lo, hi, default) in MF_PARAM_RANGES.items():
        params[key] = st.session_state.get(f"{prefix}_{key}", default)
    params["P"] = 1.0
    params["r_star"] = st.session_state.get(f"{prefix}_r_star", 0.05)
    params["r_w"] = st.session_state.get(f"{prefix}_r_w", 0.05)
    params["kappa"] = st.session_state.get(f"{prefix}_kappa", 1e9)
    params["regime"] = st.session_state.get(f"{prefix}_regime", "Flexible")
    params["e_bar"] = st.session_state.get(f"{prefix}_e_bar", 200.0)
    return params


def reset_params(prefix, ranges):
    """Restablece los parámetros de un modelo a sus valores por defecto."""
    for key, (lo, hi, default) in ranges.items():
        st.session_state[f"{prefix}_{key}"] = default


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------
def page_inicio():
    """Página de bienvenida y descripción general."""
    st.header("📈 Bienvenido al Laboratorio Macroeconómico SICM v2.0")
    st.markdown(
        "Plataforma de investigación y docencia que integra **modelos "
        "macroeconómicos**, **visualizaciones interactivas**, **análisis de "
        "datos reales** y **simulación de políticas**.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Modelos", "3", "IS-LM, AD-AS, Mundell-Fleming")
    c2.metric("Choques", "18", "Fiscales, monetarios y externos")
    c3.metric("Herramientas", "10", "Simulación, datos y reportes")

    st.divider()
    st.subheader("Módulos disponibles")
    modulos = {
        "🟦 Modelos macroeconómicos": (
            "Resolución numérica del equilibrio en IS-LM, AD-AS y "
            "Mundell-Fleming con parámetros configurables."),
        "⚡ Visualización de choques": (
            "Selector de choques (fiscal, monetario, externo) con magnitud, "
            "mecanismo de transmisión y comparación antes/después."),
        "📊 Series temporales": (
            "Datos sintéticos realistas o importación CSV/Excel con "
            "dashboard 3x2, estadísticas y filtro HP."),
        "🛠️ Simulación de políticas": (
            "Políticas fiscales y monetarias con análisis de sensibilidad."),
        "💾 Escenarios": (
            "Guardar, cargar y comparar escenarios de simulación."),
        "📄 Reportes PDF": (
            "Informes profesionales con ReportLab."),
        "🏥 Dashboard laboratorio": (
            "Indicadores clave, brecha del producto, regla de Taylor y ciclo."),
    }
    for title, desc in modulos.items():
        st.markdown(f"**{title}**  \n{desc}")

    with st.expander("ℹ️ Cómo empezar"):
        st.markdown(
            "1. Explore los **modelos** y ajuste los parámetros con los "
            "deslizadores.  \n"
            "2. Aplique **choques** y observe el nuevo equilibrio.  \n"
            "3. Cargue o genere **series** y analice indicadores.  \n"
            "4. **Simule políticas**, guarde escenarios y genere su **reporte PDF**.")


def _render_islm_page():
    """Contenido de la pestaña IS-LM."""
    st.subheader("Modelo IS-LM")

    with st.sidebar.expander("⚙️ Parámetros IS-LM", expanded=True):
        c = _param("islm", "c", ISLM_PARAM_RANGES["c"],
                   "Propensión marginal a consumir: fracción del ingreso "
                   "disponible destinada al consumo.")
        G = _param("islm", "G", ISLM_PARAM_RANGES["G"],
                   "Gasto del gobierno (política fiscal).")
        M = _param("islm", "M", ISLM_PARAM_RANGES["M"],
                   "Oferta monetaria nominal (política monetaria).")
        b = _param("islm", "b", ISLM_PARAM_RANGES["b"],
                   "Sensibilidad de la inversión a la tasa de interés.")
        with st.expander("Más parámetros"):
            C0 = _param("islm", "C0", ISLM_PARAM_RANGES["C0"], "Consumo autónomo.")
            I0 = _param("islm", "I0", ISLM_PARAM_RANGES["I0"], "Inversión autónoma.")
            T = _param("islm", "T", ISLM_PARAM_RANGES["T"], "Impuestos.")
            k = _param("islm", "k", ISLM_PARAM_RANGES["k"],
                       "Sensibilidad de la demanda de dinero al ingreso.")
            h = _param("islm", "h", ISLM_PARAM_RANGES["h"],
                       "Sensibilidad de la demanda de dinero a la tasa.")
        if st.button("Restablecer parámetros", key="islm_reset"):
            reset_params("islm", ISLM_PARAM_RANGES)
            st.rerun()

    model = ISLMModel(islm_params_from_state())
    eq = model.solve()
    mult = model.multipliers()

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        fig = Visualizer.plot_is_lm(model)
        st.plotly_chart(fig, width="stretch", key="islm_plot")

    with col_info:
        st.markdown("#### Equilibrio general")
        st.metric("Producción (Y*)", f"{eq['Y']:.2f}")
        st.metric("Tasa de interés (r*)", f"{eq['r'] * 100:.2f}%")
        st.metric("Consumo (C)", f"{eq['C']:.2f}")
        st.metric("Inversión (I)", f"{eq['I']:.2f}")
        st.divider()
        st.markdown("#### Multiplicadores")
        st.metric("Fiscal (ΔY/ΔG)", f"{mult['dY_dG']:.3f}")
        st.metric("Monetario (ΔY/ΔM)", f"{mult['dY_dM']:.4f}")

    with st.expander("📐 Ecuaciones del modelo"):
        st.latex(r"C = C_0 + c\,(Y - T)")
        st.latex(r"I = I_0 - 100\,b\,r")
        st.latex(r"Y = C + I + G \quad (IS)")
        st.latex(r"\frac{M}{P} = k\,Y - h\,r \quad (LM)")


def _render_adas_page():
    """Contenido de la pestaña AD-AS."""
    st.subheader("Modelo AD-AS")

    with st.sidebar.expander("⚙️ Parámetros AD-AS", expanded=True):
        M = _param("adas", "M", ADAS_PARAM_RANGES["M"],
                   "Oferta monetaria nominal.")
        V = _param("adas", "V", ADAS_PARAM_RANGES["V"],
                   "Velocidad del dinero.")
        Yn = _param("adas", "Yn", ADAS_PARAM_RANGES["Yn"],
                    "Producción natural (pleno empleo).")
        lam = _param("adas", "lambda", ADAS_PARAM_RANGES["lambda"],
                     "Pendiente de la curva de oferta de corto plazo.")
        Pe = _param("adas", "Pe_factor", ADAS_PARAM_RANGES["Pe_factor"],
                    "Multiplicador de precios esperados sobre el ancla "
                    f"M·V/Yn = {ADASModel.BASE_PE:.2f}.")
        if st.button("Restablecer parámetros", key="adas_reset"):
            reset_params("adas", ADAS_PARAM_RANGES)
            st.rerun()

    model = ADASModel(adas_params_from_state())
    eq = model.solve()
    lr = model.long_run_equilibrium()

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        fig = Visualizer.plot_ad_as(model)
        st.plotly_chart(fig, width="stretch", key="adas_plot")

    with col_info:
        st.markdown("#### Equilibrio de corto plazo")
        st.metric("Producción (Y)", f"{eq['Y']:.2f}")
        st.metric("Nivel de precios (P)", f"{eq['P']:.3f}")
        st.metric("Brecha del producto", f"{eq['gap']:+.2f}%")
        st.divider()
        st.markdown("#### Equilibrio de largo plazo")
        st.metric("Producción natural (Yₙ)", f"{lr['Y']:.2f}")
        st.metric("Precios de largo plazo (P*)", f"{lr['P']:.3f}")

    with st.expander("📐 Ecuaciones del modelo"):
        st.latex(r"AD: \quad Y = \frac{M\,V}{P}")
        st.latex(r"SRAS: \quad P = P_e\,[\,1 + \lambda\,(Y - Y_n)\,]")
        st.latex(r"LRAS: \quad Y = Y_n")


def page_models():
    """Página con los modelos IS-LM y AD-AS en pestañas."""
    st.header("Modelos macroeconómicos")
    tab_is, tab_ad = st.tabs(["IS-LM", "AD-AS"])
    with tab_is:
        _render_islm_page()
    with tab_ad:
        _render_adas_page()


def page_mundell_fleming():
    """Página del modelo Mundell-Fleming."""
    st.header("Modelo Mundell-Fleming (economía abierta)")

    with st.sidebar.expander("⚙️ Parámetros Mundell-Fleming", expanded=True):
        regime = st.selectbox(
            "Régimen cambiario", ["Flexible", "Fijo"],
            key="mf_regime",
            help="Con tipo de cambio fijo, la oferta monetaria se vuelve "
                 "endógena (el banco central defiende el ancla).")
        mobility = st.select_slider(
            "Movilidad de capitales", ["Nula", "Imperfecta", "Perfecta"],
            value="Perfecta", key="mf_mobility",
            help="Determina la sensibilidad de los flujos de capital a la "
                 "diferencia de tasas r* - r_w.")
        kappa_map = {"Nula": 0.0, "Imperfecta": 1.0, "Perfecta": 1e9}
        st.session_state["mf_kappa"] = kappa_map[mobility]

        r_star = st.slider("Tasa mundial (r*)", 0.01, 0.08, 0.05, 0.001,
                           key="mf_r_star")
        r_w = st.slider("Tasa mundial de la BP (r_w)", 0.01, 0.08, 0.05, 0.001,
                        key="mf_r_w")

        # Ancla cambiaria (solo régimen fijo)
        flex_params = mf_params_from_state()
        flex_params["regime"] = "Flexible"
        flexible_eq = MundellFlemingModel(flex_params).solve()
        default_e = float(flexible_eq["e"])
        if regime == "Fijo":
            if "mf_e_bar" not in st.session_state:
                st.session_state["mf_e_bar"] = default_e
            e_bar = st.slider("Ancla cambiaria (e_bar)", 50.0, 400.0, None, 1.0,
                              key="mf_e_bar",
                              help="Tipo de cambio que el banco central defiende.")
            st.info("Régimen fijo: la oferta monetaria (M) se ajusta "
                    "endógenamente para sostener el ancla cambiaria.")

        with st.expander("Parámetros de la economía"):
            _param("mf", "c", MF_PARAM_RANGES["c"], "Propensión marginal a consumir.")
            _param("mf", "G", MF_PARAM_RANGES["G"], "Gasto del gobierno.")
            _param("mf", "M", MF_PARAM_RANGES["M"],
                   "Oferta monetaria (exógena solo en régimen flexible).")
            _param("mf", "NX0", MF_PARAM_RANGES["NX0"],
                   "Exportaciones netas autónomas.")
            _param("mf", "theta", MF_PARAM_RANGES["theta"],
                   "Sensibilidad de las exportaciones netas al tipo de cambio.")
            with st.expander("Más parámetros"):
                _param("mf", "C0", MF_PARAM_RANGES["C0"], "Consumo autónomo.")
                _param("mf", "I0", MF_PARAM_RANGES["I0"], "Inversión autónoma.")
                _param("mf", "b", MF_PARAM_RANGES["b"],
                       "Sensibilidad de la inversión a la tasa.")
                _param("mf", "T", MF_PARAM_RANGES["T"], "Impuestos.")
                _param("mf", "k", MF_PARAM_RANGES["k"],
                       "Sensibilidad de la demanda de dinero al ingreso.")
                _param("mf", "h", MF_PARAM_RANGES["h"],
                       "Sensibilidad de la demanda de dinero a la tasa.")
        if st.button("Restablecer parámetros", key="mf_reset"):
            reset_params("mf", MF_PARAM_RANGES)
            st.rerun()

    model = MundellFlemingModel(mf_params_from_state())
    eq = model.solve()

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        fig = Visualizer.plot_mundell_fleming(model)
        st.plotly_chart(fig, width="stretch", key="mf_plot")

    with col_info:
        st.markdown(f"#### Equilibrio ({eq['regime']})")
        st.metric("Producción (Y*)", f"{eq['Y']:.2f}")
        st.metric("Tipo de cambio (e*)", f"{eq['e']:.2f}")
        st.metric("Oferta monetaria (M)",
                  f"{eq['M']:.2f}" + ("  (endógena)" if regime == "Fijo" else ""))
        st.divider()
        st.metric("Exportaciones netas (NX)", f"{eq['NX']:.2f}")
        st.metric("Balance de pagos (BP)", f"{eq['BP']:+.2f}",
                  help="NX + CF. Si es distinto de 0, la economía acumula "
                       "(+) o pierde (−) reservas.")

    with st.expander("📐 Ecuaciones del modelo"):
        st.latex(r"IS^*: \quad Y = C(Y-T) + I(r^*) + G + NX(e)")
        st.latex(r"LM^*: \quad \frac{M}{P} = L(Y, r^*)")
        st.latex(r"BP: \quad NX(e) + CF(r^* - r_w) = 0")
        st.markdown("Con $NX(e) = NX_0 - \\theta\\,e$ y $CF = \\kappa\\,(r^* - r_w)$.")


def page_shocks():
    """Visualización de choques con botón de actualización manual."""
    st.header("Visualización de choques macroeconómicos")

    with st.sidebar.expander("⚙️ Configuración del choque", expanded=True):
        model_name = st.selectbox("Modelo base", ["IS-LM", "AD-AS", "Mundell-Fleming"],
                                  key="shock_model_name")
        shock_options = SHOCK_KEYS[model_name]
        shock_label = st.selectbox("Tipo de choque", shock_options,
                                   key="shock_type")
        magnitude = st.slider("Magnitud", 0.05, 0.30, 0.10, 0.01,
                              key="shock_magnitude",
                              help="Proporción del cambio aplicado al "
                                   "parámetro (0.10 = 10 %).")

    # Parámetros base del modelo seleccionado
    if model_name == "IS-LM":
        base_params = islm_params_from_state(prefix="shock")
    elif model_name == "AD-AS":
        base_params = adas_params_from_state(prefix="shock")
    else:
        base_params = mf_params_from_state(prefix="shock")

    with st.sidebar.expander("🔧 Parámetros del modelo base", expanded=True):
        if model_name == "IS-LM":
            _param("shock", "c", ISLM_PARAM_RANGES["c"], "Propensión marginal a consumir.")
            _param("shock", "G", ISLM_PARAM_RANGES["G"], "Gasto del gobierno.")
            _param("shock", "M", ISLM_PARAM_RANGES["M"], "Oferta monetaria.")
            _param("shock", "b", ISLM_PARAM_RANGES["b"],
                   "Sensibilidad de la inversión.")
        elif model_name == "AD-AS":
            _param("shock", "M", ADAS_PARAM_RANGES["M"], "Oferta monetaria.")
            _param("shock", "V", ADAS_PARAM_RANGES["V"], "Velocidad del dinero.")
            _param("shock", "Yn", ADAS_PARAM_RANGES["Yn"], "Producción natural.")
            _param("shock", "lambda", ADAS_PARAM_RANGES["lambda"],
                   "Pendiente de la SRAS.")
        else:
            _param("shock", "G", MF_PARAM_RANGES["G"], "Gasto del gobierno.")
            _param("shock", "M", MF_PARAM_RANGES["M"], "Oferta monetaria.")

    # Botón de actualización manual (sin sliders automáticos)
    if st.button("🔄 Actualizar gráfico", key="shock_update"):
        model = build_model(model_name, dict(base_params))
        before = model.solve()
        model_after = model.clone()
        after = model_after.apply_shock(shock_label, magnitude)
        st.session_state.shock_before = before
        st.session_state.shock_after = after
        st.session_state.shock_model = model_name
        st.session_state.shock_label = shock_label
        st.session_state.shock_applied_magnitude = magnitude
        st.session_state.shock_params = dict(base_params)
        st.success("Choque aplicado. Resultados actualizados.")

    # Renderizado del gráfico (solo con resultados calculados)
    before = st.session_state.shock_before
    after = st.session_state.shock_after
    applied_model = st.session_state.shock_model
    applied_params = st.session_state.shock_params

    if before is None:
        st.info("Configure el choque en el panel lateral y pulse "
                "**Actualizar gráfico** para visualizar el efecto.")
        return

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        base = build_model(applied_model, dict(applied_params))
        after_model = build_model(applied_model, dict(applied_params))
        after_model.apply_shock(st.session_state.shock_label,
                                st.session_state.shock_applied_magnitude)
        if applied_model == "IS-LM":
            fig = Visualizer.plot_is_lm(base, after_model,
                                        title=f"Choque: {st.session_state.shock_label}")
        elif applied_model == "AD-AS":
            fig = Visualizer.plot_ad_as(base, after_model,
                                        title=f"Choque: {st.session_state.shock_label}")
        else:
            fig = Visualizer.plot_mundell_fleming(
                base, after_model,
                title=f"Choque: {st.session_state.shock_label}")
        st.plotly_chart(fig, width="stretch", key="shock_plot")

    with col_info:
        st.markdown(f"#### Choque aplicado")
        st.markdown(f"**{st.session_state.shock_label}** · "
                    f"magnitud {st.session_state.shock_applied_magnitude * 100:.0f}%")
        st.divider()
        if applied_model == "IS-LM":
            st.metric("Y inicial", f"{before['Y']:.2f}")
            st.metric("Y final", f"{after['Y']:.2f}",
                      delta=f"{after['Y'] - before['Y']:+.2f}")
            st.metric("r inicial", f"{before['r'] * 100:.2f}%")
            st.metric("r final", f"{after['r'] * 100:.2f}%",
                      delta=f"{(after['r'] - before['r']) * 100:+.2f} p.p.")
        elif applied_model == "AD-AS":
            st.metric("Y inicial", f"{before['Y']:.2f}")
            st.metric("Y final", f"{after['Y']:.2f}",
                      delta=f"{after['Y'] - before['Y']:+.2f}")
            st.metric("P inicial", f"{before['P']:.3f}")
            st.metric("P final", f"{after['P']:.3f}",
                      delta=f"{after['P'] - before['P']:+.3f}")
        else:
            st.metric("Y inicial", f"{before['Y']:.2f}")
            st.metric("Y final", f"{after['Y']:.2f}",
                      delta=f"{after['Y'] - before['Y']:+.2f}")
            st.metric("e inicial", f"{before['e']:.2f}")
            st.metric("e final", f"{after['e']:.2f}",
                      delta=f"{after['e'] - before['e']:+.2f}")

    mechanism = get_shock_mechanism(applied_model, st.session_state.shock_label)
    with st.expander("💡 Mecanismo de transmisión"):
        st.markdown(mechanism)


def page_series():
    """Series temporales: generación, importación y análisis."""
    st.header("Series temporales macroeconómicas")

    # Fuente de datos
    source = st.radio(
        "Fuente de datos",
        ["Datos sintéticos", "Importar archivo (CSV/Excel)"],
        horizontal=True, key="series_source")

    if source == "Datos sintéticos":
        if st.button("Generar datos sintéticos", key="series_generate",
                     help="Genera 240 observaciones mensuales realistas "
                          "(tendencia + ciclo + ruido)."):
            df = DataManager.generate_sample_data()
            st.session_state.data = df
            st.session_state.data_source = "Sintético"
            st.success("Datos sintéticos generados (2000-01 a 2019-12).")
    else:
        uploaded = st.file_uploader(
            "Subir archivo (CSV o Excel)", type=["csv", "xlsx"],
            key="series_upload",
            help="Columnas esperadas: fecha, PIB, Inflacion, Tasa_Interes, "
                 "Desempleo, Tipo_Cambio.")
        if uploaded is not None:
            try:
                df = DataManager.load_data(uploaded, uploaded.name)
                st.session_state.data = df
                st.session_state.data_source = uploaded.name
                st.success(f"Datos cargados: {len(df)} observaciones.")
            except ValueError as exc:
                st.error(f"⚠️ {exc}")
                st.markdown(
                    f"Formato esperado: columnas "
                    f"`{', '.join(REQUIRED_COLUMNS)}`.")

    data = st.session_state.data
    if data is None:
        st.info("Seleccione una fuente y genere o cargue los datos para "
                "comenzar el análisis.")
        return

    st.markdown(f"**Fuente activa:** {st.session_state.data_source} · "
                f"**{len(data)}** observaciones · "
                f"**{data['fecha'].iloc[0].date()}** a "
                f"**{data['fecha'].iloc[-1].date()}**")

    with st.expander("🔍 Vista previa de los datos"):
        st.dataframe(data.head(10), width="stretch")

    tab_dash, tab_series, tab_stats = st.tabs(
        ["Dashboard 3×2", "Series interactivas", "Estadísticas"])

    with tab_dash:
        fig = Visualizer.plot_macro_dashboard(data)
        st.plotly_chart(fig, width="stretch", key="series_dash")

    with tab_series:
        vars_ = [c for c in REQUIRED_COLUMNS[1:] if c in data.columns]
        selected = st.multiselect("Variables", vars_, default=vars_,
                                  key="series_multi")
        if selected:
            fig = Visualizer.plot_series_interactive(data, selected)
            st.plotly_chart(fig, width="stretch", key="series_lines")

        st.markdown("**Ejes secundarios (y2):**")
        c1, c2 = st.columns(2)
        with c1:
            p = st.selectbox("Variable eje principal", vars_, key="series_primary")
        with c2:
            s = st.selectbox("Variable eje secundario", vars_,
                             key="series_secondary")
        if p != s:
            fig2 = Visualizer.plot_series_dual(data, p, s)
            st.plotly_chart(fig2, width="stretch", key="series_dual")

    with tab_stats:
        st.dataframe(DataManager.descriptive_stats(data),
                     width="stretch")
        st.caption("Crecimiento_anual: variación porcentual del PIB en los "
                   "últimos 12 meses.")


def page_policies():
    """Simulación de políticas económicas."""
    st.header("Simulación de políticas económicas")

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Modelo base", ["IS-LM", "AD-AS"],
                                  key="pol_model")
    with c2:
        policy_type = st.selectbox("Política", POLICY_NAMES, key="pol_type")
    with c3:
        magnitude = st.slider("Magnitud (%)", 5, 30, 10, 1,
                              key="pol_magnitude",
                              help="Cambio porcentual del instrumento.")

    if model_name == "IS-LM":
        base_params = islm_params_from_state(prefix="pol")
        with st.sidebar.expander("⚙️ Parámetros IS-LM (política)", expanded=True):
            _param("pol", "c", ISLM_PARAM_RANGES["c"], "Propensión marginal a consumir.")
            _param("pol", "G", ISLM_PARAM_RANGES["G"], "Gasto del gobierno.")
            _param("pol", "M", ISLM_PARAM_RANGES["M"], "Oferta monetaria.")
            _param("pol", "b", ISLM_PARAM_RANGES["b"],
                   "Sensibilidad de la inversión.")
    else:
        base_params = adas_params_from_state(prefix="pol")
        with st.sidebar.expander("⚙️ Parámetros AD-AS (política)", expanded=True):
            _param("pol", "M", ADAS_PARAM_RANGES["M"], "Oferta monetaria.")
            _param("pol", "V", ADAS_PARAM_RANGES["V"], "Velocidad del dinero.")
            _param("pol", "Yn", ADAS_PARAM_RANGES["Yn"], "Producción natural.")
            _param("pol", "lambda", ADAS_PARAM_RANGES["lambda"],
                   "Pendiente de la SRAS.")

    if st.button("▶️ Ejecutar simulación", key="pol_run"):
        result = simulate_policy(model_name, base_params, policy_type,
                                 magnitude / 100.0)
        if not result["ok"]:
            st.warning(result["message"])
        else:
            st.session_state.policy_result = result
            st.success(result["message"])

    result = st.session_state.policy_result
    if result is None or result.get("model") != model_name:
        st.info("Configure la política y pulse **Ejecutar simulación**.")
        return

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        params_base = dict(result["before_params"])
        base_model = build_model(model_name, params_base)
        after_model = build_model(model_name, params_base)
        after_model.apply_shock(result["shock_label"], result["magnitude"])
        if model_name == "IS-LM":
            fig = Visualizer.plot_is_lm(base_model, after_model,
                                        title=f"{policy_type} ({magnitude}%)")
        else:
            fig = Visualizer.plot_ad_as(base_model, after_model,
                                        title=f"{policy_type} ({magnitude}%)")
        st.plotly_chart(fig, width="stretch", key="pol_plot")

    with col_info:
        rows = policy_result_metrics(result)
        if rows:
            st.markdown("#### Comparación antes / después")
            for label, before, after, delta in rows:
                st.metric(label, f"{after:.2f}", delta=f"{delta:+.2f}",
                          help=f"Antes: {before:.2f}")

    with st.expander("💡 Mecanismo de transmisión"):
        st.markdown(result["mechanism"])

    st.divider()
    st.subheader("Análisis de sensibilidad a la magnitud")
    magnitudes = np.arange(0.05, 0.31, 0.05)
    mags, results = sensitivity_analysis(model_name, base_params,
                                         policy_type, magnitudes)
    valid = [(m, r) for m, r in zip(mags, results) if r is not None]
    if valid:
        x_vals = [m * 100 for m, _ in valid]
        y_vals = [r["Y"] for _, r in valid]
        fig_sens = Visualizer.plot_sensitivity_analysis(
            x_vals, y_vals, "Magnitud (%)",
            "Producción tras el choque (Y)",
            title=f"Sensibilidad: {policy_type} en {model_name}")
        st.plotly_chart(fig_sens, width="stretch", key="pol_sens")

    with st.expander("💾 Guardar este escenario"):
        name = st.text_input("Nombre del escenario", value=f"{policy_type} ({magnitude}%)",
                             key="pol_scen_name")
        desc = st.text_area("Descripción", key="pol_scen_desc",
                            value=result.get("mechanism", "")[:120])
        if st.button("Guardar escenario", key="pol_scen_save"):
            before_eq = result["before"]
            after_eq = result["after"]
            st.session_state.scenario_manager.save(
                name=name, description=desc, model_name=model_name,
                params=base_params, before=before_eq, after=after_eq,
                policy_type=policy_type, magnitude=magnitude / 100.0)
            st.success(f"Escenario '{name}' guardado.")


def page_scenarios():
    """Gestión de escenarios guardados."""
    st.header("Guardar y comparar escenarios")

    manager = st.session_state.scenario_manager
    scenarios = manager.list_all()

    if not scenarios:
        st.info("Aún no hay escenarios guardados. Ejecute una simulación de "
                "políticas y guárdela, o cargue escenarios desde la página "
                "de Simulación.")
        return

    st.markdown(f"**{len(scenarios)} escenario(s) guardado(s).**")
    names = {f"{sc['id']} · {sc['name']}": sc for sc in scenarios}

    col_list, col_compare = st.columns([1, 2])
    with col_list:
        selected = st.multiselect("Seleccionar escenarios a comparar",
                                  list(names.keys()), key="scen_select",
                                  default=list(names.keys())[:2])
        st.divider()
        del_id = st.selectbox("Eliminar escenario", list(names.keys()),
                              key="scen_delete")
        if st.button("🗑️ Eliminar", key="scen_delete_btn"):
            manager.delete(names[del_id]["id"])
            st.rerun()
        if st.button("🗑️ Eliminar todos", key="scen_delete_all"):
            manager.delete_all()
            st.rerun()

    with col_compare:
        chosen = [names[k] for k in selected]
        if chosen:
            summary = manager.compare(chosen)
            st.dataframe(pd.DataFrame(summary), width="stretch")
            st.divider()
            st.markdown("#### Detalle")
            for sc in chosen:
                with st.expander(f"{sc['name']} · {sc['model']}"):
                    st.markdown(f"**Descripción:** {sc['description'] or '—'}")
                    st.markdown(f"**Política:** {sc['policy_type']} · "
                                f"magnitud {sc['magnitude'] * 100:.0f}% · "
                                f"creado {sc['created_at']}")
                    st.json({"before": sc.get("before"),
                             "after": sc.get("after")})


def page_reports():
    """Generación de reportes PDF."""
    st.header("Generación de reportes PDF")

    tab_sim, tab_data = st.tabs(["Reporte de simulación", "Reporte de datos"])

    with tab_sim:
        st.markdown("Genere un informe profesional con los resultados de una "
                    "nueva simulación o de un escenario guardado.")

        manager = st.session_state.scenario_manager
        scenarios = manager.list_all()

        c1, c2 = st.columns(2)
        with c1:
            source = st.radio("Origen", ["Nueva simulación", "Escenario guardado"],
                              key="rep_source")
        with c2:
            if source == "Escenario guardado" and scenarios:
                sc_options = {f"{sc['id']} · {sc['name']}": sc for sc in scenarios}
                sc_sel = st.selectbox("Escenario", list(sc_options.keys()),
                                      key="rep_scen")
            else:
                sc_sel = None

        if source == "Nueva simulación":
            c1, c2, c3 = st.columns(3)
            model_name = c1.selectbox("Modelo", ["IS-LM", "AD-AS"],
                                      key="rep_model")
            shock_label = c2.selectbox("Choque", SHOCK_KEYS[model_name],
                                       key="rep_shock")
            magnitude = c3.slider("Magnitud (%)", 5, 30, 10, 1,
                                  key="rep_mag")

        conclusions = st.text_area(
            "Conclusiones (se incluirán en el reporte)",
            key="rep_conclusions",
            help="Comentarios analíticos sobre el resultado de la simulación.")

        can_generate = True
        if source == "Escenario guardado" and sc_sel is None:
            st.info("No hay escenarios guardados. Guarde uno primero desde la "
                    "página Simulación de Políticas.")
            can_generate = False

        if can_generate and st.button("📄 Generar PDF", key="rep_generate"):
            if source == "Escenario guardado" and sc_sel:
                sc = sc_options[sc_sel]
                model_name = sc["model"]
                params_before = sc["params"]
                before = sc["before"]
                after = sc["after"]
                shock_label = None
                magnitude = sc.get("magnitude", 0.10)
                policy_type = sc.get("policy_type", "")
                deltas = {k: after.get(k, 0) - before.get(k, 0)
                          for k in before if k in after}
                subtitle = (f"Escenario guardado: {sc['name']} · "
                            f"{sc.get('description') or ''}")
            else:
                params_before = (islm_params_from_state()
                                 if model_name == "IS-LM"
                                 else adas_params_from_state())
                model = build_model(model_name, dict(params_before))
                before = model.solve()
                model_after = model.clone()
                after = model_after.apply_shock(shock_label, magnitude / 100.0)
                deltas = {k: after[k] - before[k] for k in before if k in after}
                policy_type = ""
                subtitle = "Simulación de un choque macroeconómico"

            os.makedirs(REPORT_DIR, exist_ok=True)
            report_path = os.path.join(
                REPORT_DIR,
                f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

            precios_texto = ""
            if "P" in before and "P" in after:
                precios_texto = (f"El nivel de precios pasó de "
                                 f"{before['P']:.2f} a {after['P']:.2f}. ")
            summary = (
                f"Se aplicó el choque «{shock_label or policy_type}» con "
                f"magnitud del {magnitude * 100:.0f} % sobre el modelo "
                f"{model_name}. La producción pasó de "
                f"{before['Y']:.2f} a {after['Y']:.2f} "
                f"({deltas.get('Y', 0):+.2f}). {precios_texto}"
            )

            report_data = {
                "title": "Reporte de simulación macroeconómica",
                "subtitle": subtitle,
                "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "model": model_name,
                "params_before": params_before,
                "equilibrium_before": before,
                "equilibrium_after": after,
                "shock_label": shock_label or policy_type,
                "magnitude": magnitude,
                "policy_type": policy_type,
                "deltas": deltas,
                "mechanism": get_shock_mechanism(model_name, shock_label) if shock_label else "",
                "executive_summary": summary,
                "conclusions": conclusions or "",
            }
            generate_pdf_report(report_data, report_path)
            st.session_state.report_path = report_path
            st.success(f"PDF generado: {report_path}")

        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
            with open(st.session_state.report_path, "rb") as fh:
                st.download_button(
                    "⬇️ Descargar reporte PDF", fh,
                    file_name=os.path.basename(st.session_state.report_path),
                    mime="application/pdf", key="rep_download")

    with tab_data:
        st.markdown("Genera un reporte con la estadística descriptiva de los "
                    "datos cargados.")
        data = st.session_state.data
        if data is None:
            st.info("No hay datos cargados. Genérelos o impórtelos en la "
                    "página de Series temporales.")
        elif st.button("📄 Generar reporte de datos", key="rep_data_generate"):
            from reportlab.platypus import Paragraph as PLParagraph
            from reportlab.lib.styles import ParagraphStyle
            stats = DataManager.descriptive_stats(data)
            rows = [["Variable"] + [str(c) for c in stats.columns]]
            for var in stats.index:
                rows.append([str(var)] + [f"{v:.4f}" if isinstance(v, float) else str(v)
                                          for v in stats.loc[var].values])
            os.makedirs(REPORT_DIR, exist_ok=True)
            path = os.path.join(REPORT_DIR,
                                f"datos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            doc = _simple_data_pdf(path, data, rows)
            st.session_state.report_path = path
            st.success(f"PDF generado: {path}")
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button("⬇️ Descargar reporte de datos", fh,
                                       file_name=os.path.basename(path),
                                       mime="application/pdf",
                                       key="rep_data_download")


def _simple_data_pdf(path, data, rows):
    """Construye un PDF sencillo con la estadística descriptiva de los datos."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    base = getSampleStyleSheet()
    small = ParagraphStyle("SmallSICM", parent=base["BodyText"], fontSize=8.5,
                           textColor=colors.HexColor("#555555"))
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    flow = [
        Paragraph("Simulador Integral de Choques Macroeconómicos", small),
        Spacer(1, 4),
        Paragraph("Reporte de datos macroeconómicos", base["Title"]),
        Paragraph(f"Observaciones: {len(data)} · "
                  f"{data['fecha'].iloc[0].date()} a {data['fecha'].iloc[-1].date()}",
                  base["BodyText"]),
        Spacer(1, 10),
        Paragraph("Estadística descriptiva", base["Heading2"]),
    ]
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d3e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f7")]),
    ]))
    flow.append(table)
    doc.build(flow)
    return path


def page_dashboard():
    """Dashboard de laboratorio: indicadores clave y herramientas analíticas."""
    st.header("Dashboard de laboratorio")

    data = st.session_state.data
    if data is None:
        st.info("Primero genere o cargue los datos en la página de "
                "**Series temporales** para poblar el dashboard.")
        return

    # --- Indicadores clave ---
    st.subheader("Indicadores clave")
    dm = DataManager
    cols = st.columns(4)
    with cols[0]:
        st.metric("PIB", f"{data['PIB'].iloc[-1]:.2f}",
                  delta=f"{dm.annual_change(data, 'PIB'):+.2f}%",
                  delta_color="normal",
                  help="PIB del último periodo y variación interanual.")
    with cols[1]:
        st.metric("Inflación (%)", f"{data['Inflacion'].iloc[-1]:.2f}",
                  delta=f"{data['Inflacion'].iloc[-1] - data['Inflacion'].iloc[-13]:+.2f} p.p."
                        if len(data) > 12 else None,
                  help="Inflación anualizada del último periodo.")
    with cols[2]:
        st.metric("Desempleo (%)", f"{data['Desempleo'].iloc[-1]:.2f}",
                  delta=f"{data['Desempleo'].iloc[-1] - data['Desempleo'].iloc[-13]:+.2f} p.p."
                        if len(data) > 12 else None,
                  help="Tasa de desempleo del último periodo.")
    with cols[3]:
        st.metric("Tasa de interés (%)", f"{data['Tasa_Interes'].iloc[-1]:.2f}",
                  delta=f"{data['Tasa_Interes'].iloc[-1] - data['Tasa_Interes'].iloc[-13]:+.2f} p.p."
                        if len(data) > 12 else None,
                  help="Tasa de interés de política del último periodo.")

    st.divider()

    # --- Herramientas analíticas ---
    tab_gap, tab_taylor, tab_cycle = st.tabs(
        ["Brecha del producto", "Regla de Taylor", "Análisis del ciclo"])

    with tab_gap:
        fig = Visualizer.plot_output_gap(data)
        st.plotly_chart(fig, width="stretch", key="dash_gap")

    with tab_taylor:
        c1, c2 = st.columns(2)
        r_neutral = c1.slider("Tasa neutral (%)", 0.0, 6.0, 2.5, 0.1,
                              key="dash_rneutral")
        pi_target = c2.slider("Meta de inflación (%)", 1.0, 6.0, 3.5, 0.1,
                              key="dash_pitarget")
        fig = Visualizer.plot_taylor_rule(data, r_neutral, pi_target)
        st.plotly_chart(fig, width="stretch", key="dash_taylor")
        rate, inflation, gap = dm.taylor_recommendation(
            data, r_neutral=r_neutral, pi_target=pi_target)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tasa recomendada", f"{rate:.2f}%" if rate == rate else "—",
                  help="r_neutral + 1.5·(π - π*) + 0.5·brecha.")
        c2.metric("Inflación actual", f"{inflation:.2f}%" if inflation == inflation else "—")
        c3.metric("Brecha del producto", f"{gap:+.2f}%" if gap == gap else "—")

    with tab_cycle:
        fig = Visualizer.plot_business_cycle(data)
        st.plotly_chart(fig, width="stretch", key="dash_cycle")
        cycle_df = dm.cyclical_analysis(data)
        last = cycle_df["Ciclo"].iloc[-1]
        st.metric("Componente cíclico actual", f"{last:+.2f}",
                  help="Positivo: economía por encima de la tendencia. "
                       "Negativo: por debajo.")


def page_docs():
    """Documentación: guía de usuario, modelos y referencias."""
    st.header("Documentación")

    tab_guia, tab_modelos, tab_refs, tab_ej = st.tabs(
        ["Guía de usuario", "Modelos", "Referencias", "Ejemplos"])

    with tab_guia:
        st.subheader("Guía de usuario")
        st.markdown("""
### Navegación
- Use el **panel lateral** para desplazarse entre los módulos.
- En cada módulo, los **parámetros se configuran con deslizadores** en la
  barra lateral; los resultados se actualizan automáticamente.

### Flujo de trabajo recomendado
1. **Modelos** → explore el equilibrio IS-LM y AD-AS ajustando parámetros.
2. **Visualización de choques** → elija un choque y su magnitud, pulse
   *Actualizar gráfico* y compare el equilibrio antes/después.
3. **Series temporales** → genere datos sintéticos o importe un CSV/Excel.
4. **Simulación de políticas** → aplique políticas fiscales o monetarias y
   revise la sensibilidad.
5. **Guardar escenarios** → guarde y compare configuraciones.
6. **Reportes PDF** → genere un informe con los resultados.
7. **Dashboard** → monitoree indicadores, brecha, regla de Taylor y ciclo.
""")

    with tab_modelos:
        st.subheader("Explicación de los modelos")
        st.markdown("""
### Modelo IS-LM
El mercado de bienes y el de dinero determinan conjuntamente la producción y
la tasa de interés de equilibrio.

- **IS** (bienes): Y = C + I + G, con C = C₀ + c(Y-T) e I = I₀ - 100b·r.
  Pendiente negativa en (Y, r).
- **LM** (dinero): M/P = k·Y - h·r. Pendiente positiva en (Y, r).
- **Multiplicadores**:
  - Fiscal: ΔY/ΔG = h / [h(1-c) + 100·b·k].
  - Monetario: ΔY/ΔM = 100·b / [P·(h(1-c) + 100·b·k)].

### Modelo AD-AS
Determina el nivel de precios y la producción de corto y largo plazo.

- **AD**: Y = M·V/P (pendiente negativa).
- **SRAS**: P = Pₑ[1 + λ(Y - Yₙ)] (pendiente positiva).
- **LRAS**: vertical en Y = Yₙ.
- Un choque de demanda mueve la AD (efecto real en el corto plazo porque los
  precios esperados no se ajustan de inmediato); un choque de oferta mueve
  la SRAS/LRAS.

### Modelo Mundell-Fleming
Extiende el IS-LM a una economía abierta pequeña con tipo de cambio fijo o
flexible.

- **Flexible**: la política monetaria afecta a Y (e deprecia); la fiscal solo
  aprecia el tipo de cambio (no afecta a Y) con alta movilidad de capitales.
- **Fijo**: la política fiscal es efectiva; la monetaria es endógena y pierde
  autonomía.
""")

    with tab_refs:
        st.subheader("Referencias bibliográficas")
        st.markdown("""
- Blanchard, O. (2017). *Macroeconomía* (7.ª ed.). Pearson.
- Mankiw, N. G. (2019). *Macroeconomía* (10.ª ed.). Antoni Bosch.
- Dornbusch, R., Fischer, S. y Startz, R. (2018). *Macroeconomics* (13.ª ed.).
  McGraw-Hill.
- Taylor, J. B. (1993). "Discretion versus policy rules in practice".
  *Carnegie-Rochester Conference Series on Public Policy*, 39, 195-214.
- Hodrick, R. J. y Prescott, E. C. (1997). "Postwar U.S. business cycles: An
  empirical investigation". *Journal of Money, Credit and Banking*, 29(1), 1-16.
""")

    with tab_ej:
        st.subheader("Ejemplos de uso")
        st.markdown("""
### Ejemplo 1: Choque fiscal expansivo en IS-LM
- Modelo IS-LM, choque «Gasto Gobierno ↑», magnitud 20 %.
- Resultado: Y aumenta (multiplicador fiscal ~3.8) y r sube por el crowding out.

### Ejemplo 2: Política monetaria bajo tipo de cambio fijo
- Mundell-Fleming, régimen **Fijo**, choque «Oferta Monetaria ↑».
- Resultado: la política monetaria no puede aplicarse de forma autónoma;
  M se vuelve endógena para defender el ancla cambiaria.

### Ejemplo 3: Regla de Taylor
- En el Dashboard, la tasa recomendada se compara con la tasa vigente.
- Si la brecha del producto es negativa, la regla sugiere bajar la tasa.
""")


# ---------------------------------------------------------------------------
# Aplicación principal
# ---------------------------------------------------------------------------
def main():
    """Punto de entrada de la aplicación Streamlit."""
    inject_css()
    init_session_state()

    NAV_OPTIONS = [
        "🏠 Inicio",
        "🧮 Modelos IS-LM y AD-AS",
        "🌐 Mundell-Fleming",
        "⚡ Visualización de Choques",
        "📊 Series Temporales",
        "🛠️ Simulación de Políticas",
        "💾 Guardar Escenarios",
        "📄 Reportes PDF",
        "📈 Dashboard Laboratorio",
        "📚 Documentación",
    ]

    with st.sidebar:
        st.markdown("## 📈 SICM v2.0")
        st.caption("Simulador Integral de Choques Macroeconómicos")
        st.divider()
        selection = st.radio("Navegación", NAV_OPTIONS,
                             label_visibility="collapsed")
        st.divider()
        st.caption("Universidad · Investigación · Docencia")

    pages = {
        "🏠 Inicio": page_inicio,
        "🧮 Modelos IS-LM y AD-AS": page_models,
        "🌐 Mundell-Fleming": page_mundell_fleming,
        "⚡ Visualización de Choques": page_shocks,
        "📊 Series Temporales": page_series,
        "🛠️ Simulación de Políticas": page_policies,
        "💾 Guardar Escenarios": page_scenarios,
        "📄 Reportes PDF": page_reports,
        "📈 Dashboard Laboratorio": page_dashboard,
        "📚 Documentación": page_docs,
    }

    pages[selection]()


if __name__ == "__main__":
    main()
