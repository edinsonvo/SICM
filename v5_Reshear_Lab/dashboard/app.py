"""
SICM v5 Research Lab — Dashboard Principal
===========================================
Fase 4: Aplicación Streamlit para simulación macroeconómica universitaria.

Tabs:
- Simulación: Vista simple con indicadores
- Cuatro Planos: Análisis completo
- Choques: Motor de choques + mecanismo de transmisión
- Comparador: Escenarios Base/A/B/C
- Exportar: CSV, JSON, Configuración
"""
pip install -r requirements.txt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parameters import EconomyConfig, TipoEconomia, RegimenCambiario, MovilidadCapital
from core.shocks import ShockEngine
from core.equilibrium import EquilibriumSolver, EquilibriumResult
from models.islm import ISLMModel
from models.mundell_fleming import MundellFlemingModel
from models.classical_closed import ClassicalClosedModel
from models.classical_open import ClassicalOpenModel
from visualization.single_view import SingleView
from visualization.four_planes import FourPlanesView
from visualization.transmission import TransmissionMechanism

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="SICM v5 Research Lab",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .indicator-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .indicator-value {
        font-size: 2.2rem;
        font-weight: bold;
    }
    .indicator-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .indicator-delta {
        font-size: 0.8rem;
        margin-top: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZACIÓN DE ESTADO DE SESIÓN
# ============================================================
def init_session_state():
    defaults = {
        'config': EconomyConfig(),
        'scenarios': {"Base": None, "Escenario A": None, "Escenario B": None, "Escenario C": None},
        'current_scenario': "Base",
        'shock_history': [],
        'last_result': None,
        'model_type': 'keynesian'
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">� SICM v5 Research Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Laboratorio de Simulación Macroeconómica Universitaria</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR — CONFIGURACIÓN
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")

    # Selector de modelo
    model_type = st.radio(
        "Paradigma",
        options=["Keynesiano", "Clásico"],
        index=0 if st.session_state.model_type == 'keynesian' else 1,
        help="Seleccione el paradigma económico para la simulación"
    )
    st.session_state.model_type = 'keynesian' if model_type == "Keynesiano" else 'classical'

    st.divider()

    # Tipo de economía
    tipo_economia = st.selectbox(
        "Tipo de Economía",
        options=["Cerrada", "Abierta"],
        index=0 if st.session_state.config.tipo_economia == TipoEconomia.CERRADA else 1
    )

    if tipo_economia == "Abierta":
        st.session_state.config.tipo_economia = TipoEconomia.ABIERTA

        col1, col2 = st.columns(2)
        with col1:
            regimen = st.selectbox(
                "Régimen Cambiario",
                options=["Flexible", "Fijo"],
                index=0 if st.session_state.config.regimen_cambiario == RegimenCambiario.FLEXIBLE else 1
            )
            st.session_state.config.regimen_cambiario = RegimenCambiario.FLEXIBLE if regimen == "Flexible" else RegimenCambiario.FIJO

        with col2:
            movilidad = st.selectbox(
                "Movilidad de Capital",
                options=["Perfecta", "Imperfecta", "Nula"],
                index=0 if st.session_state.config.movilidad_capital == MovilidadCapital.PERFECTA else 
                      (1 if st.session_state.config.movilidad_capital == MovilidadCapital.IMPERFECTA else 2)
            )
            st.session_state.config.movilidad_capital = {
                "Perfecta": MovilidadCapital.PERFECTA,
                "Imperfecta": MovilidadCapital.IMPERFECTA,
                "Nula": MovilidadCapital.NULA
            }[movilidad]
    else:
        st.session_state.config.tipo_economia = TipoEconomia.CERRADA

    st.divider()

    # Parámetros principales
    st.subheader("� Parámetros del Modelo")

    with st.expander("Demanda Agregada", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.config.G = st.number_input("G (Gasto público)", value=float(st.session_state.config.G), step=10.0, format="%.1f")
            st.session_state.config.consumo.c0 = st.number_input("c₀ (Consumo autónomo)", value=float(st.session_state.config.consumo.c0), step=10.0, format="%.1f")
            st.session_state.config.consumo.c1 = st.slider("c₁ (PMC)", 0.0, 1.0, float(st.session_state.config.consumo.c1), 0.05)
        with col2:
            st.session_state.config.consumo.T = st.number_input("T (Impuestos)", value=float(st.session_state.config.consumo.T), step=10.0, format="%.1f")
            st.session_state.config.inversion.I0 = st.number_input("I₀ (Inversión autónoma)", value=float(st.session_state.config.inversion.I0), step=10.0, format="%.1f")
            st.session_state.config.inversion.b = st.number_input("b (Sens. interés)", value=float(st.session_state.config.inversion.b), step=5.0, format="%.1f")

    with st.expander("Mercado de Dinero"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.config.dinero.M = st.number_input("M (Oferta monetaria)", value=float(st.session_state.config.dinero.M), step=50.0, format="%.1f")
            st.session_state.config.dinero.P = st.number_input("P (Nivel de precios)", value=float(st.session_state.config.dinero.P), step=0.1, format="%.3f")
        with col2:
            st.session_state.config.dinero.k = st.number_input("k (Sens. ingreso)", value=float(st.session_state.config.dinero.k), step=0.1, format="%.3f")
            st.session_state.config.dinero.h = st.number_input("h (Sens. interés)", value=float(st.session_state.config.dinero.h), step=2.0, format="%.1f")

    with st.expander("Producción"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.config.produccion.A = st.number_input("A (Productividad)", value=float(st.session_state.config.produccion.A), step=0.1, format="%.2f")
            st.session_state.config.produccion.K = st.number_input("K (Capital)", value=float(st.session_state.config.produccion.K), step=10.0, format="%.1f")
        with col2:
            st.session_state.config.produccion.alpha = st.slider("α (Elasticidad capital)", 0.0, 1.0, float(st.session_state.config.produccion.alpha), 0.05)
            st.session_state.config.produccion.L_bar = st.number_input("L̄ (Oferta trabajo)", value=float(st.session_state.config.produccion.L_bar), step=10.0, format="%.1f")

    if st.session_state.config.tipo_economia == TipoEconomia.ABIERTA:
        with st.expander("Sector Externo"):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.config.externo.X0 = st.number_input("X₀ (Exportaciones)", value=float(st.session_state.config.externo.X0), step=10.0, format="%.1f")
                st.session_state.config.externo.m = st.number_input("m (PMI)", value=float(st.session_state.config.externo.m), step=0.05, format="%.3f")
            with col2:
                st.session_state.config.externo.E = st.number_input("E (Tipo de cambio)", value=float(st.session_state.config.externo.E), step=0.1, format="%.3f")
                st.session_state.config.externo.r_f = st.number_input("r* (Interés extranjero)", value=float(st.session_state.config.externo.r_f), step=0.01, format="%.3f")

    st.divider()

    # Acciones
    col1, col2 = st.columns(2)
    with col1:
        if st.button("� Resetear", use_container_width=True):
            st.session_state.config = EconomyConfig()
            st.session_state.scenarios = {"Base": None, "Escenario A": None, "Escenario B": None, "Escenario C": None}
            st.session_state.shock_history = []
            st.rerun()
    with col2:
        if st.button("� Guardar Config", use_container_width=True):
            config_json = json.dumps(st.session_state.config.to_dict(), indent=2)
            st.download_button(
                label="⬇️ Descargar JSON",
                data=config_json,
                file_name="sicm_v5_config.json",
                mime="application/json",
                use_container_width=True
            )

# ============================================================
# FUNCIÓN AUXILIAR: CALCULAR EQUILIBRIO
# ============================================================
def calculate_equilibrium():
    """Calcula el equilibrio según la configuración actual"""
    config = st.session_state.config
    solver = EquilibriumSolver(config)

    if st.session_state.model_type == 'classical':
        if config.tipo_economia == TipoEconomia.CERRADA:
            result = solver.solve_classical_closed()
            modelo_nombre = "Clásico Cerrado"
        else:
            result = solver.solve_classical_open()
            modelo_nombre = "Clásico Abierto"
    else:
        if config.tipo_economia == TipoEconomia.CERRADA:
            result = solver.solve_islm_cerrado()
            modelo_nombre = "IS-LM (Cerrado)"
        else:
            result = solver.solve_mundell_fleming()
            regime = config.regimen_cambiario.value.title()
            movil = config.movilidad_capital.value.title()
            modelo_nombre = f"Mundell-Fleming ({regime}, {movil})"

    st.session_state.last_result = result
    st.session_state.scenarios[st.session_state.current_scenario] = result

    return result, modelo_nombre

# ============================================================
# TABS PRINCIPALES
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "� Simulación", 
    "� Cuatro Planos", 
    "⚡ Choques & Transmisión", 
    "� Comparador", 
    "� Exportar"
])

# ============================================================
# TAB 1: SIMULACIÓN
# ============================================================
with tab1:
    result, modelo_nombre = calculate_equilibrium()

    # Indicadores principales
    st.subheader(f"Modelo: {modelo_nombre}")

    cols = st.columns(5)
    indicators = [
        ("PIB (Y)", f"{result.Y:.1f}", "�", result.Y),
        ("Interés (r)", f"{result.r:.3f}", "�", result.r),
        ("Precios (P)", f"{result.P:.3f}", "�️", result.P),
        ("Empleo (L)", f"{result.L:.1f}", "�", result.L),
        ("Desempleo", f"{result.desempleo:.1f}%", "⚠️", result.desempleo)
    ]

    for col, (label, value, icon, raw_val) in zip(cols, indicators):
        with col:
            # Color según valor
            color = "#28a745" if raw_val > 0 else "#dc3545"
            st.markdown(f"""
            <div class="indicator-card">
                <div class="indicator-label">{icon} {label}</div>
                <div class="indicator-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # Segunda fila de indicadores
    cols2 = st.columns(4)
    indicators2 = [
        ("Consumo (C)", f"{result.C:.1f}", "�"),
        ("Inversión (I)", f"{result.I:.1f}", "�"),
        ("Gasto Público (G)", f"{result.G:.1f}", "�️"),
        ("NX", f"{result.NX:.1f}", "�")
    ]
    for col, (label, value, icon) in zip(cols2, indicators2):
        with col:
            st.metric(label=f"{icon} {label}", value=value)

    # Gráfica
    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Visualización del Modelo")

        if st.session_state.config.tipo_economia == TipoEconomia.CERRADA:
            modelo_viz = "islm" if st.session_state.model_type == 'keynesian' else "classical_closed"
        else:
            modelo_viz = "mundell_fleming" if st.session_state.model_type == 'keynesian' else "classical_open"

        view = SingleView(st.session_state.config, modelo_viz)
        fig = view.plot(width=800, height=550)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Parámetros Actuales")

        params_df = pd.DataFrame({
            'Parámetro': ['c₀', 'c₁', 'T', 'I₀', 'b', 'M', 'P', 'k', 'h', 'G', 'A', 'K', 'α', 'L̄'],
            'Valor': [
                st.session_state.config.consumo.c0,
                st.session_state.config.consumo.c1,
                st.session_state.config.consumo.T,
                st.session_state.config.inversion.I0,
                st.session_state.config.inversion.b,
                st.session_state.config.dinero.M,
                st.session_state.config.dinero.P,
                st.session_state.config.dinero.k,
                st.session_state.config.dinero.h,
                st.session_state.config.G,
                st.session_state.config.produccion.A,
                st.session_state.config.produccion.K,
                st.session_state.config.produccion.alpha,
                st.session_state.config.produccion.L_bar
            ]
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Multiplicadores
        st.subheader("Multiplicadores")
        c1 = st.session_state.config.consumo.c1
        multiplier = 1 / (1 - c1) if c1 < 1 else float('inf')
        st.metric("Multiplicador simple", f"{multiplier:.3f}")

        # Crowding-out
        b = st.session_state.config.inversion.b
        h = st.session_state.config.dinero.h
        k = st.session_state.config.dinero.k
        denom = (1 - c1) + (b * k) / h
        multiplier_full = 1 / denom
        st.metric("Multiplicador completo", f"{multiplier_full:.3f}")

# ============================================================
# TAB 2: CUATRO PLANOS
# ============================================================
with tab2:
    st.header("Análisis de Cuatro Planos")

    four_view = FourPlanesView(st.session_state.config)

    if st.session_state.model_type == 'keynesian':
        fig4 = four_view.plot_keynesian(width=1200, height=900)
        st.caption("Plano 1: IS-LM | Plano 2: DA-OA | Plano 3: Mercado de Trabajo | Plano 4: Mecanismo de Transmisión")
    else:
        fig4 = four_view.plot_classical(width=1200, height=900)
        st.caption("Plano 1: Mercado de Trabajo | Plano 2: Función de Producción | Plano 3: Fondos Prestables | Plano 4: OA-DA")

    st.plotly_chart(fig4, use_container_width=True)

    # Explicación teórica
    with st.expander("� Explicación teórica de los 4 planos"):
        if st.session_state.model_type == 'keynesian':
            st.markdown("""
            **Plano 1: IS-LM**
            - IS: Equilibrio en el mercado de bienes (Y = C + I + G)
            - LM: Equilibrio en el mercado de dinero (M/P = L(Y,r))
            - Intersección: Equilibrio simultáneo de ambos mercados

            **Plano 2: DA-OA**
            - DA: Demanda Agregada (derivada de IS-LM)
            - OA: Oferta Agregada keynesiana (horizontal a corto plazo)

            **Plano 3: Mercado de Trabajo**
            - Nd: Demanda de trabajo (PML = W/P)
            - Ns: Oferta de trabajo efectiva

            **Plano 4: Mecanismo de Transmisión**
            - Flujo causal de políticas económicas
            """)
        else:
            st.markdown("""
            **Plano 1: Mercado de Trabajo**
            - Nd: Demanda de trabajo (PML = W/P)
            - Ns: Oferta de trabajo (vertical en L̄)

            **Plano 2: Función de Producción**
            - Y = A·K^α·L^(1-α)
            - Pleno empleo determina Y*

            **Plano 3: Fondos Prestables**
            - S: Ahorro (función creciente de r)
            - I: Inversión (función decreciente de r)

            **Plano 4: OA-DA**
            - OA: Vertical en Y* (pleno empleo)
            - DA: Determinada por S=I y M
            """)

# ============================================================
# TAB 3: CHOQUES & TRANSMISIÓN
# ============================================================
with tab3:
    st.header("⚡ Motor de Choques Económicos")

    shock_engine = ShockEngine(st.session_state.config)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("� Choque Fiscal")
        delta_G = st.number_input("ΔG (Gasto público)", value=0.0, step=10.0, key="shock_g")
        delta_T = st.number_input("ΔT (Impuestos)", value=0.0, step=10.0, key="shock_t")
        delta_c0 = st.number_input("Δc₀ (Consumo autónomo)", value=0.0, step=10.0, key="shock_c0")

        if st.button("� Aplicar Choque Fiscal", use_container_width=True):
            new_config, shock, narrativa = shock_engine.aplicar_choque_fiscal(delta_G, delta_T, delta_c0)
            st.session_state.config = new_config
            st.session_state.shock_history.append({
                'tipo': 'Fiscal',
                'descripcion': shock.descripcion,
                'narrativa': narrativa
            })
            st.success(f"✅ {shock.descripcion}")
            st.info(f"**Mecanismo:** {narrativa}")
            st.rerun()

    with col2:
        st.subheader("� Choque Monetario")
        delta_M = st.number_input("ΔM (Oferta monetaria)", value=0.0, step=50.0, key="shock_m")
        delta_P = st.number_input("ΔP (Nivel de precios)", value=0.0, step=0.1, key="shock_p")

        if st.button("� Aplicar Choque Monetario", use_container_width=True):
            new_config, shock, narrativa = shock_engine.aplicar_choque_monetario(delta_M, delta_P)
            st.session_state.config = new_config
            st.session_state.shock_history.append({
                'tipo': 'Monetario',
                'descripcion': shock.descripcion,
                'narrativa': narrativa
            })
            st.success(f"✅ {shock.descripcion}")
            st.info(f"**Mecanismo:** {narrativa}")
            st.rerun()

    with col3:
        st.subheader("� Choque de Oferta")
        delta_A = st.number_input("ΔA (Productividad)", value=0.0, step=0.1, format="%.2f", key="shock_a")
        delta_K = st.number_input("ΔK (Capital)", value=0.0, step=10.0, key="shock_k")

        if st.button("� Aplicar Choque de Oferta", use_container_width=True):
            new_config, shock, narrativa = shock_engine.aplicar_choque_oferta(delta_A, delta_K)
            st.session_state.config = new_config
            st.session_state.shock_history.append({
                'tipo': 'Oferta',
                'descripcion': shock.descripcion,
                'narrativa': narrativa
            })
            st.success(f"✅ {shock.descripcion}")
            st.info(f"**Mecanismo:** {narrativa}")
            st.rerun()

    # Historial de choques
    if st.session_state.shock_history:
        with st.expander("� Historial de Choques Aplicados"):
            for i, h in enumerate(st.session_state.shock_history, 1):
                st.write(f"{i}. **{h['tipo']}**: {h['descripcion']}")
                st.caption(f"→ {h['narrativa']}")

    # Mecanismo de transmisión
    st.divider()
    st.header("� Mecanismo de Transmisión")

    mech_type = st.selectbox(
        "Seleccione política para visualizar mecanismo",
        ["Expansión Fiscal", "Contracción Fiscal", "Expansión Monetaria", "Contracción Monetaria", "Shock de Oferta Negativo"]
    )

    transmission = TransmissionMechanism(st.session_state.config)

    if mech_type == "Expansión Fiscal":
        mech = transmission.generate_fiscal_expansion(50)
    elif mech_type == "Contracción Fiscal":
        mech = transmission.generate_fiscal_expansion(-50)
    elif mech_type == "Expansión Monetaria":
        mech = transmission.generate_monetary_expansion(100)
    elif mech_type == "Contracción Monetaria":
        mech = transmission.generate_monetary_expansion(-100)
    else:
        mech = transmission.generate_supply_shock(-0.2)

    col_mech1, col_mech2 = st.columns([3, 2])

    with col_mech1:
        fig_mech = transmission.plot(mech, width=900, height=500)
        st.plotly_chart(fig_mech, use_container_width=True)

    with col_mech2:
        st.subheader("Resumen del Mecanismo")
        st.text(transmission.get_text_summary(mech))

        # Pasos detallados
        st.subheader("Pasos del Mecanismo")
        for i, paso in enumerate(mech['pasos'], 1):
            with st.container():
                st.markdown(f"**{i}. {paso['nodo']}**")
                st.caption(f"{paso['descripcion']}")
                st.caption(f"*Efecto: {paso['efecto']}*")

# ============================================================
# TAB 4: COMPARADOR DE ESCENARIOS
# ============================================================
with tab4:
    st.header("� Comparador de Escenarios")

    st.info("� Guarde escenarios desde la barra lateral para comparar políticas alternativas")

    # Guardar escenarios
    cols_save = st.columns(4)
    scenario_names = ["Base", "Escenario A", "Escenario B", "Escenario C"]

    for col, name in zip(cols_save, scenario_names):
        with col:
            if st.button(f"� Guardar como {name}", use_container_width=True):
                st.session_state.scenarios[name] = st.session_state.last_result
                st.success(f"✅ {name} guardado")
                st.rerun()

    # Comparar
    st.divider()

    available = [k for k, v in st.session_state.scenarios.items() if v is not None]

    if len(available) >= 2:
        selected = st.multiselect(
            "Seleccionar escenarios para comparar", 
            available, 
            default=available[:min(2, len(available))]
        )

        if len(selected) >= 2:
            # Tabla comparativa
            comparison_data = []
            for name in selected:
                r = st.session_state.scenarios[name]
                row = {"Escenario": name}
                row.update(r.to_dict())
                comparison_data.append(row)

            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(df_comp.set_index("Escenario"), use_container_width=True)

            # Gráfico de comparación radar
            categories = ["PIB (Y)", "Consumo (C)", "Inversión (I)", "Interés (r)"]

            fig_comp = go.Figure()

            for name in selected:
                r = st.session_state.scenarios[name]
                values = [r.Y, r.C, r.I, r.r * 1000]  # Escalar r para visualización

                fig_comp.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Cerrar el polígono
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=name
                ))

            fig_comp.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Comparación de Escenarios (Radar)"
            )

            st.plotly_chart(fig_comp, use_container_width=True)

            # Diferencias porcentuales
            st.subheader("Diferencias respecto al Base")
            if "Base" in selected:
                base_result = st.session_state.scenarios["Base"]
                diff_data = []
                for name in selected:
                    if name == "Base":
                        continue
                    r = st.session_state.scenarios[name]
                    diff_data.append({
                        "Escenario": name,
                        "ΔY (%)": f"{((r.Y - base_result.Y) / base_result.Y * 100):.2f}",
                        "Δr (pp)": f"{(r.r - base_result.r):.4f}",
                        "ΔC (%)": f"{((r.C - base_result.C) / base_result.C * 100):.2f}",
                        "ΔI (%)": f"{((r.I - base_result.I) / base_result.I * 100):.2f}"
                    })

                if diff_data:
                    st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)
    else:
        st.info("� Guarde al menos 2 escenarios para comparar. Use los botones de 'Guardar como' arriba.")

# ============================================================
# TAB 5: EXPORTAR
# ============================================================
with tab5:
    st.header("� Exportación de Resultados")

    result = st.session_state.last_result

    if result:
        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            st.subheader("� CSV")
            df = pd.DataFrame([result.to_dict()])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Descargar resultados CSV",
                data=csv,
                file_name=f"sicm_v5_{st.session_state.current_scenario}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_exp2:
            st.subheader("� JSON")
            json_data = json.dumps(result.to_dict(), indent=2)
            st.download_button(
                label="Descargar resultados JSON",
                data=json_data,
                file_name=f"sicm_v5_{st.session_state.current_scenario}.json",
                mime="application/json",
                use_container_width=True
            )

        with col_exp3:
            st.subheader("⚙️ Configuración")
            config_json = json.dumps(st.session_state.config.to_dict(), indent=2)
            st.download_button(
                label="Descargar configuración",
                data=config_json,
                file_name="sicm_v5_config.json",
                mime="application/json",
                use_container_width=True
            )

        # Resumen completo
        st.divider()
        st.subheader("Resumen Completo del Escenario")

        full_summary = {
            "modelo": modelo_nombre if 'modelo_nombre' in dir() else "N/A",
            "paradigma": st.session_state.model_type,
            "tipo_economia": st.session_state.config.tipo_economia.value,
            "configuracion": st.session_state.config.to_dict(),
            "equilibrio": result.to_dict(),
            "historial_choques": st.session_state.shock_history
        }

        st.json(full_summary)

        # Exportar todo
        full_json = json.dumps(full_summary, indent=2)
        st.download_button(
            label="� Descargar Resumen Completo (JSON)",
            data=full_json,
            file_name="sicm_v5_full_report.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No hay resultados para exportar. Ejecute una simulación primero.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 20px;">
    <b>SICM v5 Research Lab</b> — Desarrollado para docencia e investigación macroeconómica<br>
    Modelos: IS-LM | Mundell-Fleming | Clásico Cerrado | Clásico Abierto<br>
    Econometría: ARIMA | VAR | VECM | Monte Carlo
</div>
""", unsafe_allow_html=True)
