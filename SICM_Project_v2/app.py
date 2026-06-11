import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.settings import Settings, EconomyType, ExchangeRate, CapitalMobility, Horizon
from models.keynesian import ISLM, MundellFleming
from models.classical import ADASClassical, LoanableFunds
from models.nk import NewKeynesian
from models.growth import Solow
from shocks.engine import ShockEngine
from dynamics.simulator import DynamicSimulator
from econometrics.lab import EconometricLab
from visualization.plots import *

st.set_page_config(page_title="SICM - Simulador Macroeconómico", layout="wide", page_icon="📊")

# Inicializar estado de sesión
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model = None
    st.session_state.settings = None
    st.session_state.results = {}

# Título
st.title("📊 Simulador Integral de Choques Macroeconómicos (SICM)")
st.markdown("*Enseña, simula, compara y analiza política económica*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de modelo
    model_category = st.selectbox("Familia de modelos", 
                                  ["Keynesianos", "Clásicos", "Nuevos Keynesianos", "Crecimiento"])
    
    if model_category == "Keynesianos":
        model_type = st.selectbox("Modelo específico", ["IS-LM", "Mundell-Fleming"])
    elif model_category == "Clásicos":
        model_type = st.selectbox("Modelo específico", ["AD-AS Clásico", "Fondos Prestables"])
    elif model_category == "Nuevos Keynesianos":
        model_type = st.selectbox("Modelo específico", ["NK 3 ecuaciones", "Regla de Taylor"])
    else:
        model_type = st.selectbox("Modelo específico", ["Solow", "Ramsey"])
    
    st.markdown("---")
    st.header("🏛️ Régimen económico")
    
    economy = st.selectbox("Tipo de economía", [e.value for e in EconomyType])
    exchange = st.selectbox("Régimen cambiario", [e.value for e in ExchangeRate])
    capital = st.selectbox("Movilidad de capitales", [c.value for c in CapitalMobility])
    horizon = st.selectbox("Horizonte", [h.value for h in Horizon])
    
    st.markdown("---")
    st.header("🔧 Choque")
    
    # Choques según modelo
    if model_category == "Keynesianos":
        shock_options = ["↑ G", "↓ G", "↑ M", "↓ M", "↑ T", "↓ T"]
    elif model_category == "Clásicos":
        shock_options = ["↑ M", "↓ M", "↑ Productividad", "↓ Productividad"]
    elif model_category == "Nuevos Keynesianos":
        shock_options = ["Demanda", "Oferta (Costos)", "↑ M"]
    else:
        shock_options = ["↑ s (Ahorro)", "↑ n (Población)", "↑ g (Tecnología)"]
    
    shock_type = st.selectbox("Tipo de choque", shock_options)
    magnitude = st.slider("Magnitud", -0.5, 0.5, 0.1, 0.01)
    persistence = st.slider("Persistencia (solo dinámico)", 0.0, 1.0, 0.8, 0.05)
    
    apply_shock = st.button("▶️ Aplicar Choque", type="primary", use_container_width=True)

# Crear modelo según configuración
settings = Settings(
    economy=EconomyType(economy),
    exchange_rate=ExchangeRate(exchange),
    capital_mobility=CapitalMobility(capital),
    horizon=Horizon(horizon)
)
settings.validate()

# Instanciar modelo
if model_type == "IS-LM":
    model = ISLM(settings)
elif model_type == "Mundell-Fleming":
    model = MundellFleming(settings)
elif model_type == "AD-AS Clásico":
    model = ADASClassical(settings)
elif model_type == "Fondos Prestables":
    model = LoanableFunds(settings)
elif model_type in ["NK 3 ecuaciones", "Regla de Taylor"]:
    model = NewKeynesian(settings)
else:  # Solow
    model = Solow(settings)

# Resolver equilibrio inicial
if st.session_state.model != model:
    st.session_state.model = model
    initial_eq = model.solve()
    st.session_state.initial_eq = initial_eq
    st.session_state.current_eq = initial_eq
    st.session_state.shocked = False
else:
    initial_eq = st.session_state.initial_eq
    if not st.session_state.shocked:
        st.session_state.current_eq = initial_eq

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Curvas y Equilibrio", "🔄 Mecanismo y Simulación", "🏫 Comparador Escuelas", 
     "📐 Crecimiento (Solow)", "🔬 Laboratorio Econométrico"]
)

# Tab 1: Curvas
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Modelo: {model_type}")
        
        if model_type in ["IS-LM", "Mundell-Fleming"]:
            if st.session_state.shocked:
                fig = plot_is_lm(model, initial_eq, st.session_state.current_eq)
            else:
                fig = plot_is_lm(model, initial_eq)
        elif model_type == "AD-AS Clásico":
            fig = plot_ad_as(model, st.session_state.current_eq)
        else:
            # Placeholder para otros modelos
            fig = go.Figure()
            fig.add_annotation(text=f"Visualización para {model_type} en desarrollo", 
                              x=0.5, y=0.5, showarrow=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Equilibrio Actual")
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Producción (Y)", f"{st.session_state.current_eq['Y']:.2f}")
            if 'i' in st.session_state.current_eq:
                st.metric("Tasa interés", f"{st.session_state.current_eq['i']:.2%}")
            if 'P' in st.session_state.current_eq:
                st.metric("Precios", f"{st.session_state.current_eq['P']:.2f}")
        
        if st.session_state.shocked:
            with col2b:
                st.metric("Δ Producción", 
                         f"{st.session_state.current_eq['Y'] - initial_eq['Y']:+.2f}",
                         delta_color="inverse")
                if 'i' in st.session_state.current_eq:
                    delta_i = st.session_state.current_eq['i'] - initial_eq.get('i', 0.05)
                    st.metric("Δ Tasa", f"{delta_i:+.2%}")

# Tab 2: Mecanismo y simulación dinámica
with tab2:
    if apply_shock:
        engine = ShockEngine(model)
        new_eq, mechanism = engine.apply(shock_type, magnitude)
        st.session_state.current_eq = new_eq
        st.session_state.shocked = True
        st.session_state.mechanism = mechanism
        st.session_state.shock_type = shock_type
        st.session_state.magnitude = magnitude
    
    if st.session_state.shocked:
        col_mec, col_sim = st.columns([1, 1])
        
        with col_mec:
            st.subheader("🔄 Mecanismo de Transmisión")
            for step in st.session_state.mechanism:
                st.markdown(f"• {step}")
        
        with col_sim:
            st.subheader("📈 Simulación Temporal")
            
            if st.button("Ejecutar simulación dinámica"):
                simulator = DynamicSimulator(model)
                
                with st.spinner("Simulando trayectoria temporal..."):
                    df_sim = simulator.time_series_simulation(
                        st.session_state.shock_type, 
                        st.session_state.magnitude,
                        periods=20,
                        persistence=persistence
                    )
                    
                    st.session_state.simulation_df = df_sim
                    
                    # Mostrar gráfico
                    fig = plot_time_series_simulation(df_sim, 'PIB')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar tabla
                    st.dataframe(df_sim.round(2), use_container_width=True)
        
        # IRF
        st.markdown("---")
        st.subheader("📊 Funciones Impulso-Respuesta (IRF)")
        
        if st.button("Calcular IRF"):
            simulator = DynamicSimulator(model)
            irf = simulator.impulse_response_function(st.session_state.shock_type, 
                                                      st.session_state.magnitude)
            
            fig = plot_irf(irf, ['Y', 'i', 'π'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de convergencia
            convergence = simulator.convergence_dynamics('Y')
            st.info(f"⚡ Velocidad de convergencia: media vida de {convergence['half_life_periods']} periodos")

# Tab 3: Comparador de escuelas
with tab3:
    st.subheader("🏛️ Comparación entre Escuelas para un Mismo Choque")
    
    if st.button("Comparar escuelas", key="compare_schools"):
        # Crear diferentes modelos
        schools = {
            "Keynesiano": ISLM(settings),
            "Clásico": ADASClassical(settings),
            "Nuevo Keynesiano": NewKeynesian(settings)
        }
        
        results_comparison = {}
        
        for name, school_model in schools.items():
            # Resolver estado inicial
            school_model.solve()
            
            # Aplicar choque
            if hasattr(school_model, 'apply_shock'):
                school_model.apply_shock(shock_type, magnitude)
                eq_new = school_model.solve()
                
                results_comparison[name] = {
                    "PIB_CP": eq_new['Y'],
                    "PIB_LP": school_model.steady_state()['Y'] if hasattr(school_model, 'steady_state') else eq_new['Y'],
                    "Inflación_CP": eq_new.get('π', 0.02),
                    "Empleo_CP": eq_new.get('Y', 100) / 100  # Proxy
                }
        
        # Mostrar tabla comparativa
        df_comparison = pd.DataFrame(results_comparison).T
        st.dataframe(df_comparison.round(3), use_container_width=True)
        
        # Gráfico comparativo
        fig = plot_comparative_schools(results_comparison, 'PIB')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretación:**
        - **Keynesiano**: Efectos reales persistentes, ajuste lento de precios
        - **Clásico**: Neutralidad monetaria, solo efectos nominales
        - **Nuevo Keynesiano**: Efectos reales en CP pero neutralidad en LP con expectativas racionales
        """)

# Tab 4: Modelo de Solow - Crecimiento
with tab4:
    st.subheader("📈 Modelo de Crecimiento de Solow")
    
    solow_model = Solow(settings)
    ss = solow_model.steady_state()
    
    col_ss1, col_ss2, col_ss3 = st.columns(3)
    with col_ss1:
        st.metric("Capital per cápita (k*)", f"{ss['k_ss']:.3f}")
    with col_ss2:
        st.metric("Producto per cápita (y*)", f"{ss['y_ss']:.3f}")
    with col_ss3:
        st.metric("Consumo per cápita (c*)", f"{ss['c_ss']:.3f}")
    
    # Dinámica de convergencia
    path = solow_model.time_path(periods=100)
    fig = plot_solow_dynamics(solow_model)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regla de oro
    st.markdown("---")
    st.subheader("💰 Regla de Oro")
    
    s_gold = ss['α']  # α = elasticidad capital
    st.latex(r"s_{oro} = \alpha = " + f"{ss['α']}")
    st.info(f"La tasa de ahorro que maximiza el consumo en estado estacionario es {s_gold:.3f}. "
            f"Tasa actual: {solow_model.params['s']:.3f}")

# Tab 5: Laboratorio Econométrico
with tab5:
    st.subheader("🔬 Integración Econométrica")
    
    col_lab1, col_lab2 = st.columns([1, 1])
    
    with col_lab1:
        if st.button("Cargar datos económicos (simulados)", key="load_data"):
            lab = EconometricLab()
            data = lab.load_real_data(source='simulated')
            st.session_state.econ_data = data
            st.success("✅ Datos cargados: 200 observaciones trimestrales")
            
            st.dataframe(data.tail(), use_container_width=True)
    
    with col_lab2:
        if 'econ_data' in st.session_state:
            selected_var = st.selectbox("Variable a modelar", 
                                        st.session_state.econ_data.columns)
            
            if st.button("Estimar ARIMA", key="estimate_arima"):
                lab = EconometricLab()
                lab.data = st.session_state.econ_data
                results = lab.estimate_arima(selected_var)
                
                st.session_state.arima_results = results
                st.success(f"✅ ARIMA{results['order']} estimado | AIC: {results['aic']:.2f}")
                
                # Gráfico de pronóstico
                fig = plot_forecast(results, selected_var)
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Estimar VAR y IRF", key="estimate_var"):
                lab = EconometricLab()
                lab.data = st.session_state.econ_data
                vars_list = [selected_var, 'interest', 'inflation']
                vars_list = [v for v in vars_list if v in lab.data.columns]
                
                results = lab.estimate_var(vars_list)
                st.success("✅ VAR estimado exitosamente")
                
                st.subheader("Matriz de coeficientes")
                st.write(results['coeffs'])
                
                if st.button("Calcular multiplicador fiscal"):
                    mult = lab.fiscal_multiplier_estimation()
                    st.metric("Multiplicador fiscal estimado", f"{mult['multiplier']:.3f}")
                    st.info("Multiplicador > 1 indica efecto expansivo neto")

# Aplicar choque desde cualquier tab
if apply_shock:
    with st.spinner("Aplicando choque..."):
        engine = ShockEngine(model)
        new_eq, mechanism = engine.apply(shock_type, magnitude)
        st.session_state.current_eq = new_eq
        st.session_state.shocked = True
        st.session_state.mechanism = mechanism
        st.session_state.shock_type = shock_type
        st.session_state.magnitude = magnitude
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
**SICM v1.0 - Simulador Integral de Choques Macroeconómicos**  
*Desarrollado con Python, Streamlit, Plotly, Statsmodels y economía computacional*
""")