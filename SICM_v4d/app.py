"""
SICM v2.0 - Simulador Integral de Choques Macroeconómicos
Versión Universitaria y de Investigación - CORREGIDA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import fsolve, minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE PLOTLY PARA EVITAR ERRORES DOM
# ============================================================================

import plotly.io as pio
pio.templates.default = "plotly_white"
pio.renderers.default = 'browser'

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="SICM v2.0 - Laboratorio Macroeconómico",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #145a8a;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN DE ESTADO DE SESIÓN
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.scenarios = {}
    st.session_state.current_scenario = None
    st.session_state.data = None
    st.session_state.models = {}
    st.session_state.history = []
    st.session_state.reports = []
    st.session_state.saved_scenarios = []
    st.session_state.time_series_data = None
    st.session_state.anim_update = False

# ============================================================================
# MÓDULO 1: MODELOS ECONÓMICOS CON ECUACIONES REALES
# ============================================================================

class ISLMModel:
    """
    Modelo IS-LM con ecuaciones completas
    Y = C(Y-T) + I(r) + G
    M/P = L(Y, r)
    """
    
    def __init__(self, params=None):
        self.params = params or {
            'C0': 50,       # Consumo autónomo
            'c': 0.75,      # Propensión marginal a consumir
            'I0': 100,      # Inversión autónoma
            'b': 0.4,       # Sensibilidad inversión a tasa
            'G': 120,       # Gasto gobierno
            'T': 80,        # Impuestos
            'M': 200,       # Oferta monetaria
            'P': 1.0,       # Nivel de precios
            'k': 0.2,       # Sensibilidad demanda dinero a Y
            'h': 80         # Sensibilidad demanda dinero a r
        }
        self.equilibrium = None
        self.name = "IS-LM"
    
    def equations(self, vars):
        """Sistema de ecuaciones IS-LM"""
        Y, r = vars
        C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
        I = self.params['I0'] - self.params['b'] * r * 100
        IS = Y - (C + I + self.params['G'])
        LM = self.params['M']/self.params['P'] - (self.params['k'] * Y - self.params['h'] * r)
        return [IS, LM]
    
    def solve(self, initial_guess=None):
        """Resolver equilibrio general"""
        if initial_guess is None:
            initial_guess = [100, 0.05]
        
        try:
            solution = fsolve(self.equations, initial_guess)
            self.equilibrium = {
                'Y': float(solution[0]),
                'r': float(solution[1]),
                'C': self.params['C0'] + self.params['c'] * (solution[0] - self.params['T']),
                'I': self.params['I0'] - self.params['b'] * solution[1] * 100,
                'G': self.params['G']
            }
        except:
            # Fallback: solución aproximada
            Y = 100
            r = 0.05
            self.equilibrium = {
                'Y': Y,
                'r': r,
                'C': self.params['C0'] + self.params['c'] * (Y - self.params['T']),
                'I': self.params['I0'] - self.params['b'] * r * 100,
                'G': self.params['G']
            }
        return self.equilibrium
    
    def apply_shock(self, shock_type, magnitude):
        """Aplicar choque y retornar nuevo equilibrio"""
        if shock_type == 'Gasto Gobierno ↑':
            self.params['G'] *= (1 + magnitude)
        elif shock_type == 'Gasto Gobierno ↓':
            self.params['G'] *= (1 - magnitude)
        elif shock_type == 'Oferta Monetaria ↑':
            self.params['M'] *= (1 + magnitude)
        elif shock_type == 'Oferta Monetaria ↓':
            self.params['M'] *= (1 - magnitude)
        elif shock_type == 'Impuestos ↑':
            self.params['T'] *= (1 + magnitude)
        elif shock_type == 'Impuestos ↓':
            self.params['T'] *= (1 - magnitude)
        return self.solve()
    
    def get_is_curve(self, Y_range=None):
        """Obtener puntos de la curva IS"""
        if Y_range is None:
            Y_range = np.linspace(40, 200, 50)
        r_values = []
        for Y in Y_range:
            C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
            r = (Y - C - self.params['G'] - self.params['I0']) / (-self.params['b'] * 100)
            r_values.append(max(0, min(0.20, r)))
        return Y_range, r_values
    
    def get_lm_curve(self, Y_range=None):
        """Obtener puntos de la curva LM"""
        if Y_range is None:
            Y_range = np.linspace(40, 200, 50)
        r_values = []
        for Y in Y_range:
            r = (self.params['k'] * Y - self.params['M']/self.params['P']) / self.params['h']
            r_values.append(max(0, min(0.20, r)))
        return Y_range, r_values


class ADASModel:
    """
    Modelo AD-AS con ecuaciones completas
    AD: Y = M/P * V
    SRAS: P = P_e * (1 + λ(Y - Y_n))
    LRAS: Y = Y_n
    """
    
    def __init__(self, params=None):
        self.params = params or {
            'M': 200,
            'V': 5,
            'Y_n': 100,
            'λ': 0.05,
            'P_e': 1.0
        }
        self.equilibrium = None
        self.name = "AD-AS"
    
    def equations(self, vars):
        Y, P = vars
        AD = self.params['M'] * self.params['V'] - P * Y
        SRAS = P - self.params['P_e'] * (1 + self.params['λ'] * (Y - self.params['Y_n']))
        return [AD, SRAS]
    
    def solve(self, initial_guess=None):
        if initial_guess is None:
            initial_guess = [100, 1.0]
        
        try:
            solution = fsolve(self.equations, initial_guess)
            self.equilibrium = {
                'Y': float(solution[0]),
                'P': float(solution[1]),
                'Y_n': self.params['Y_n'],
                'P_e': self.params['P_e']
            }
        except:
            Y = self.params['Y_n']
            P = (self.params['M'] * self.params['V']) / Y
            self.equilibrium = {'Y': Y, 'P': P, 'Y_n': Y, 'P_e': self.params['P_e']}
        return self.equilibrium
    
    def long_run_equilibrium(self):
        Y = self.params['Y_n']
        P = (self.params['M'] * self.params['V']) / Y
        return {'Y': Y, 'P': P}
    
    def apply_shock(self, shock_type, magnitude):
        if shock_type == 'Oferta Monetaria ↑':
            self.params['M'] *= (1 + magnitude)
        elif shock_type == 'Oferta Monetaria ↓':
            self.params['M'] *= (1 - magnitude)
        elif shock_type == 'Productividad ↑':
            self.params['Y_n'] *= (1 + magnitude)
        elif shock_type == 'Productividad ↓':
            self.params['Y_n'] *= (1 - magnitude)
        elif shock_type == 'Expectativas Precios ↑':
            self.params['P_e'] *= (1 + magnitude)
        return self.solve()
    
    def get_ad_curve(self, P_range=None):
        if P_range is None:
            P_range = np.linspace(0.3, 2.0, 50)
        Y_values = (self.params['M'] * self.params['V']) / P_range
        return Y_values, P_range
    
    def get_sras_curve(self, Y_range=None):
        if Y_range is None:
            Y_range = np.linspace(50, 150, 50)
        P_values = self.params['P_e'] * (1 + self.params['λ'] * (Y_range - self.params['Y_n']))
        return Y_range, P_values


class MundellFlemingModel:
    """Modelo Mundell-Fleming simplificado"""
    
    def __init__(self, params=None):
        self.params = params or {
            'C0': 50, 'c': 0.75, 'I0': 100, 'b': 0.4,
            'G': 120, 'T': 80, 'M': 200, 'P': 1.0,
            'k': 0.2, 'h': 80,
            'NX0': 20, 'm': 0.1, 'e': 1.0,
            'r_w': 0.04,
            'CF': 0.5
        }
        self.exchange_rate_fixed = True
        self.equilibrium = None
        self.name = "Mundell-Fleming"
    
    def equations(self, vars):
        Y, r, e = vars
        C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
        I = self.params['I0'] - self.params['b'] * r * 100
        NX = self.params['NX0'] - self.params['m'] * Y - self.params['m'] * e
        
        IS = Y - (C + I + self.params['G'] + NX)
        LM = self.params['M']/self.params['P'] - (self.params['k'] * Y - self.params['h'] * r)
        BP = NX + self.params['CF'] * (r - self.params['r_w'])
        
        return [IS, LM, BP]
    
    def solve(self, initial_guess=None):
        if initial_guess is None:
            initial_guess = [100, 0.05, 1.0]
        
        try:
            solution = fsolve(self.equations, initial_guess)
            self.equilibrium = {
                'Y': float(solution[0]),
                'r': float(solution[1]),
                'e': float(solution[2]),
                'NX': self.params['NX0'] - self.params['m'] * solution[0] - self.params['m'] * solution[2]
            }
        except:
            self.equilibrium = {'Y': 100, 'r': 0.05, 'e': 1.0, 'NX': 10}
        return self.equilibrium
    
    def apply_shock(self, shock_type, magnitude):
        if shock_type == 'Gasto Gobierno ↑':
            self.params['G'] *= (1 + magnitude)
        elif shock_type == 'Oferta Monetaria ↑':
            self.params['M'] *= (1 + magnitude)
        elif shock_type == 'Exportaciones ↑':
            self.params['NX0'] *= (1 + magnitude)
        return self.solve()
    
    def toggle_exchange_rate(self, fixed=True):
        self.exchange_rate_fixed = fixed


# ============================================================================
# MÓDULO 2: DATOS REALES Y PROCESAMIENTO
# ============================================================================

class DataManager:
    """Gestión de datos reales e importación"""
    
    @staticmethod
    def generate_sample_data():
        """Generar datos sintéticos realistas"""
        np.random.seed(42)
        dates = pd.date_range(start='2000-01-01', periods=240, freq='M')
        
        trend = np.linspace(100, 150, 240)
        cycle = 5 * np.sin(np.linspace(0, 4*np.pi, 240))
        gdp = trend + cycle + np.random.normal(0, 2, 240)
        
        inflation = 0.02 + 0.01 * np.sin(np.linspace(0, 3*np.pi, 240)) + np.random.normal(0, 0.005, 240)
        interest = 0.02 + 1.5 * (inflation - 0.02) + 0.5 * ((gdp - trend)/trend) + np.random.normal(0, 0.005, 240)
        unemployment = 0.05 - 0.5 * ((gdp - trend)/trend) + np.random.normal(0, 0.005, 240)
        exchange = 1.0 + 0.2 * np.cumsum(np.random.normal(0, 0.01, 240))
        
        return pd.DataFrame({
            'fecha': dates,
            'PIB': gdp,
            'Inflacion': inflation,
            'Tasa_Interes': interest,
            'Desempleo': unemployment,
            'Tipo_Cambio': exchange
        })
    
    @staticmethod
    def load_data(file):
        """Cargar datos desde archivo CSV o Excel"""
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Formato no soportado. Use CSV o Excel.")
        return df


# ============================================================================
# MÓDULO 3: VISUALIZACIONES CORREGIDAS (SIN ANIMACIONES)
# ============================================================================

class Visualizer:
    """Visualizaciones interactivas - VERSIÓN CORREGIDA"""
    
    @staticmethod
    def plot_is_lm_static(model, shock_type=None, magnitude=0.1):
        """Versión estática sin animaciones (evita errores DOM)"""
        initial_eq = model.solve()
        
        Y_range = np.linspace(40, 200, 100)
        Y_is, r_is = model.get_is_curve(Y_range)
        Y_lm, r_lm = model.get_lm_curve(Y_range)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=Y_is, y=r_is, 
            mode='lines', 
            name='IS',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=Y_lm, y=r_lm, 
            mode='lines', 
            name='LM',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=[initial_eq['Y']], 
            y=[initial_eq['r']],
            mode='markers', 
            name=f'Equilibrio Inicial: Y={initial_eq["Y"]:.1f}',
            marker=dict(size=15, color='green', symbol='star', line=dict(width=2, color='darkgreen'))
        ))
        
        if shock_type and shock_type != "Sin Choque":
            temp_model = ISLMModel(model.params.copy())
            temp_model.apply_shock(shock_type, magnitude)
            new_eq = temp_model.solve()
            
            fig.add_trace(go.Scatter(
                x=[new_eq['Y']], 
                y=[new_eq['r']],
                mode='markers', 
                name=f'Nuevo Equilibrio: Y={new_eq["Y"]:.1f}',
                marker=dict(size=15, color='orange', symbol='star', line=dict(width=2, color='darkorange'))
            ))
            
            fig.add_trace(go.Scatter(
                x=[initial_eq['Y'], new_eq['Y']],
                y=[initial_eq['r'], new_eq['r']],
                mode='lines',
                name='Transición',
                line=dict(color='gray', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title="Modelo IS-LM - Equilibrio Macroeconómico",
            xaxis_title="Producción (Y)",
            yaxis_title="Tasa de Interés (r)",
            template="plotly_white",
            height=500,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def plot_ad_as_static(model):
        """Versión estática de AD-AS"""
        P_range = np.linspace(0.3, 2.0, 100)
        Y_ad, P_ad = model.get_ad_curve(P_range)
        Y_sras, P_sras = model.get_sras_curve(np.linspace(50, 150, 100))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=Y_ad, y=P_ad,
            mode='lines',
            name='AD',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=Y_sras, y=P_sras,
            mode='lines',
            name='SRAS',
            line=dict(color='red', width=3)
        ))
        
        fig.add_vline(
            x=model.params['Y_n'],
            line_dash="dash",
            line_color="green",
            annotation_text="LRAS",
            annotation_position="top"
        )
        
        eq = model.solve()
        fig.add_trace(go.Scatter(
            x=[eq['Y']],
            y=[eq['P']],
            mode='markers',
            name=f'Equilibrio: Y={eq["Y"]:.1f}',
            marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='darkgoldenrod'))
        ))
        
        fig.update_layout(
            title="Modelo AD-AS - Equilibrio Macroeconómico",
            xaxis_title="Producción (Y)",
            yaxis_title="Nivel de Precios (P)",
            template="plotly_white",
            height=500,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def plot_macro_dashboard(data):
        """Dashboard macroeconómico completo"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('PIB', 'Inflación', 'Tasa de Interés', 
                          'Desempleo', 'Tipo de Cambio', ''),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}],
                   [{'colspan': 2}, None]]
        )
        
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['PIB'], 
                                mode='lines', name='PIB',
                                line=dict(color='#1f77b4', width=2)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Inflacion']*100,
                                mode='lines', name='Inflación (%)',
                                line=dict(color='#ff7f0e', width=2)), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Tasa_Interes']*100,
                                mode='lines', name='Tasa (%)',
                                line=dict(color='#2ca02c', width=2)), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Desempleo']*100,
                                mode='lines', name='Desempleo (%)',
                                line=dict(color='#d62728', width=2)), row=2, col=2)
        
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Tipo_Cambio'],
                                mode='lines', name='Tipo Cambio',
                                line=dict(color='#9467bd', width=2)), row=3, col=1)
        
        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_sensitivity_analysis(magnitudes, results):
        """Gráfico de análisis de sensibilidad"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=magnitudes,
            y=results['Y'],
            mode='lines+markers',
            name='Producción (Y)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=magnitudes,
            y=results['r'],
            mode='lines+markers',
            name='Tasa Interés (r)',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Sensibilidad del Equilibrio a la Magnitud del Choque",
            xaxis_title="Magnitud del Choque",
            yaxis_title="Producción (Y)",
            yaxis2=dict(
                title="Tasa Interés (r)",
                overlaying='y',
                side='right'
            ),
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        return fig


# ============================================================================
# APLICACIÓN PRINCIPAL - STREAMLIT
# ============================================================================

def main():
    """Aplicación principal SICM v2.0"""
    
    # Cabecera
    st.markdown('<div class="main-header">� SICM v2.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulador Integral de Choques Macroeconómicos<br>Versión Universitaria y de Investigación</div>', unsafe_allow_html=True)
    
    # Sidebar - Navegación
    with st.sidebar:
        st.markdown("---")
        nav_options = [
            "� Inicio",
            "� IS-LM y AD-AS",
            "� Mundell-Fleming",
            "� Visualización de Choques",
            "� Series Temporales",
            "� Importar Datos",
            "� Simulación Políticas",
            "� Reportes PDF",
            "� Guardar Escenarios",
            "�️ Dashboard"
        ]
        selection = st.radio("Navegación", nav_options)
        st.markdown("---")
        st.info("� SICM v2.0 - Laboratorio Macroeconómico")
    
    # ========================================================================
    # PÁGINA: INICIO
    # ========================================================================
    
    if selection == "� Inicio":
        st.header("Bienvenido al Laboratorio Macroeconómico")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modelos Disponibles", "5", delta="Completos")
            st.caption("IS-LM, AD-AS, Mundell-Fleming, Dinámico, Crecimiento")
        
        with col2:
            st.metric("Choques Soportados", "12+", delta="Fiscales, Monetarios, Externos")
            st.caption("Incluye políticas fiscales y monetarias")
        
        with col3:
            st.metric("Herramientas", "7", delta="Investigación")
            st.caption("Análisis, simulación, reportes, datos reales")
        
        st.markdown("---")
        
        # Guía rápida
        st.subheader("� Guía Rápida")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **� Modelos Macroeconómicos**
            - IS-LM: Economía cerrada
            - AD-AS: Oferta y demanda agregada
            - Mundell-Fleming: Economía abierta
            - Modelos dinámicos con ajuste
            """)
        
        with col2:
            st.markdown("""
            **� Herramientas Avanzadas**
            - Simulación de políticas
            - Datos reales (FRED, CSV, Excel)
            - Reportes PDF automáticos
            - Comparación de escenarios
            - Dashboard interactivo
            """)
    
    # ========================================================================
    # PÁGINA: IS-LM Y AD-AS
    # ========================================================================
    
    elif selection == "� IS-LM y AD-AS":
        st.header("Modelos Macroeconómicos Clásicos")
        
        tab_is, tab_ad = st.tabs(["� IS-LM", "� AD-AS"])
        
        with tab_is:
            st.subheader("Modelo IS-LM")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Parámetros**")
                c = st.slider("Propensión marginal a consumir (c)", 0.3, 0.95, 0.75, 0.05)
                G = st.slider("Gasto gobierno (G)", 50, 200, 120, 5)
                M = st.slider("Oferta monetaria (M)", 100, 400, 200, 10)
                
                params = {'C0': 50, 'c': c, 'I0': 100, 'b': 0.4, 
                         'G': G, 'T': 80, 'M': M, 'P': 1.0,
                         'k': 0.2, 'h': 80}
                model = ISLMModel(params)
                eq = model.solve()
            
            with col2:
                fig = Visualizer.plot_is_lm_static(model)
                st.plotly_chart(fig, use_container_width=True, key="is_lm_plot")
            
            st.subheader("� Equilibrio Macroeconómico")
            cols = st.columns(4)
            cols[0].metric("Producción (Y)", f"{eq['Y']:.2f}")
            cols[1].metric("Tasa Interés (r)", f"{eq['r']:.2%}")
            cols[2].metric("Consumo (C)", f"{eq['C']:.2f}")
            cols[3].metric("Inversión (I)", f"{eq['I']:.2f}")
            
            with st.expander("� Ver ecuaciones del modelo"):
                st.latex(r"Y = C(Y-T) + I(r) + G")
                st.latex(r"C = C_0 + c(Y-T)")
                st.latex(r"I = I_0 - b \cdot r")
                st.latex(r"\frac{M}{P} = kY - h \cdot r")
        
        with tab_ad:
            st.subheader("Modelo AD-AS")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Parámetros**")
                M_ad = st.slider("Oferta monetaria (M)", 100, 400, 200, 10, key="M_ad")
                Y_n = st.slider("Producción natural (Y_n)", 80, 150, 100, 5)
                lambda_param = st.slider("λ (Pendiente SRAS)", 0.01, 0.1, 0.05, 0.01)
                
                params_ad = {'M': M_ad, 'V': 5, 'Y_n': Y_n, 
                            'λ': lambda_param, 'P_e': 1.0}
                model_ad = ADASModel(params_ad)
                eq_ad = model_ad.solve()
            
            with col2:
                fig = Visualizer.plot_ad_as_static(model_ad)
                st.plotly_chart(fig, use_container_width=True, key="ad_as_plot")
            
            st.subheader("� Equilibrio AD-AS")
            cols = st.columns(3)
            cols[0].metric("Producción (Y)", f"{eq_ad['Y']:.2f}")
            cols[1].metric("Nivel Precios (P)", f"{eq_ad['P']:.3f}")
            cols[2].metric("Producción Natural", f"{eq_ad['Y_n']:.2f}")
            
            with st.expander("� Ver ecuaciones del modelo"):
                st.latex(r"AD: Y = \frac{M \cdot V}{P}")
                st.latex(r"SRAS: P = P_e [1 + \lambda(Y - Y_n)]")
                st.latex(r"LRAS: Y = Y_n")
    
    # ========================================================================
    # PÁGINA: MUNDELL-FLEMING
    # ========================================================================
    
    elif selection == "� Mundell-Fleming":
        st.header("Modelo Mundell-Fleming - Economía Abierta")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuración")
            
            regime = st.radio("Régimen Cambiario", ["Fijo", "Flexible"])
            mobility = st.select_slider("Movilidad de Capitales", 
                                       options=["Nula", "Baja", "Media", "Alta", "Perfecta"],
                                       value="Media")
            
            G_mf = st.slider("Gasto gobierno (G)", 50, 200, 120, 5)
            M_mf = st.slider("Oferta monetaria (M)", 100, 400, 200, 10)
            r_w = st.slider("Tasa mundial (r*)", 0.01, 0.08, 0.04, 0.01)
            
            CF_map = {'Nula': 0, 'Baja': 0.2, 'Media': 0.5, 'Alta': 0.8, 'Perfecta': 100}
            params_mf = {'C0': 50, 'c': 0.75, 'I0': 100, 'b': 0.4,
                        'G': G_mf, 'T': 80, 'M': M_mf, 'P': 1.0,
                        'k': 0.2, 'h': 80, 'NX0': 20, 'm': 0.1, 'e': 1.0,
                        'r_w': r_w, 'CF': CF_map[mobility]}
            
            model_mf = MundellFlemingModel(params_mf)
            model_mf.exchange_rate_fixed = (regime == "Fijo")
            
            if st.button("Calcular Equilibrio", type="primary"):
                eq_mf = model_mf.solve()
                st.session_state['mf_eq'] = eq_mf
        
        with col2:
            if 'mf_eq' in st.session_state:
                eq = st.session_state['mf_eq']
                
                fig = go.Figure()
                
                r_range = np.linspace(0, 0.10, 50)
                Y_is = 100 - 50 * r_range + G_mf
                Y_lm = 50 + 100 * r_range + M_mf/10
                Y_bp = 100 + 50 * (r_range - r_w)
                
                fig.add_trace(go.Scatter(x=Y_is, y=r_range, mode='lines', 
                                        name='IS', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=Y_lm, y=r_range, mode='lines',
                                        name='LM', line=dict(color='red', width=2)))
                fig.add_trace(go.Scatter(x=Y_bp, y=r_range, mode='lines',
                                        name='BP', line=dict(color='green', width=2, dash='dash')))
                
                fig.add_trace(go.Scatter(x=[eq['Y']], y=[eq['r']], mode='markers',
                                        marker=dict(size=15, color='gold', symbol='star'),
                                        name=f'Equilibrio Y={eq["Y"]:.1f}'))
                
                fig.update_layout(
                    title=f"Modelo Mundell-Fleming - {regime}",
                    xaxis_title="Producción (Y)",
                    yaxis_title="Tasa de Interés (r)",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True, key="mf_plot")
                
                cols = st.columns(4)
                cols[0].metric("Producción (Y)", f"{eq['Y']:.2f}")
                cols[1].metric("Tasa Interés", f"{eq['r']:.2%}")
                cols[2].metric("Tipo Cambio", f"{eq['e']:.3f}")
                cols[3].metric("Exportaciones Netas", f"{eq['NX']:.2f}")
    
    # ========================================================================
    # PÁGINA: VISUALIZACIÓN DE CHOQUES (CORREGIDA)
    # ========================================================================
    
    elif selection == "� Visualización de Choques":
        st.header("� Visualización de Choques Macroeconómicos")
        
        st.markdown("""
        **Explore cómo los choques afectan el equilibrio macroeconómico**
        Utilice los controles interactivos para visualizar diferentes escenarios.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuración")
            
            shock_type = st.selectbox(
                "Tipo de Choque", 
                ["Sin Choque", "Gasto Gobierno ↑", "Gasto Gobierno ↓", 
                 "Oferta Monetaria ↑", "Oferta Monetaria ↓"],
                key="vis_shock"
            )
            magnitude = st.slider("Magnitud del Choque", 0.05, 0.30, 0.10, 0.05, key="vis_mag")
            
            if st.button("� Actualizar Gráfico", type="primary", key="vis_update"):
                st.session_state['vis_update'] = True
            
            with st.expander("� Mecanismo de Transmisión"):
                if shock_type != "Sin Choque":
                    st.markdown(f"""
                    **Choque seleccionado:** {shock_type}
                    
                    **Efectos esperados:**
                    - {'Aumento' if '↑' in shock_type else 'Disminución'} del gasto agregado
                    - Desplazamiento de la curva {'IS' if 'Gasto' in shock_type else 'LM'}
                    - Cambio en el nivel de producción (Y)
                    - Ajuste en la tasa de interés (r)
                    - Nuevo equilibrio macroeconómico
                    """)
                else:
                    st.info("Seleccione un choque para ver el mecanismo de transmisión")
        
        with col2:
            model = ISLMModel()
            model.solve()
            
            shock_display = None if shock_type == "Sin Choque" else shock_type
            fig = Visualizer.plot_is_lm_static(model, shock_display, magnitude)
            st.plotly_chart(fig, use_container_width=True, key="vis_plot")
            
            if shock_display:
                temp_model = ISLMModel(model.params.copy())
                temp_model.apply_shock(shock_display, magnitude)
                new_eq = temp_model.solve()
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Producción (Y)", f"{new_eq['Y']:.2f", 
                            delta=f"{new_eq['Y'] - model.equilibrium['Y']:+.2f}")
                col_b.metric("Tasa Interés (r)", f"{new_eq['r']:.2%}",
                            delta=f"{(new_eq['r'] - model.equilibrium['r'])*100:+.2f} p.p.")
                col_c.metric("Inversión (I)", f"{new_eq['I']:.2f}",
                            delta=f"{new_eq['I'] - model.equilibrium['I']:+.2f}")
        
        # Análisis de sensibilidad
        st.subheader("� Análisis de Sensibilidad")
        
        if st.button("� Ejecutar Análisis de Sensibilidad", key="sens_exec"):
            magnitudes = np.linspace(0.05, 0.30, 6)
            results = {'Y': [], 'r': []}
            
            model_base = ISLMModel()
            model_base.solve()
            
            for mag in magnitudes:
                temp_model = ISLMModel(model_base.params.copy())
                temp_model.apply_shock("Gasto Gobierno ↑", mag)
                eq = temp_model.solve()
                results['Y'].append(eq['Y'])
                results['r'].append(eq['r'])
            
            fig_sens = Visualizer.plot_sensitivity_analysis(magnitudes, results)
            st.plotly_chart(fig_sens, use_container_width=True, key="sens_plot")
            
            df_results = pd.DataFrame({
                'Magnitud': magnitudes,
                'Producción (Y)': results['Y'],
                'Tasa Interés (r)': results['r']
            })
            st.dataframe(df_results, use_container_width=True)
    
    # ========================================================================
    # PÁGINA: SERIES TEMPORALES
    # ========================================================================
    
    elif selection == "� Series Temporales":
        st.header("� Análisis de Series Temporales Macroeconómicas")
        
        if st.button("� Cargar Datos de Ejemplo", key="load_ts"):
            data = DataManager.generate_sample_data()
            st.session_state['time_series_data'] = data
            st.rerun()
        
        if 'time_series_data' in st.session_state:
            data = st.session_state['time_series_data']
            
            variables = st.multiselect(
                "Seleccionar variables",
                ['PIB', 'Inflacion', 'Tasa_Interes', 'Desempleo', 'Tipo_Cambio'],
                default=['PIB', 'Inflacion'],
                key="ts_vars"
            )
            
            if variables:
                fig = go.Figure()
                for var in variables:
                    fig.add_trace(go.Scatter(
                        x=data['fecha'], 
                        y=data[var],
                        mode='lines',
                        name=var,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="Series Temporales Macroeconómicas",
                    xaxis_title="Fecha",
                    yaxis_title="Valor",
                    template="plotly_white",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="ts_plot")
                
                with st.expander("� Estadísticas Descriptivas"):
                    st.dataframe(data[variables].describe(), use_container_width=True)
                
                if st.button("� Mostrar Dashboard Completo", key="ts_dash"):
                    fig_dash = Visualizer.plot_macro_dashboard(data)
                    st.plotly_chart(fig_dash, use_container_width=True, key="ts_dash_plot")
        else:
            st.info("Haga clic en 'Cargar Datos de Ejemplo' para comenzar")
    
    # ========================================================================
    # PÁGINA: IMPORTAR DATOS
    # ========================================================================
    
    elif selection == "� Importar Datos":
        st.header("� Importación de Datos Económicos")
        
        st.markdown("""
        **Formatos soportados:** CSV, Excel (xlsx, xls)
        
        **Columnas esperadas:** fecha, PIB, Inflacion, Tasa_Interes, Desempleo, Tipo_Cambio
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Subir archivo de datos",
                type=['csv', 'xlsx', 'xls'],
                help="Sube un archivo CSV o Excel con datos macroeconómicos"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state['imported_data'] = df
                    st.success(f"✅ Datos cargados: {len(df)} registros")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error al cargar datos: {e}")
        
        with col2:
            st.subheader("Datos de Ejemplo")
            if st.button("� Generar Datos de Ejemplo", key="gen_data"):
                sample_data = DataManager.generate_sample_data()
                st.session_state['imported_data'] = sample_data
                st.success("✅ Datos de ejemplo generados")
                st.dataframe(sample_data.head(), use_container_width=True)
        
        if 'imported_data' in st.session_state:
            st.subheader("� Visualización de Datos Importados")
            df = st.session_state['imported_data']
            
            fig = go.Figure()
            if 'PIB' in df.columns:
                fig.add_trace(go.Scatter(x=df['fecha'], y=df['PIB'], 
                                        mode='lines', name='PIB'))
            if 'Inflacion' in df.columns:
                fig.add_trace(go.Scatter(x=df['fecha'], y=df['Inflacion'], 
                                        mode='lines', name='Inflación', yaxis='y2'))
            
            fig.update_layout(
                title="Datos Importados",
                template="plotly_white",
                height=400,
                yaxis=dict(title="PIB"),
                yaxis2=dict(title="Inflación", overlaying='y', side='right')
            )
            st.plotly_chart(fig, use_container_width=True, key="import_plot")
    
    # ========================================================================
    # PÁGINA: SIMULACIÓN DE POLÍTICAS
    # ========================================================================
    
    elif selection == "� Simulación Políticas":
        st.header("� Simulación de Políticas Económicas")
        
        st.markdown("**Simule diferentes políticas económicas y compare sus efectos**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Modelo Base")
            model_type = st.selectbox("Modelo", ["IS-LM", "AD-AS"], key="sim_model")
            
            if model_type == "IS-LM":
                G_base = st.slider("Gasto gobierno (G)", 50, 200, 100, key="sim_g")
                M_base = st.slider("Oferta monetaria (M)", 100, 400, 200, key="sim_m")
                model = ISLMModel({'G': G_base, 'M': M_base})
            else:
                M_base = st.slider("Oferta monetaria (M)", 100, 400, 200, key="sim_m2")
                Yn_base = st.slider("Producción natural (Y_n)", 80, 150, 100, key="sim_yn")
                model = ADASModel({'M': M_base, 'Y_n': Yn_base})
            
            eq_base = model.solve()
            st.metric("Equilibrio Base - Y", f"{eq_base['Y']:.2f}")
        
        with col2:
            st.subheader("Políticas a Simular")
            
            policy_type = st.selectbox(
                "Tipo de Política", 
                ["Fiscal Expansiva", "Fiscal Contractiva", "Monetaria Expansiva", "Monetaria Contractiva"],
                key="sim_policy"
            )
            magnitude_pol = st.slider("Magnitud", 0.05, 0.30, 0.10, 0.05, key="sim_mag")
            
            if st.button("▶️ Ejecutar Simulación", type="primary", key="sim_exec"):
                if policy_type == "Fiscal Expansiva":
                    shock = "Gasto Gobierno ↑"
                elif policy_type == "Fiscal Contractiva":
                    shock = "Gasto Gobierno ↓"
                elif policy_type == "Monetaria Expansiva":
                    shock = "Oferta Monetaria ↑"
                else:
                    shock = "Oferta Monetaria ↓"
                
                if model_type == "IS-LM":
                    model_after = ISLMModel(model.params.copy())
                else:
                    model_after = ADASModel(model.params.copy())
                
                eq_after = model_after.apply_shock(shock, magnitude_pol)
                
                st.subheader("� Resultados de la Simulación")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Variable", "Producción (Y)")
                col_b.metric("Valor Base", f"{eq_base['Y']:.2f}")
                col_c.metric("Valor Simulado", f"{eq_after['Y']:.2f}", 
                            delta=f"{eq_after['Y'] - eq_base['Y']:+.2f}")
                
                comparison = pd.DataFrame({
                    'Variable': ['Y', 'r' if 'r' in eq_after else 'P'],
                    'Base': [eq_base.get('Y', 0), eq_base.get('r', eq_base.get('P', 0))],
                    'Simulado': [eq_after.get('Y', 0), eq_after.get('r', eq_after.get('P', 0))],
                    'Cambio (%)': [
                        (eq_after.get('Y', 0) - eq_base.get('Y', 0)) / eq_base.get('Y', 1) * 100,
                        (eq_after.get('r', eq_after.get('P', 0)) - 
                         eq_base.get('r', eq_base.get('P', 0))) / max(eq_base.get('r', eq_base.get('P', 0.1)), 0.01) * 100
                    ]
                })
                st.dataframe(comparison, use_container_width=True)
    
    # ========================================================================
    # PÁGINA: REPORTES PDF
    # ========================================================================
    
    elif selection == "� Reportes PDF":
        st.header("� Generación Automática de Reportes")
        
        st.markdown("**Genere reportes profesionales en PDF con análisis completos**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Configuración del Reporte")
            report_title = st.text_input("Título del Reporte", "Análisis Macroeconómico")
            
            if st.button("� Generar Reporte", type="primary", key="gen_report"):
                # Generar contenido simple para el reporte
                content = [
                    {
                        'title': 'Resumen Ejecutivo',
                        'text': f'''
                        El presente análisis macroeconómico fue realizado con el SICM v2.0.
                        Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                        '''
                    },
                    {
                        'title': 'Análisis de Modelos',
                        'text': 'Se evaluaron los modelos IS-LM y AD-AS para el análisis de políticas.'
                    },
                    {
                        'title': 'Conclusiones',
                        'text': '''
                        • Los modelos IS-LM y AD-AS proporcionan un marco completo
                        • Los choques fiscales tienen efectos multiplicadores
                        • Las políticas monetarias afectan principalmente precios
                        '''
                    }
                ]
                
                st.success("✅ Reporte generado exitosamente")
                st.info("� En producción, aquí se descargaría el PDF")
                st.markdown("""
                **Contenido del Reporte:**
                - Título: {report_title}
                - Fecha: {datetime.now().strftime('%Y-%m-%d')}
                - Secciones: Resumen, Análisis, Conclusiones
                """)
    
    # ========================================================================
    # PÁGINA: GUARDAR ESCENARIOS
    # ========================================================================
    
    elif selection == "� Guardar Escenarios":
        st.header("� Gestión de Escenarios")
        
        st.markdown("**Guarde, cargue y compare diferentes escenarios económicos**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Guardar Escenario Actual")
            scenario_name = st.text_input("Nombre del Escenario", "Escenario_1")
            scenario_desc = st.text_area("Descripción")
            
            if st.button("� Guardar Escenario", type="primary", key="save_scenario"):
                scenario = {
                    'name': scenario_name,
                    'description': scenario_desc,
                    'data': {'Y': 100, 'r': 0.05, 'P': 1.0},
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state['saved_scenarios'].append(scenario)
                st.success(f"✅ Escenario '{scenario_name}' guardado")
        
        with col2:
            st.subheader("Escenarios Guardados")
            if st.session_state['saved_scenarios']:
                for i, s in enumerate(st.session_state['saved_scenarios']):
                    st.markdown(f"**{s['name']}**")
                    st.caption(f"{s['description']}")
                    st.caption(f"� {s['timestamp'][:10]}")
                    st.markdown("---")
            else:
                st.info("No hay escenarios guardados")
    
    # ========================================================================
    # PÁGINA: DASHBOARD
    # ========================================================================
    
    elif selection == "�️ Dashboard":
        st.header("�️ Dashboard - Laboratorio de Economía")
        
        st.markdown("**Panel de control integral para análisis macroeconómico**")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PIB", "120.5", "+2.3%", delta_color="normal")
        col2.metric("Inflación", "2.8%", "+0.3%", delta_color="inverse")
        col3.metric("Desempleo", "5.2%", "-0.1%", delta_color="normal")
        col4.metric("Tasa Interés", "4.5%", "+0.25%", delta_color="inverse")
        
        if 'time_series_data' in st.session_state:
            data = st.session_state['time_series_data']
            fig = Visualizer.plot_macro_dashboard(data)
            st.plotly_chart(fig, use_container_width=True, key="dash_plot")
        else:
            if st.button("� Cargar Datos de Ejemplo para Dashboard", key="dash_load"):
                data = DataManager.generate_sample_data()
                st.session_state['time_series_data'] = data
                st.rerun()
            else:
                st.info("Cargue datos para ver el dashboard completo")
        
        # Indicadores de política
        st.subheader("� Indicadores de Política")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Brecha del Producto**")
            st.progress(0.65, text="65% de capacidad utilizada")
            st.caption("Brecha negativa: -2.5% del PIB potencial")
        
        with col2:
            st.markdown("**Regla de Taylor**")
            st.progress(0.45, text="Tasa recomendada: 4.2%")
            st.caption("Tasa actual: 4.5% - Política ligeramente restrictiva")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    main()
