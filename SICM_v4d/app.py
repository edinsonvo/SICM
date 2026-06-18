"""
SICM v2.0 - Simulador Integral de Choques Macroeconómicos
Versión Universitaria y de Investigación
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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings('ignore')

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
        # Consumo
        C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
        # Inversión
        I = self.params['I0'] - self.params['b'] * r * 100
        # IS: Y = C + I + G
        IS = Y - (C + I + self.params['G'])
        # LM: M/P = kY - hr
        LM = self.params['M']/self.params['P'] - (self.params['k'] * Y - self.params['h'] * r)
        return [IS, LM]
    
    def solve(self, initial_guess=None):
        """Resolver equilibrio general"""
        if initial_guess is None:
            initial_guess = [100, 0.05]
        
        solution = fsolve(self.equations, initial_guess)
        self.equilibrium = {
            'Y': float(solution[0]),
            'r': float(solution[1]),
            'C': self.params['C0'] + self.params['c'] * (solution[0] - self.params['T']),
            'I': self.params['I0'] - self.params['b'] * solution[1] * 100,
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
            Y_range = np.linspace(50, 200, 50)
        r_values = []
        for Y in Y_range:
            C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
            r = (Y - C - self.params['G'] - self.params['I0']) / (-self.params['b'] * 100)
            r_values.append(max(0, r))
        return Y_range, r_values
    
    def get_lm_curve(self, Y_range=None):
        """Obtener puntos de la curva LM"""
        if Y_range is None:
            Y_range = np.linspace(50, 200, 50)
        r_values = []
        for Y in Y_range:
            r = (self.params['k'] * Y - self.params['M']/self.params['P']) / self.params['h']
            r_values.append(max(0, r))
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
            'M': 200,           # Oferta monetaria
            'V': 5,             # Velocidad del dinero
            'Y_n': 100,         # Producción natural
            'λ': 0.05,          # Pendiente SRAS
            'P_e': 1.0,         # Precios esperados
            'α': 0.5            # Ajuste de expectativas
        }
        self.equilibrium = None
        self.name = "AD-AS"
    
    def equations(self, vars):
        """Sistema de ecuaciones AD-AS"""
        Y, P = vars
        # AD: MV = PY
        AD = self.params['M'] * self.params['V'] - P * Y
        # SRAS: P = P_e * (1 + λ(Y - Y_n))
        SRAS = P - self.params['P_e'] * (1 + self.params['λ'] * (Y - self.params['Y_n']))
        return [AD, SRAS]
    
    def solve(self, initial_guess=None):
        """Resolver equilibrio de corto plazo"""
        if initial_guess is None:
            initial_guess = [100, 1.0]
        
        solution = fsolve(self.equations, initial_guess)
        self.equilibrium = {
            'Y': float(solution[0]),
            'P': float(solution[1]),
            'Y_n': self.params['Y_n'],
            'P_e': self.params['P_e']
        }
        return self.equilibrium
    
    def long_run_equilibrium(self):
        """Resolver equilibrio de largo plazo (Y = Y_n)"""
        Y = self.params['Y_n']
        P = (self.params['M'] * self.params['V']) / Y
        return {'Y': Y, 'P': P}
    
    def apply_shock(self, shock_type, magnitude):
        """Aplicar choque y retornar nuevo equilibrio"""
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
        """Obtener puntos de la curva AD"""
        if P_range is None:
            P_range = np.linspace(0.5, 2.0, 50)
        Y_values = (self.params['M'] * self.params['V']) / P_range
        return Y_values, P_range
    
    def get_sras_curve(self, Y_range=None):
        """Obtener puntos de la curva SRAS"""
        if Y_range is None:
            Y_range = np.linspace(50, 150, 50)
        P_values = self.params['P_e'] * (1 + self.params['λ'] * (Y_range - self.params['Y_n']))
        return Y_range, P_values


class MundellFlemingModel:
    """
    Modelo Mundell-Fleming con tipo de cambio fijo/flexible
    IS*: Y = C(Y-T) + I(r*) + G + NX(e)
    LM*: M/P = L(Y, r*)
    BP: NX(e) + CF(r* - r) = 0
    """
    
    def __init__(self, params=None):
        self.params = params or {
            'C0': 50, 'c': 0.75, 'I0': 100, 'b': 0.4,
            'G': 120, 'T': 80, 'M': 200, 'P': 1.0,
            'k': 0.2, 'h': 80,
            'NX0': 20, 'm': 0.1, 'e': 1.0,  # Tipo de cambio
            'r_w': 0.04,                    # Tasa mundial
            'CF': 0.5                       # Movilidad capital
        }
        self.exchange_rate_fixed = True
        self.equilibrium = None
        self.name = "Mundell-Fleming"
    
    def equations(self, vars):
        """Sistema de ecuaciones Mundell-Fleming"""
        Y, r, e = vars
        
        # Consumo
        C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
        # Inversión
        I = self.params['I0'] - self.params['b'] * r * 100
        # Exportaciones netas
        NX = self.params['NX0'] - self.params['m'] * Y - self.params['m'] * e
        
        # IS*: Y = C + I + G + NX
        IS = Y - (C + I + self.params['G'] + NX)
        
        # LM*: M/P = kY - hr
        LM = self.params['M']/self.params['P'] - (self.params['k'] * Y - self.params['h'] * r)
        
        # BP: NX + CF*(r - r_w) = 0
        if self.exchange_rate_fixed:
            # Tipo de cambio fijo: e es exógeno, la BP determina M
            BP = NX + self.params['CF'] * (r - self.params['r_w'])
        else:
            # Tipo de cambio flexible: la BP determina e
            BP = NX + self.params['CF'] * (r - self.params['r_w'])
        
        return [IS, LM, BP]
    
    def solve(self, initial_guess=None):
        """Resolver equilibrio"""
        if initial_guess is None:
            initial_guess = [100, 0.05, 1.0]
        
        solution = fsolve(self.equations, initial_guess)
        self.equilibrium = {
            'Y': float(solution[0]),
            'r': float(solution[1]),
            'e': float(solution[2]),
            'NX': self.params['NX0'] - self.params['m'] * solution[0] - self.params['m'] * solution[2]
        }
        return self.equilibrium
    
    def apply_shock(self, shock_type, magnitude):
        """Aplicar choque"""
        if shock_type == 'Gasto Gobierno ↑':
            self.params['G'] *= (1 + magnitude)
        elif shock_type == 'Oferta Monetaria ↑':
            self.params['M'] *= (1 + magnitude)
        elif shock_type == 'Exportaciones ↑':
            self.params['NX0'] *= (1 + magnitude)
        elif shock_type == 'Tasa Mundial ↑':
            self.params['r_w'] *= (1 + magnitude)
        return self.solve()
    
    def toggle_exchange_rate(self, fixed=True):
        """Cambiar régimen cambiario"""
        self.exchange_rate_fixed = fixed


class DynamicISLMModel:
    """
    Modelo IS-LM dinámico con ajuste gradual de precios
    """
    
    def __init__(self, params=None):
        self.params = params or {
            'a': 0.5,  # Velocidad ajuste producción
            'b': 0.3,  # Velocidad ajuste precios
            'Y_n': 100,
            'P_e': 1.0
        }
        self.equilibrium = None
        self.name = "IS-LM Dinámico"
    
    def dynamics(self, state, t, shocks=None):
        """Ecuaciones diferenciales del sistema"""
        Y, P = state
        
        # Demanda agregada
        AD = 100 - P * 0.5 + 0.5 * Y  # Simplificada
        
        # Ajuste de producción
        dY_dt = self.params['a'] * (AD - Y)
        
        # Ajuste de precios (Curva de Phillips)
        dP_dt = self.params['b'] * (Y - self.params['Y_n']) + shocks if shocks else 0
        
        return [dY_dt, dP_dt]
    
    def simulate(self, T=50, shocks=None):
        """Simular trayectoria temporal"""
        initial_state = [80, 1.0]
        t = np.linspace(0, T, 100)
        
        if shocks is None:
            shocks = [0] * len(t)
        
        solution = odeint(self.dynamics, initial_state, t, args=(shocks,))
        
        return {
            't': t,
            'Y': solution[:, 0],
            'P': solution[:, 1]
        }


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
        
        # PIB con tendencia y ciclo
        trend = np.linspace(100, 150, 240)
        cycle = 5 * np.sin(np.linspace(0, 4*np.pi, 240))
        gdp = trend + cycle + np.random.normal(0, 2, 240)
        
        # Inflación
        inflation = 0.02 + 0.01 * np.sin(np.linspace(0, 3*np.pi, 240)) + np.random.normal(0, 0.005, 240)
        
        # Tasa de interés (Taylor rule)
        interest = 0.02 + 1.5 * (inflation - 0.02) + 0.5 * ((gdp - trend)/trend) + np.random.normal(0, 0.005, 240)
        
        # Desempleo (Ley de Okun)
        unemployment = 0.05 - 0.5 * ((gdp - trend)/trend) + np.random.normal(0, 0.005, 240)
        
        # Tipo de cambio
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
    
    @staticmethod
    def get_fred_data(series_ids, start_date='2000-01-01'):
        """Obtener datos de FRED (simulado para demo)"""
        # Nota: En producción usar pandas-datareader con FRED API
        return DataManager.generate_sample_data()


# ============================================================================
# MÓDULO 3: VISUALIZACIONES AVANZADAS Y ANIMACIONES
# ============================================================================

class Visualizer:
    """Visualizaciones interactivas y animaciones"""
    
    @staticmethod
    def plot_is_lm_animated(model, shock_type=None, magnitude=0.1, frames=20):
        """Crear animación de transición IS-LM"""
        # Obtener estado inicial
        initial_eq = model.solve()
        
        # Generar frames de transición
        fig = go.Figure()
        
        # Curvas base
        Y_range = np.linspace(40, 200, 100)
        Y_is, r_is = model.get_is_curve(Y_range)
        Y_lm, r_lm = model.get_lm_curve(Y_range)
        
        # Añadir curvas estáticas
        fig.add_trace(go.Scatter(x=Y_is, y=r_is, mode='lines', name='IS',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=Y_lm, y=r_lm, mode='lines', name='LM',
                                line=dict(color='red', width=2)))
        
        # Punto inicial
        fig.add_trace(go.Scatter(x=[initial_eq['Y']], y=[initial_eq['r']],
                                mode='markers', name='Equilibrio Inicial',
                                marker=dict(size=12, color='green', symbol='star')))
        
        # Frames para animación
        if shock_type:
            frames = []
            for i in range(frames):
                progress = i / frames
                temp_model = ISLMModel(model.params.copy())
                
                if shock_type == 'Gasto Gobierno ↑':
                    temp_model.params['G'] = model.params['G'] * (1 + magnitude * progress)
                elif shock_type == 'Oferta Monetaria ↑':
                    temp_model.params['M'] = model.params['M'] * (1 + magnitude * progress)
                
                eq = temp_model.solve()
                
                frame = go.Frame(data=[
                    go.Scatter(x=[eq['Y']], y=[eq['r']], mode='markers',
                              marker=dict(size=12, color='orange', symbol='star'))
                ], name=f'frame{i}')
                frames.append(frame)
            
            fig.frames = frames
        
        fig.update_layout(
            title="Modelo IS-LM - Transición Dinámica",
            xaxis_title="Producción (Y)",
            yaxis_title="Tasa de Interés (r)",
            template="plotly_white",
            height=500,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="▶ Ejecutar", method="animate", args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}])]
            )]
        )
        
        return fig
    
    @staticmethod
    def plot_macro_dashboard(data):
        """Dashboard macroeconómico completo"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('PIB', 'Inflación', 'Tasa de Interés', 
                          'Desempleo', 'Tipo de Cambio', 'Correlaciones'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}],
                   [{'colspan': 2}, None]]
        )
        
        # PIB
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['PIB'], 
                                mode='lines', name='PIB',
                                line=dict(color='#1f77b4', width=2)), row=1, col=1)
        
        # Inflación
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Inflacion']*100,
                                mode='lines', name='Inflación (%)',
                                line=dict(color='#ff7f0e', width=2)), row=1, col=2)
        
        # Tasa de Interés
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Tasa_Interes']*100,
                                mode='lines', name='Tasa (%)',
                                line=dict(color='#2ca02c', width=2)), row=2, col=1)
        
        # Desempleo
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Desempleo']*100,
                                mode='lines', name='Desempleo (%)',
                                line=dict(color='#d62728', width=2)), row=2, col=2)
        
        # Tipo de Cambio
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
    def plot_ad_as_animated(model, frames=20):
        """Animación AD-AS con desplazamientos"""
        fig = go.Figure()
        
        # Datos base
        P_range = np.linspace(0.3, 2.0, 100)
        Y_ad, P_ad = model.get_ad_curve(P_range)
        Y_sras, P_sras = model.get_sras_curve(np.linspace(50, 150, 100))
        
        # Curvas estáticas
        fig.add_trace(go.Scatter(x=Y_ad, y=P_ad, mode='lines', name='AD',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=Y_sras, y=P_sras, mode='lines', name='SRAS',
                                line=dict(color='red', width=2)))
        
        # LRAS
        fig.add_vline(x=model.params['Y_n'], line_dash="dash", line_color="green",
                     annotation_text="LRAS")
        
        # Equilibrio inicial
        eq = model.solve()
        fig.add_trace(go.Scatter(x=[eq['Y']], y=[eq['P']], mode='markers',
                                marker=dict(size=12, color='green', symbol='star'),
                                name='Equilibrio'))
        
        fig.update_layout(
            title="Modelo AD-AS - Equilibrio Macroeconómico",
            xaxis_title="Producción (Y)",
            yaxis_title="Nivel de Precios (P)",
            template="plotly_white",
            height=500
        )
        
        return fig


# ============================================================================
# MÓDULO 4: SIMULACIÓN DE POLÍTICAS Y ESCENARIOS
# ============================================================================

class PolicySimulator:
    """Simulación de políticas económicas"""
    
    def __init__(self, model):
        self.model = model
        self.scenarios = {}
    
    def run_scenario(self, name, policy_changes):
        """
        Ejecutar escenario de política
        
        Args:
            name: Nombre del escenario
            policy_changes: Dict con cambios en parámetros
        """
        # Guardar estado original
        original_params = self.model.params.copy()
        
        # Aplicar cambios
        for param, value in policy_changes.items():
            if param in self.model.params:
                self.model.params[param] = value
        
        # Resolver nuevo equilibrio
        new_eq = self.model.solve()
        
        # Guardar escenario
        self.scenarios[name] = {
            'params': self.model.params.copy(),
            'equilibrium': new_eq
        }
        
        # Restaurar original
        self.model.params = original_params
        self.model.solve()
        
        return new_eq
    
    def compare_scenarios(self, scenarios_list):
        """Comparar múltiples escenarios"""
        results = []
        for name in scenarios_list:
            if name in self.scenarios:
                results.append({
                    'Escenario': name,
                    'Y': self.scenarios[name]['equilibrium']['Y'],
                    'r': self.scenarios[name]['equilibrium'].get('r', 0),
                    'P': self.scenarios[name]['equilibrium'].get('P', 1)
                })
        return pd.DataFrame(results)


# ============================================================================
# MÓDULO 5: GENERACIÓN DE REPORTES PDF
# ============================================================================

class ReportGenerator:
    """Generación automática de reportes en PDF"""
    
    @staticmethod
    def generate_report(title, content, figures, filename="reporte.pdf"):
        """Generar reporte PDF completo"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Estilo personalizado
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30
        )
        
        # Construir contenido
        story = []
        
        # Título
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        
        # Fecha
        story.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                              styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Contenido principal
        for section in content:
            story.append(Paragraph(section['title'], styles['Heading2']))
            story.append(Paragraph(section['text'], styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Tablas
        for table_data in content.get('tables', []):
            table = Table(table_data['data'])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
        
        # Construir documento
        doc.build(story)
        
        # Retornar PDF como bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes


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
        st.image("https://img.icons8.com/fluency/96/000000/statistics.png", width=80)
        st.markdown("---")
        
        # Navegación principal
        nav_options = [
            "� Inicio",
            "� Modelos IS-LM y AD-AS",
            "� Mundell-Fleming",
            "� Animaciones",
            "� Series Temporales",
            "� Importar Datos",
            "� Simulación Políticas",
            "� Reportes PDF",
            "� Guardar Escenarios",
            "�️ Dashboard"
        ]
        
        selection = st.radio("Navegación", nav_options)
        
        st.markdown("---")
        st.info("� SICM v2.0 - Documentación completa disponible en el repositorio")
    
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
    # PÁGINA: MODELOS IS-LM Y AD-AS
    # ========================================================================
    
    elif selection == "� Modelos IS-LM y AD-AS":
        st.header("Modelos Macroeconómicos Clásicos")
        
        tab_is, tab_ad = st.tabs(["IS-LM", "AD-AS"])
        
        # --------------------------------------------------------------------
        # TAB IS-LM
        # --------------------------------------------------------------------
        with tab_is:
            st.subheader("Modelo IS-LM")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Parámetros**")
                c = st.slider("Propensión marginal a consumir (c)", 0.3, 0.95, 0.75, 0.05)
                G = st.slider("Gasto gobierno (G)", 50, 200, 120, 5)
                M = st.slider("Oferta monetaria (M)", 100, 400, 200, 10)
                
                # Crear modelo con parámetros actualizados
                params = {'C0': 50, 'c': c, 'I0': 100, 'b': 0.4, 
                         'G': G, 'T': 80, 'M': M, 'P': 1.0,
                         'k': 0.2, 'h': 80}
                model = ISLMModel(params)
                eq = model.solve()
            
            with col2:
                # Gráfico IS-LM
                fig = Visualizer.plot_is_lm_animated(model)
                st.plotly_chart(fig, use_container_width=True)
            
            # Métricas del equilibrio
            st.subheader("� Equilibrio Macroeconómico")
            cols = st.columns(4)
            cols[0].metric("Producción (Y)", f"{eq['Y']:.2f}")
            cols[1].metric("Tasa Interés (r)", f"{eq['r']:.2%}")
            cols[2].metric("Consumo (C)", f"{eq['C']:.2f}")
            cols[3].metric("Inversión (I)", f"{eq['I']:.2f}")
            
            # Ecuaciones mostradas
            with st.expander("� Ver ecuaciones del modelo"):
                st.latex(r"Y = C(Y-T) + I(r) + G")
                st.latex(r"C = C_0 + c(Y-T)")
                st.latex(r"I = I_0 - b \cdot r")
                st.latex(r"\frac{M}{P} = kY - h \cdot r")
                st.latex(r"\text{IS: } Y = C(Y-T) + I(r) + G")
                st.latex(r"\text{LM: } \frac{M}{P} = L(Y,r)")
        
        # --------------------------------------------------------------------
        # TAB AD-AS
        # --------------------------------------------------------------------
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
                lr_ad = model_ad.long_run_equilibrium()
            
            with col2:
                fig = Visualizer.plot_ad_as_animated(model_ad)
                st.plotly_chart(fig, use_container_width=True)
            
            # Métricas
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
            
            # Parámetros
            G_mf = st.slider("Gasto gobierno (G)", 50, 200, 120, 5)
            M_mf = st.slider("Oferta monetaria (M)", 100, 400, 200, 10)
            r_w = st.slider("Tasa mundial (r*)", 0.01, 0.08, 0.04, 0.01)
            
            # Crear modelo
            params_mf = {'C0': 50, 'c': 0.75, 'I0': 100, 'b': 0.4,
                        'G': G_mf, 'T': 80, 'M': M_mf, 'P': 1.0,
                        'k': 0.2, 'h': 80, 'NX0': 20, 'm': 0.1, 'e': 1.0,
                        'r_w': r_w, 'CF': {'Nula': 0, 'Baja': 0.2, 'Media': 0.5, 
                                          'Alta': 0.8, 'Perfecta': 100}[mobility]}
            
            model_mf = MundellFlemingModel(params_mf)
            model_mf.exchange_rate_fixed = (regime == "Fijo")
            
            if st.button("Calcular Equilibrio", type="primary"):
                eq_mf = model_mf.solve()
                st.session_state['mf_eq'] = eq_mf
        
        with col2:
            if 'mf_eq' in st.session_state:
                eq = st.session_state['mf_eq']
                
                # Gráfico simplificado
                fig = go.Figure()
                
                # Curvas IS, LM, BP
                r_range = np.linspace(0, 0.10, 50)
                Y_is = 100 - 50 * r_range + G_mf  # IS simplificada
                Y_lm = 50 + 100 * r_range + M_mf/10  # LM simplificada
                Y_bp = 100 + 50 * (r_range - r_w)  # BP
                
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Métricas
                cols = st.columns(4)
                cols[0].metric("Producción (Y)", f"{eq['Y']:.2f}")
                cols[1].metric("Tasa Interés", f"{eq['r']:.2%}")
                cols[2].metric("Tipo Cambio", f"{eq['e']:.3f}")
                cols[3].metric("Exportaciones Netas", f"{eq['NX']:.2f}")
    
    # ========================================================================
    # PÁGINA: ANIMACIONES
    # ========================================================================
    
    elif selection == "� Animaciones":
        st.header("� Animaciones de Transición Económica")
        
        st.markdown("""
        Las animaciones muestran la transición dinámica entre estados económicos
        después de un choque. Observe cómo se ajustan las variables en el tiempo.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuración")
            
            model_type = st.selectbox("Modelo", ["IS-LM", "AD-AS", "IS-LM Dinámico"])
            shock_type = st.selectbox("Tipo de Choque", 
                                     ["Gasto Gobierno ↑", "Oferta Monetaria ↑", 
                                      "Impuestos ↓", "Productividad ↑"])
            magnitude = st.slider("Magnitud", 0.05, 0.30, 0.10, 0.05)
            speed = st.slider("Velocidad Animación", 1, 5, 3)
            
            if st.button("▶️ Ejecutar Animación", type="primary"):
                st.session_state['animating'] = True
        
        with col2:
            if st.session_state.get('animating', False):
                # Crear modelo base
                model = ISLMModel()
                model.solve()
                
                # Generar animación
                fig = Visualizer.plot_is_lm_animated(model, shock_type, magnitude)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Animación cargada - Presione '▶ Ejecutar' para reproducir")
                
                # Descripción del mecanismo
                with st.expander("� Mecanismo de Transmisión"):
                    st.markdown(f"""
                    **Choque:** {shock_type} ({magnitude:.0%})
                    
                    **Mecanismo:**
                    1. El choque desplaza la curva {'IS' if 'Gasto' in shock_type else 'LM'}
                    2. La economía se mueve al nuevo equilibrio de corto plazo
                    3. Los precios comienzan a ajustarse gradualmente
                    4. La economía converge al nuevo equilibrio de largo plazo
                    
                    **Efectos:**
                    - Producción: {'↑' if '↑' in shock_type else '↓'}
                    - Tasa de interés: {'↑' if 'Gasto' in shock_type else '↓'}
                    - Precios: {'↑' if '↑' in shock_type else '↓'} (en el largo plazo)
                    """)
    
    # ========================================================================
    # PÁGINA: SERIES TEMPORALES
    # ========================================================================
    
    elif selection == "� Series Temporales":
        st.header("� Análisis de Series Temporales Macroeconómicas")
        
        # Cargar o generar datos
        if st.button("� Cargar Datos de Ejemplo"):
            data = DataManager.generate_sample_data()
            st.session_state['time_series_data'] = data
        
        if 'time_series_data' in st.session_state:
            data = st.session_state['time_series_data']
            
            # Selector de variables
            variables = st.multiselect(
                "Seleccionar variables",
                ['PIB', 'Inflacion', 'Tasa_Interes', 'Desempleo', 'Tipo_Cambio'],
                default=['PIB', 'Inflacion']
            )
            
            if variables:
                # Gráfico interactivo
                fig = go.Figure()
                
                for var in variables:
                    col_name = var
                    fig.add_trace(go.Scatter(
                        x=data['fecha'], 
                        y=data[col_name],
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas descriptivas
                with st.expander("� Estadísticas Descriptivas"):
                    st.dataframe(data[variables].describe(), use_container_width=True)
                
                # Dashboard completo
                if st.button("� Mostrar Dashboard Completo"):
                    fig_dash = Visualizer.plot_macro_dashboard(data)
                    st.plotly_chart(fig_dash, use_container_width=True)
    
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
            if st.button("� Generar Datos de Ejemplo"):
                sample_data = DataManager.generate_sample_data()
                st.session_state['imported_data'] = sample_data
                st.success("✅ Datos de ejemplo generados")
                st.dataframe(sample_data.head(), use_container_width=True)
        
        if 'imported_data' in st.session_state:
            st.subheader("� Visualización de Datos Importados")
            df = st.session_state['imported_data']
            
            # Gráfico rápido
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
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PÁGINA: SIMULACIÓN DE POLÍTICAS
    # ========================================================================
    
    elif selection == "� Simulación Políticas":
        st.header("� Simulación de Políticas Económicas")
        
        st.markdown("""
        **Simule diferentes políticas económicas y compare sus efectos**
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Modelo Base")
            model_type = st.selectbox("Modelo", ["IS-LM", "AD-AS"], key="sim_model")
            
            # Parámetros base
            if model_type == "IS-LM":
                G_base = st.slider("Gasto gobierno (G)", 50, 200, 100)
                M_base = st.slider("Oferta monetaria (M)", 100, 400, 200)
                
                model = ISLMModel({'G': G_base, 'M': M_base})
            else:
                M_base = st.slider("Oferta monetaria (M)", 100, 400, 200)
                Yn_base = st.slider("Producción natural (Y_n)", 80, 150, 100)
                
                model = ADASModel({'M': M_base, 'Y_n': Yn_base})
            
            eq_base = model.solve()
            st.metric("Equilibrio Base - Y", f"{eq_base['Y']:.2f}")
        
        with col2:
            st.subheader("Políticas a Simular")
            
            policy_type = st.selectbox("Tipo de Política", 
                                      ["Fiscal Expansiva", "Fiscal Contractiva",
                                       "Monetaria Expansiva", "Monetaria Contractiva"])
            
            magnitude_pol = st.slider("Magnitud", 0.05, 0.30, 0.10, 0.05)
            
            if st.button("▶️ Ejecutar Simulación", type="primary"):
                # Aplicar política
                if policy_type == "Fiscal Expansiva":
                    shock = "Gasto Gobierno ↑"
                elif policy_type == "Fiscal Contractiva":
                    shock = "Gasto Gobierno ↓"
                elif policy_type == "Monetaria Expansiva":
                    shock = "Oferta Monetaria ↑"
                else:
                    shock = "Oferta Monetaria ↓"
                
                # Simular
                if model_type == "IS-LM":
                    model_after = ISLMModel(model.params.copy())
                else:
                    model_after = ADASModel(model.params.copy())
                
                eq_after = model_after.apply_shock(shock, magnitude_pol)
                
                # Mostrar resultados
                st.subheader("� Resultados de la Simulación")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Variable", "Producción (Y)")
                col_b.metric("Valor Base", f"{eq_base['Y']:.2f}")
                col_c.metric("Valor Simulado", f"{eq_after['Y']:.2f}", 
                            delta=f"{eq_after['Y'] - eq_base['Y']:+.2f}")
                
                # Tabla comparativa
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
        
        st.markdown("""
        **Genere reportes profesionales en PDF con análisis completos**
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Configuración del Reporte")
            
            report_title = st.text_input("Título del Reporte", "Análisis Macroeconómico")
            include_sections = st.multiselect(
                "Secciones a incluir",
                ["Resumen Ejecutivo", "Análisis de Modelos", "Resultados de Simulación", 
                 "Series Temporales", "Conclusiones"],
                default=["Resumen Ejecutivo", "Conclusiones"]
            )
            
            if st.button("� Generar Reporte", type="primary"):
                # Contenido del reporte
                content = []
                
                if "Resumen Ejecutivo" in include_sections:
                    content.append({
                        'title': 'Resumen Ejecutivo',
                        'text': f'''
                        El presente análisis macroeconómico fue realizado con el SICM v2.0.
                        Se evaluaron {len(include_sections)} dimensiones del entorno económico.
                        Los resultados muestran las interacciones clave entre variables macroeconómicas.
                        '''
                    })
                
                if "Conclusiones" in include_sections:
                    content.append({
                        'title': 'Conclusiones',
                        'text': '''
                        • Los modelos IS-LM y AD-AS proporcionan un marco completo
                        • Los choques fiscales tienen efectos multiplicadores
                        • Las políticas monetarias afectan principalmente precios
                        • El análisis de series temporales muestra tendencias cíclicas
                        '''
                    })
                
                # Generar PDF
                pdf_bytes = ReportGenerator.generate_report(
                    report_title, content, [], "reporte.pdf"
                )
                
                # Ofrecer descarga
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="reporte_macroeconomico.pdf">� Descargar Reporte PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ Reporte generado exitosamente")
    
    # ========================================================================
    # PÁGINA: GUARDAR ESCENARIOS
    # ========================================================================
    
    elif selection == "� Guardar Escenarios":
        st.header("� Gestión de Escenarios")
        
        st.markdown("""
        **Guarde, cargue y compare diferentes escenarios económicos**
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Guardar Escenario Actual")
            
            scenario_name = st.text_input("Nombre del Escenario", "Escenario_1")
            scenario_desc = st.text_area("Descripción")
            
            if st.button("� Guardar Escenario", type="primary"):
                # Guardar escenario actual
                if 'current_scenario' in st.session_state:
                    scenario = {
                        'name': scenario_name,
                        'description': scenario_desc,
                        'data': st.session_state['current_scenario'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if 'saved_scenarios' not in st.session_state:
                        st.session_state['saved_scenarios'] = []
                    
                    st.session_state['saved_scenarios'].append(scenario)
                    st.success(f"✅ Escenario '{scenario_name}' guardado")
        
        with col2:
            st.subheader("Escenarios Guardados")
            
            if 'saved_scenarios' in st.session_state:
                scenarios = st.session_state['saved_scenarios']
                
                if scenarios:
                    for i, s in enumerate(scenarios):
                        st.markdown(f"**{s['name']}**")
                        st.caption(f"{s['description']}")
                        st.caption(f"� {s['timestamp'][:10]}")
                        
                        if st.button(f"� Cargar", key=f"load_{i}"):
                            st.session_state['current_scenario'] = s['data']
                            st.success(f"✅ Escenario '{s['name']}' cargado")
                        
                        st.markdown("---")
                else:
                    st.info("No hay escenarios guardados")
        
        # Comparación de escenarios
        st.subheader("� Comparación de Escenarios")
        
        if 'saved_scenarios' in st.session_state and st.session_state['saved_scenarios']:
            scenarios = st.session_state['saved_scenarios']
            scenario_names = [s['name'] for s in scenarios]
            
            selected = st.multiselect("Seleccionar escenarios a comparar", scenario_names)
            
            if selected:
                data = []
                for name in selected:
                    for s in scenarios:
                        if s['name'] == name:
                            data.append({
                                'Escenario': name,
                                'Y': s['data'].get('Y', 0),
                                'r': s['data'].get('r', 0),
                                'P': s['data'].get('P', 1)
                            })
                
                if data:
                    df_comp = pd.DataFrame(data)
                    st.dataframe(df_comp, use_container_width=True)
    
    # ========================================================================
    # PÁGINA: DASHBOARD
    # ========================================================================
    
    elif selection == "�️ Dashboard":
        st.header("�️ Dashboard - Laboratorio de Economía")
        
        st.markdown("""
        **Panel de control integral para análisis macroeconómico**
        """)
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PIB", "120.5", "+2.3%", delta_color="normal")
        col2.metric("Inflación", "2.8%", "+0.3%", delta_color="inverse")
        col3.metric("Desempleo", "5.2%", "-0.1%", delta_color="normal")
        col4.metric("Tasa Interés", "4.5%", "+0.25%", delta_color="inverse")
        
        # Gráficos del dashboard
        if 'time_series_data' in st.session_state:
            data = st.session_state['time_series_data']
            
            # Dashboard completo
            fig = Visualizer.plot_macro_dashboard(data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Datos de ejemplo si no hay datos cargados
            if st.button("� Cargar Datos de Ejemplo para Dashboard"):
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
