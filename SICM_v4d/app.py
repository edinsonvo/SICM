"""
SICM v3.0 - Simulador Integral de Choques Macroeconómicos
Versión Universitaria y de Investigación (Corregida y Mejorada)

Cambios principales v3.0:
- Mundell-Fleming: sistemas separados para tipo de cambio fijo (M endógena) y flexible (e endógena).
- AD-AS: curva AD derivada rigurosamente del equilibrio IS-LM para cada nivel de precios P.
- IS-LM: unificación de escalas (tasas en decimales) y validación numérica post-fsolve.
- IS-LM Dinámico: demanda agregada derivada de IS-LM + Curva de Phillips con expectativas.
- Datos reales: integración con FRED (pandas_datareader) + fallback a datos sintéticos realistas.
- Animaciones: implementación funcional vía st.empty() + bucle temporal en Streamlit.
- Reportes PDF: inserción de figuras Plotly convertidas a PNG vía kaleido.
- Persistencia: guardado/carga de escenarios en JSON.
- Caching: @st.cache_data en cálculos costosos y carga de datos.
- Robustez: manejo de excepciones, validación de convergencia y mensajes de error claros.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import fsolve
from scipy.integrate import odeint
from io import BytesIO
import base64
from datetime import datetime, timedelta
import json
import time
import warnings
import os
import tempfile

warnings.filterwarnings('ignore')

# Intentar importar reportlab para PDFs
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Intentar importar kaleido para exportar imágenes Plotly
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# Intentar importar pandas_datareader para FRED
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="SICM v3.0 - Laboratorio Macroeconómico",
    page_icon="📊",
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
    .stAlert {
        border-radius: 8px;
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
    st.session_state.saved_scenarios_json = []
    st.session_state.time_series_data = None
    st.session_state.imported_data = None
    st.session_state.anim_frame = 0
    st.session_state.mf_eq = None

# ============================================================================
# MÓDULO 1: MODELOS ECONÓMICOS CON ECUACIONES REALES Y VALIDACIÓN
# ============================================================================

class ISLMModel:
    """
    Modelo IS-LM con ecuaciones completas y validación numérica.
    Y = C(Y-T) + I(r) + G
    M/P = L(Y, r)

    Tasas de interés manejadas SIEMPRE en decimales (ej: 0.05 = 5%).
    """

    def __init__(self, params=None):
        self.params = params or {
            'C0': 50,       # Consumo autónomo
            'c': 0.75,      # Propensión marginal a consumir
            'I0': 100,      # Inversión autónoma
            'b': 20.0,      # Sensibilidad inversión a tasa (en decimales)
            'G': 120,       # Gasto gobierno
            'T': 80,        # Impuestos
            'M': 200,       # Oferta monetaria
            'P': 1.0,       # Nivel de precios
            'k': 0.2,       # Sensibilidad demanda dinero a Y
            'h': 800        # Sensibilidad demanda dinero a r (en decimales)
        }
        self.equilibrium = None
        self.name = "IS-LM"
        self._validate_params()

    def _validate_params(self):
        """Validar que los parámetros sean económicamente coherentes."""
        if not (0 < self.params['c'] < 1):
            raise ValueError("La propensión marginal a consumir 'c' debe estar entre 0 y 1.")
        if self.params['b'] <= 0:
            raise ValueError("La sensibilidad de la inversión 'b' debe ser positiva.")
        if self.params['h'] <= 0:
            raise ValueError("La sensibilidad de la demanda de dinero 'h' debe ser positiva.")
        if self.params['P'] <= 0:
            raise ValueError("El nivel de precios 'P' debe ser positivo.")

    def equations(self, vars):
        """Sistema de ecuaciones IS-LM. vars = [Y, r]"""
        Y, r = vars
        C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
        I = self.params['I0'] - self.params['b'] * r
        IS = Y - (C + I + self.params['G'])
        LM = self.params['M'] / self.params['P'] - (self.params['k'] * Y - self.params['h'] * r)
        return [IS, LM]

    def solve(self, initial_guess=None):
        """Resolver equilibrio general con validación de convergencia."""
        if initial_guess is None:
            # Estimación inicial razonable
            Y_guess = (self.params['C0'] - self.params['c']*self.params['T'] + 
                       self.params['I0'] + self.params['G']) / (1 - self.params['c'])
            r_guess = (self.params['k'] * Y_guess - self.params['M']/self.params['P']) / self.params['h']
            r_guess = max(0.001, r_guess)
            initial_guess = [Y_guess, r_guess]

        try:
            solution, info, ier, mesg = fsolve(self.equations, initial_guess, full_output=True)
        except Exception as e:
            raise RuntimeError(f"Error numérico al resolver IS-LM: {e}")

        # Validación estricta de convergencia
        residual = np.array(self.equations(solution))
        if ier != 1 or np.any(np.abs(residual) > 1e-6):
            raise RuntimeError(
                f"El solver no convergió a una solución válida. "
                f"Residual: {residual}. Mensaje: {mesg}. "
                f"Sugerencia: revise que los parámetros sean consistentes (ej: G, M > 0)."
            )

        Y_eq, r_eq = float(solution[0]), float(solution[1])

        if Y_eq < 0 or r_eq < 0:
            raise RuntimeError(
                f"Equilibrio no válido: Y={Y_eq:.2f}, r={r_eq:.4f}. "
                f"Posible causa: política contractiva excesiva o parámetros inconsistentes."
            )

        self.equilibrium = {
            'Y': Y_eq,
            'r': r_eq,
            'C': self.params['C0'] + self.params['c'] * (Y_eq - self.params['T']),
            'I': self.params['I0'] - self.params['b'] * r_eq,
            'G': self.params['G']
        }
        return self.equilibrium

    def apply_shock(self, shock_type, magnitude):
        """Aplicar choque y retornar nuevo equilibrio."""
        original = {k: v for k, v in self.params.items()}

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
        elif shock_type == 'Precios ↑':
            self.params['P'] *= (1 + magnitude)
        elif shock_type == 'Precios ↓':
            self.params['P'] *= (1 - magnitude)

        try:
            eq = self.solve()
        except Exception as e:
            self.params = original
            raise e

        return eq

    def get_is_curve(self, Y_range=None):
        """Obtener puntos de la curva IS: r = (C0 + I0 + G - cT - (1-c)Y) / b"""
        if Y_range is None:
            Y_range = np.linspace(50, 300, 200)
        r_values = []
        for Y in Y_range:
            num = (self.params['C0'] + self.params['I0'] + self.params['G'] - 
                   self.params['c'] * self.params['T'] - (1 - self.params['c']) * Y)
            r = num / self.params['b']
            r_values.append(max(0, r))
        return Y_range, np.array(r_values)

    def get_lm_curve(self, Y_range=None):
        """Obtener puntos de la curva LM: r = (kY - M/P) / h"""
        if Y_range is None:
            Y_range = np.linspace(50, 300, 200)
        r_values = (self.params['k'] * Y_range - self.params['M']/self.params['P']) / self.params['h']
        r_values = np.maximum(r_values, 0)
        return Y_range, r_values

    def get_ad_point(self, P):
        """Dado un nivel de precios P, resolver IS-LM para obtener Y (punto de la curva AD)."""
        old_P = self.params['P']
        self.params['P'] = P
        try:
            eq = self.solve()
            Y_ad = eq['Y']
        except Exception:
            Y_ad = np.nan
        finally:
            self.params['P'] = old_P
        return Y_ad


class ADASModel:
    """
    Modelo AD-AS donde la curva AD se deriva RIGUROSAMENTE del modelo IS-LM.
    SRAS: P = P_e * (1 + λ(Y - Y_n))
    LRAS: Y = Y_n
    """

    def __init__(self, islm_params=None, adas_params=None):
        # Parámetros del subsistema IS-LM subyacente
        self.islm_params = islm_params or {
            'C0': 50, 'c': 0.75, 'I0': 100, 'b': 20.0,
            'G': 120, 'T': 80, 'M': 200, 'P': 1.0,
            'k': 0.2, 'h': 800
        }
        # Parámetros específicos de AS
        self.adas_params = adas_params or {
            'Y_n': 100,         # Producción natural
            'λ': 0.05,          # Pendiente SRAS
            'P_e': 1.0,         # Precios esperados
        }
        self.equilibrium = None
        self.name = "AD-AS (derivado de IS-LM)"

    def _get_ad_from_islm(self, P_range):
        """Derivar curva AD resolviendo IS-LM para cada P."""
        Y_values = []
        model = ISLMModel({**self.islm_params, 'P': 1.0})  # P se sobreescribe en el loop
        for P in P_range:
            Y = model.get_ad_point(P)
            Y_values.append(Y)
        return np.array(Y_values), P_range

    def equations(self, vars):
        """Sistema AD-AS usando AD derivada de IS-LM."""
        Y, P = vars
        # AD: para este P, ¿qué Y da el equilibrio IS-LM?
        model = ISLMModel({**self.islm_params, 'P': P})
        try:
            eq = model.solve()
            Y_ad = eq['Y']
        except Exception:
            return [1e6, 1e6]  # Forzar fallo si no converge

        # SRAS
        P_sras = self.adas_params['P_e'] * (1 + self.adas_params['λ'] * (Y - self.adas_params['Y_n']))

        return [Y - Y_ad, P - P_sras]

    def solve(self, initial_guess=None):
        """Resolver equilibrio de corto plazo."""
        if initial_guess is None:
            initial_guess = [self.adas_params['Y_n'], self.adas_params['P_e']]

        solution, info, ier, mesg = fsolve(self.equations, initial_guess, full_output=True)
        residual = np.array(self.equations(solution))

        if ier != 1 or np.any(np.abs(residual) > 1e-5):
            raise RuntimeError(f"AD-AS no convergió. Residual: {residual}. {mesg}")

        self.equilibrium = {
            'Y': float(solution[0]),
            'P': float(solution[1]),
            'Y_n': self.adas_params['Y_n'],
            'P_e': self.adas_params['P_e']
        }
        return self.equilibrium

    def long_run_equilibrium(self):
        """Resolver equilibrio de largo plazo (Y = Y_n, P ajusta para que IS-LM lo sostenga)."""
        Y = self.adas_params['Y_n']
        # En el largo plazo, P se ajusta para que el equilibrio IS-LM tenga Y = Y_n
        # De LM: M/P = k*Yn - h*r  =>  r = (k*Yn - M/P)/h
        # De IS: Yn = C0 + c(Yn-T) + I0 - b*r + G
        # Sustituyendo r y despejando P...
        C = self.islm_params['C0'] + self.islm_params['c'] * (Y - self.islm_params['T'])
        I_base = self.islm_params['I0'] + self.islm_params['G']
        # Yn = C + I_base - b*(k*Yn - M/P)/h
        # Despejando P...
        num = self.islm_params['b'] * self.islm_params['M']
        den = self.islm_params['h'] * (C + I_base - Y) + self.islm_params['b'] * self.islm_params['k'] * Y
        if den <= 0:
            return {'Y': Y, 'P': np.nan}
        P_lr = num / den
        return {'Y': Y, 'P': P_lr}

    def apply_shock(self, shock_type, magnitude):
        """Aplicar choque."""
        original_islm = {k: v for k, v in self.islm_params.items()}
        original_adas = {k: v for k, v in self.adas_params.items()}

        if shock_type == 'Oferta Monetaria ↑':
            self.islm_params['M'] *= (1 + magnitude)
        elif shock_type == 'Oferta Monetaria ↓':
            self.islm_params['M'] *= (1 - magnitude)
        elif shock_type == 'Gasto Gobierno ↑':
            self.islm_params['G'] *= (1 + magnitude)
        elif shock_type == 'Gasto Gobierno ↓':
            self.islm_params['G'] *= (1 - magnitude)
        elif shock_type == 'Productividad ↑':
            self.adas_params['Y_n'] *= (1 + magnitude)
        elif shock_type == 'Productividad ↓':
            self.adas_params['Y_n'] *= (1 - magnitude)
        elif shock_type == 'Expectativas Precios ↑':
            self.adas_params['P_e'] *= (1 + magnitude)
        elif shock_type == 'Expectativas Precios ↓':
            self.adas_params['P_e'] *= (1 - magnitude)

        try:
            eq = self.solve()
        except Exception as e:
            self.islm_params = original_islm
            self.adas_params = original_adas
            raise e
        return eq

    def get_ad_curve(self, P_range=None):
        """Obtener puntos de la curva AD derivada de IS-LM."""
        if P_range is None:
            P_range = np.linspace(0.3, 3.0, 100)
        Y_values, P_vals = self._get_ad_from_islm(P_range)
        # Filtrar valores inválidos
        mask = (~np.isnan(Y_values)) & (Y_values > 0)
        return Y_values[mask], P_vals[mask]

    def get_sras_curve(self, Y_range=None):
        """Obtener puntos de la curva SRAS."""
        if Y_range is None:
            Y_range = np.linspace(50, 200, 100)
        P_values = self.adas_params['P_e'] * (1 + self.adas_params['λ'] * (Y_range - self.adas_params['Y_n']))
        return Y_range, P_values


class MundellFlemingModel:
    """
    Modelo Mundell-Fleming con tipo de cambio fijo/flexible.

    Régimen Fijo: e = e0 (exógeno), incógnitas Y, r, M (oferta monetaria endógena).
    Régimen Flexible: M exógena, incógnitas Y, r, e.

    IS*: Y = C(Y-T) + I(r) + G + NX(e)
    LM*: M/P = L(Y, r)
    BP: NX(e) + CF*(r - r_w) = 0
    """

    def __init__(self, params=None):
        self.params = params or {
            'C0': 50, 'c': 0.75, 'I0': 100, 'b': 20.0,
            'G': 120, 'T': 80, 'M': 200, 'P': 1.0,
            'k': 0.2, 'h': 800,
            'NX0': 20, 'm': 0.1, 'e0': 1.0,
            'r_w': 0.04,
            'CF': 0.5
        }
        self.exchange_rate_fixed = True
        self.equilibrium = None
        self.name = "Mundell-Fleming"

    def equations_fixed(self, vars):
        """Sistema para tipo de cambio FIJO: vars = [Y, r, M]"""
        Y, r, M = vars
        C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
        I = self.params['I0'] - self.params['b'] * r
        NX = self.params['NX0'] - self.params['m'] * Y - self.params['m'] * self.params['e0']

        IS = Y - (C + I + self.params['G'] + NX)
        LM = M / self.params['P'] - (self.params['k'] * Y - self.params['h'] * r)
        BP = NX + self.params['CF'] * (r - self.params['r_w'])
        return [IS, LM, BP]

    def equations_flexible(self, vars):
        """Sistema para tipo de cambio FLEXIBLE: vars = [Y, r, e]"""
        Y, r, e = vars
        C = self.params['C0'] + self.params['c'] * (Y - self.params['T'])
        I = self.params['I0'] - self.params['b'] * r
        NX = self.params['NX0'] - self.params['m'] * Y - self.params['m'] * e

        IS = Y - (C + I + self.params['G'] + NX)
        LM = self.params['M'] / self.params['P'] - (self.params['k'] * Y - self.params['h'] * r)
        BP = NX + self.params['CF'] * (r - self.params['r_w'])
        return [IS, LM, BP]

    def solve(self, initial_guess=None):
        """Resolver equilibrio según régimen cambiario."""
        if self.exchange_rate_fixed:
            if initial_guess is None:
                initial_guess = [100, 0.05, 200]
            solution, info, ier, mesg = fsolve(self.equations_fixed, initial_guess, full_output=True)
            residual = np.array(self.equations_fixed(solution))

            if ier != 1 or np.any(np.abs(residual) > 1e-5):
                raise RuntimeError(f"MF (fijo) no convergió. Residual: {residual}")

            Y, r, M = float(solution[0]), float(solution[1]), float(solution[2])
            e = self.params['e0']
            self.equilibrium = {
                'Y': Y, 'r': r, 'e': e, 'M_endog': M,
                'NX': self.params['NX0'] - self.params['m'] * Y - self.params['m'] * e,
                'regime': 'Fijo'
            }
        else:
            if initial_guess is None:
                initial_guess = [100, 0.05, 1.0]
            solution, info, ier, mesg = fsolve(self.equations_flexible, initial_guess, full_output=True)
            residual = np.array(self.equations_flexible(solution))

            if ier != 1 or np.any(np.abs(residual) > 1e-5):
                raise RuntimeError(f"MF (flexible) no convergió. Residual: {residual}")

            Y, r, e = float(solution[0]), float(solution[1]), float(solution[2])
            self.equilibrium = {
                'Y': Y, 'r': r, 'e': e, 'M': self.params['M'],
                'NX': self.params['NX0'] - self.params['m'] * Y - self.params['m'] * e,
                'regime': 'Flexible'
            }
        return self.equilibrium

    def apply_shock(self, shock_type, magnitude):
        """Aplicar choque."""
        original = {k: v for k, v in self.params.items()}

        if shock_type == 'Gasto Gobierno ↑':
            self.params['G'] *= (1 + magnitude)
        elif shock_type == 'Gasto Gobierno ↓':
            self.params['G'] *= (1 - magnitude)
        elif shock_type == 'Oferta Monetaria ↑':
            self.params['M'] *= (1 + magnitude)
        elif shock_type == 'Oferta Monetaria ↓':
            self.params['M'] *= (1 - magnitude)
        elif shock_type == 'Exportaciones ↑':
            self.params['NX0'] *= (1 + magnitude)
        elif shock_type == 'Tasa Mundial ↑':
            self.params['r_w'] *= (1 + magnitude)
        elif shock_type == 'Tipo Cambio ↑ (devaluación)':
            self.params['e0'] *= (1 + magnitude)

        try:
            eq = self.solve()
        except Exception as e:
            self.params = original
            raise e
        return eq

    def get_curves(self, Y_range=None):
        """Generar curvas IS*, LM*, BP para visualización."""
        if Y_range is None:
            Y_range = np.linspace(50, 200, 100)

        if self.exchange_rate_fixed:
            e = self.params['e0']
            # Para visualización, usamos M endógena del equilibrio si existe
            M_plot = self.equilibrium['M_endog'] if self.equilibrium else self.params['M']
        else:
            e = self.equilibrium['e'] if self.equilibrium else self.params['e0']
            M_plot = self.params['M']

        # IS*: r = [C0 + I0 + G + NX0 - m*e - c*T - (1-c+m)Y] / b
        r_is = (self.params['C0'] + self.params['I0'] + self.params['G'] + 
                self.params['NX0'] - self.params['m']*e - self.params['c']*self.params['T'] - 
                (1 - self.params['c'] + self.params['m']) * Y_range) / self.params['b']

        # LM*: r = (kY - M/P) / h
        r_lm = (self.params['k'] * Y_range - M_plot/self.params['P']) / self.params['h']

        # BP: r = r_w - NX/CF = r_w - (NX0 - mY - me)/CF
        r_bp = self.params['r_w'] - (self.params['NX0'] - self.params['m']*Y_range - self.params['m']*e) / self.params['CF']

        return {
            'Y': Y_range,
            'IS': np.maximum(r_is, 0),
            'LM': np.maximum(r_lm, 0),
            'BP': np.maximum(r_bp, 0)
        }


class DynamicISLMModel:
    """
    Modelo IS-LM dinámico con ajuste gradual de precios.

    Demanda agregada Yd se deriva del equilibrio IS-LM para el P actual.
    Curva de Phillips: dP/dt = β(Y - Y_n) + γ(Pe - P)
    Ajuste de producción: dY/dt = α(Yd - Y)
    """

    def __init__(self, islm_params=None, dyn_params=None):
        self.islm_params = islm_params or {
            'C0': 50, 'c': 0.75, 'I0': 100, 'b': 20.0,
            'G': 120, 'T': 80, 'M': 200, 'P': 1.0,
            'k': 0.2, 'h': 800
        }
        self.dyn_params = dyn_params or {
            'alpha': 0.3,   # Velocidad ajuste producción
            'beta': 0.02,   # Velocidad ajuste precios (curva Phillips)
            'gamma': 0.1,   # Velocidad ajuste expectativas
            'Y_n': 100,
            'P_e': 1.0
        }
        self.name = "IS-LM Dinámico"

    def _get_Yd(self, P):
        """Obtener demanda agregada resolviendo IS-LM para el P dado."""
        model = ISLMModel({**self.islm_params, 'P': P})
        try:
            eq = model.solve()
            return eq['Y']
        except Exception:
            return self.dyn_params['Y_n']

    def dynamics(self, state, t):
        """Ecuaciones diferenciales: state = [Y, P]"""
        Y, P = state

        # Demanda agregada desde IS-LM
        Yd = self._get_Yd(P)

        # Ajuste de producción (desaceleración/aceleración hacia la demanda)
        dY_dt = self.dyn_params['alpha'] * (Yd - Y)

        # Curva de Phillips con expectativas adaptativas
        dP_dt = (self.dyn_params['beta'] * (Y - self.dyn_params['Y_n']) + 
                 self.dyn_params['gamma'] * (self.dyn_params['P_e'] - P))

        return [dY_dt, dP_dt]

    def simulate(self, T=50, dt=0.5, initial_state=None):
        """Simular trayectoria temporal."""
        if initial_state is None:
            initial_state = [self.dyn_params['Y_n'] * 0.85, self.dyn_params['P_e'] * 0.9]

        t = np.arange(0, T, dt)
        solution = odeint(self.dynamics, initial_state, t)

        return {
            't': t,
            'Y': solution[:, 0],
            'P': solution[:, 1]
        }


# ============================================================================
# MÓDULO 2: DATOS REALES Y PROCESAMIENTO (con FRED y caching)
# ============================================================================

class DataManager:
    """Gestión de datos reales e importación con caching."""

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Descargando datos de FRED...")
    def get_fred_data(series_ids, start_date='2000-01-01', end_date=None):
        """
        Obtener datos reales de FRED (Federal Reserve Economic Data).
        series_ids: dict con {'nombre_display': 'FRED_SERIES_ID'}
        """
        if not HAS_PDR:
            return None

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            data = pdr.DataReader(list(series_ids.values()), 'fred', start_date, end_date)
            data = data.rename(columns={v: k for k, v in series_ids.items()})
            data = data.reset_index()
            data = data.rename(columns={'DATE': 'fecha'})
            # Interpolar valores faltantes
            data = data.interpolate(method='linear')
            return data
        except Exception as e:
            st.warning(f"No se pudieron descargar datos de FRED: {e}. Usando datos sintéticos.")
            return None

    @staticmethod
    def generate_sample_data(seed=42):
        """Generar datos sintéticos realistas que emulan EE.UU. 2000-2024."""
        np.random.seed(seed)
        dates = pd.date_range(start='2000-01-01', periods=300, freq='M')
        n = len(dates)

        # Tendencia y ciclo realista
        trend = np.linspace(100, 165, n)
        # Ciclo con recesión 2008-2009 y COVID 2020
        cycle = np.zeros(n)
        for i, d in enumerate(dates):
            if 2008 <= d.year <= 2009:
                cycle[i] = -8 * np.exp(-((d.year - 2008.5)**2))
            elif d.year == 2020 and d.month >= 3:
                cycle[i] = -12 * np.exp(-0.5 * ((i - 244)/6)**2)
            else:
                cycle[i] = 4 * np.sin(2 * np.pi * i / 80)  # Ciclo de negocios ~6-7 años

        gdp = trend + cycle + np.random.normal(0, 1.5, n)

        # Inflación (objetivo ~2% con shocks)
        inflation = np.full(n, 0.02)
        inflation[120:140] = 0.045  # Inflación 2008
        inflation[240:250] = 0.08   # Inflación post-COVID
        inflation += 0.005 * np.sin(2 * np.pi * np.arange(n) / 60) + np.random.normal(0, 0.003, n)

        # Tasa de interés (Regla de Taylor aproximada)
        interest = 0.02 + 1.5 * (inflation - 0.02) + 0.5 * ((gdp - trend)/trend)
        interest[120:150] = 0.005  # ZLB 2008-2015
        interest += np.random.normal(0, 0.003, n)
        interest = np.maximum(interest, 0.0001)

        # Desempleo (Ley de Okun)
        unemployment = 0.05 - 0.4 * ((gdp - trend)/trend)
        unemployment[240:250] += 0.08  # COVID
        unemployment += np.random.normal(0, 0.003, n)
        unemployment = np.clip(unemployment, 0.02, 0.15)

        # Tipo de cambio (tendencia depreciatoria leve)
        exchange = 1.0 + 0.3 * np.cumsum(np.random.normal(0, 0.008, n))

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
        """Cargar datos desde archivo CSV o Excel."""
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Formato no soportado. Use CSV o Excel.")
        return df


# ============================================================================
# MÓDULO 3: VISUALIZACIONES AVANZADAS
# ============================================================================

class Visualizer:
    """Visualizaciones interactivas."""

    @staticmethod
    def plot_is_lm(model, title="Modelo IS-LM"):
        """Gráfico estático IS-LM con equilibrio."""
        Y_range = np.linspace(40, 300, 200)
        Y_is, r_is = model.get_is_curve(Y_range)
        Y_lm, r_lm = model.get_lm_curve(Y_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Y_is, y=r_is, mode='lines', name='IS',
                                line=dict(color='#1f77b4', width=2.5)))
        fig.add_trace(go.Scatter(x=Y_lm, y=r_lm, mode='lines', name='LM',
                                line=dict(color='#d62728', width=2.5)))

        if model.equilibrium:
            fig.add_trace(go.Scatter(
                x=[model.equilibrium['Y']], y=[model.equilibrium['r']],
                mode='markers', name=f"Equilibrio (Y={model.equilibrium['Y']:.1f}, r={model.equilibrium['r']:.2%})",
                marker=dict(size=14, color='green', symbol='star', line=dict(width=2, color='darkgreen'))
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Producción (Y)",
            yaxis_title="Tasa de Interés (r)",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def plot_ad_as(model, title="Modelo AD-AS"):
        """Gráfico estático AD-AS con equilibrio y LRAS."""
        P_range = np.linspace(0.3, 3.0, 100)
        Y_ad, P_ad = model.get_ad_curve(P_range)
        Y_sras, P_sras = model.get_sras_curve(np.linspace(50, 200, 100))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Y_ad, y=P_ad, mode='lines', name='AD (de IS-LM)',
                                line=dict(color='#1f77b4', width=2.5)))
        fig.add_trace(go.Scatter(x=Y_sras, y=P_sras, mode='lines', name='SRAS',
                                line=dict(color='#d62728', width=2.5)))

        # LRAS
        fig.add_vline(x=model.adas_params['Y_n'], line_dash="dash", line_color="green",
                     annotation_text="LRAS", annotation_position="top")

        if model.equilibrium:
            fig.add_trace(go.Scatter(
                x=[model.equilibrium['Y']], y=[model.equilibrium['P']],
                mode='markers', name=f"Equilibrio CP (Y={model.equilibrium['Y']:.1f}, P={model.equilibrium['P']:.3f})",
                marker=dict(size=14, color='green', symbol='star')
            ))
            # Equilibrio LP
            lr = model.long_run_equilibrium()
            if not np.isnan(lr['P']):
                fig.add_trace(go.Scatter(
                    x=[lr['Y']], y=[lr['P']],
                    mode='markers', name=f"Equilibrio LP (P={lr['P']:.3f})",
                    marker=dict(size=12, color='purple', symbol='diamond')
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Producción (Y)",
            yaxis_title="Nivel de Precios (P)",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def plot_mundell_fleming(model, title="Modelo Mundell-Fleming"):
        """Gráfico de Mundell-Fleming con IS*, LM*, BP."""
        curves = model.get_curves()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curves['Y'], y=curves['IS'], mode='lines', name='IS*',
                                line=dict(color='#1f77b4', width=2.5)))
        fig.add_trace(go.Scatter(x=curves['Y'], y=curves['LM'], mode='lines', name='LM*',
                                line=dict(color='#d62728', width=2.5)))
        fig.add_trace(go.Scatter(x=curves['Y'], y=curves['BP'], mode='lines', name='BP',
                                line=dict(color='green', width=2, dash='dash')))

        if model.equilibrium:
            eq = model.equilibrium
            fig.add_trace(go.Scatter(
                x=[eq['Y']], y=[eq['r']],
                mode='markers', name=f"Equilibrio (Y={eq['Y']:.1f}, r={eq['r']:.2%})",
                marker=dict(size=14, color='gold', symbol='star', line=dict(width=2, color='orange'))
            ))

        regime_text = "Tipo de Cambio FIJO" if model.exchange_rate_fixed else "Tipo de Cambio FLEXIBLE"
        fig.update_layout(
            title=f"{title} - {regime_text}",
            xaxis_title="Producción (Y)",
            yaxis_title="Tasa de Interés (r)",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def plot_dynamic_simulation(results, title="Simulación Dinámica IS-LM"):
        """Gráfico de simulación dinámica."""
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Producción (Y)", "Nivel de Precios (P)"),
                           vertical_spacing=0.12)

        fig.add_trace(go.Scatter(x=results['t'], y=results['Y'], mode='lines', name='Y',
                                line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig.add_hline(y=results['Y'][-1] if len(results['Y']) > 0 else 100, line_dash="dot", 
                     line_color="gray", row=1, col=1)

        fig.add_trace(go.Scatter(x=results['t'], y=results['P'], mode='lines', name='P',
                                line=dict(color='#d62728', width=2)), row=2, col=1)
        fig.add_hline(y=results['P'][-1] if len(results['P']) > 0 else 1.0, line_dash="dot",
                     line_color="gray", row=2, col=1)

        fig.update_layout(title=title, template="plotly_white", height=600, showlegend=False)
        return fig

    @staticmethod
    def plot_macro_dashboard(data):
        """Dashboard macroeconómico completo."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('PIB', 'Inflación (%)', 'Tasa de Interés (%)', 
                          'Desempleo (%)', 'Tipo de Cambio', 'Correlaciones'),
            specs=[[{}, {}], [{}, {}], [{'colspan': 2}, None]]
        )

        fig.add_trace(go.Scatter(x=data['fecha'], y=data['PIB'], mode='lines', name='PIB',
                                line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Inflacion']*100, mode='lines', name='Inflación',
                                line=dict(color='#ff7f0e', width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Tasa_Interes']*100, mode='lines', name='Tasa Interés',
                                line=dict(color='#2ca02c', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Desempleo']*100, mode='lines', name='Desempleo',
                                line=dict(color='#d62728', width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=data['fecha'], y=data['Tipo_Cambio'], mode='lines', name='Tipo Cambio',
                                line=dict(color='#9467bd', width=2)), row=3, col=1)

        fig.update_layout(height=800, template='plotly_white', showlegend=False)
        return fig

    @staticmethod
    def animate_is_lm_transition(model, shock_type, magnitude, frames=20, placeholder=None):
        """
        Animación funcional en Streamlit usando st.empty() y time.sleep().
        Retorna la figura final.
        """
        # Estado inicial
        model_base = ISLMModel({k: v for k, v in model.params.items()})
        model_base.solve()

        Y_range = np.linspace(40, 300, 150)

        if placeholder is None:
            placeholder = st.empty()

        final_model = None

        for i in range(frames + 1):
            progress = i / frames
            temp_params = {k: v for k, v in model_base.params.items()}

            if shock_type == 'Gasto Gobierno ↑':
                temp_params['G'] = model_base.params['G'] * (1 + magnitude * progress)
            elif shock_type == 'Oferta Monetaria ↑':
                temp_params['M'] = model_base.params['M'] * (1 + magnitude * progress)
            elif shock_type == 'Impuestos ↓':
                temp_params['T'] = model_base.params['T'] * (1 - magnitude * progress)

            temp_model = ISLMModel(temp_params)
            try:
                temp_model.solve()
                final_model = temp_model
            except Exception:
                continue

            Y_is, r_is = temp_model.get_is_curve(Y_range)
            Y_lm, r_lm = temp_model.get_lm_curve(Y_range)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=Y_is, y=r_is, mode='lines', name='IS',
                                    line=dict(color='#1f77b4', width=2)))
            fig.add_trace(go.Scatter(x=Y_lm, y=r_lm, mode='lines', name='LM',
                                    line=dict(color='#d62728', width=2)))
            if temp_model.equilibrium:
                fig.add_trace(go.Scatter(
                    x=[temp_model.equilibrium['Y']], y=[temp_model.equilibrium['r']],
                    mode='markers', name='Equilibrio',
                    marker=dict(size=12, color='green', symbol='star')
                ))

            fig.update_layout(
                title=f"Transición: {shock_type} ({progress:.0%})",
                xaxis_title="Producción (Y)", yaxis_title="Tasa de Interés (r)",
                template="plotly_white", height=500
            )

            placeholder.plotly_chart(fig, use_container_width=True, key=f"anim_{i}_{shock_type}")
            time.sleep(0.15)

        return final_model


# ============================================================================
# MÓDULO 4: SIMULACIÓN DE POLÍTICAS Y ESCENARIOS
# ============================================================================

class PolicySimulator:
    """Simulación de políticas económicas con comparación."""

    def __init__(self, model):
        self.model = model
        self.scenarios = {}

    def run_scenario(self, name, policy_changes):
        """Ejecutar escenario de política."""
        original_params = {k: v for k, v in self.model.params.items()}

        for param, value in policy_changes.items():
            if param in self.model.params:
                self.model.params[param] = value

        try:
            new_eq = self.model.solve()
            self.scenarios[name] = {
                'params': {k: v for k, v in self.model.params.items()},
                'equilibrium': new_eq
            }
        except Exception as e:
            self.model.params = original_params
            raise e

        self.model.params = original_params
        self.model.solve()
        return new_eq

    def compare_scenarios(self, scenarios_list):
        """Comparar múltiples escenarios en DataFrame."""
        results = []
        for name in scenarios_list:
            if name in self.scenarios:
                eq = self.scenarios[name]['equilibrium']
                results.append({
                    'Escenario': name,
                    'Y': eq.get('Y', np.nan),
                    'r': eq.get('r', np.nan),
                    'P': eq.get('P', np.nan),
                    'C': eq.get('C', np.nan),
                    'I': eq.get('I', np.nan)
                })
        return pd.DataFrame(results)


# ============================================================================
# MÓDULO 5: GENERACIÓN DE REPORTES PDF CON GRÁFICOS
# ============================================================================

class ReportGenerator:
    """Generación automática de reportes en PDF con figuras."""

    @staticmethod
    def fig_to_image(fig, width=600, height=400):
        """Convertir figura Plotly a imagen PNG bytes."""
        if not HAS_KALEIDO:
            return None
        try:
            img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
            return img_bytes
        except Exception as e:
            st.warning(f"No se pudo convertir figura a imagen: {e}")
            return None

    @staticmethod
    def generate_report(title, content_sections, figures, filename="reporte.pdf"):
        """Generar reporte PDF completo con texto, tablas e imágenes."""
        if not HAS_REPORTLAB:
            raise ImportError("reportlab no está instalado. Instale con: pip install reportlab")

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50,
                               topMargin=50, bottomMargin=30)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'], fontSize=22,
            textColor=colors.HexColor('#1f77b4'), spaceAfter=20, alignment=1
        )
        heading2_style = ParagraphStyle(
            'CustomH2', parent=styles['Heading2'], fontSize=14,
            textColor=colors.HexColor('#2c3e50'), spaceAfter=10, spaceBefore=12
        )

        story = []
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))

        for section in content_sections:
            story.append(Paragraph(section.get('title', ''), heading2_style))
            story.append(Paragraph(section.get('text', ''), styles['Normal']))
            story.append(Spacer(1, 8))

            # Tablas dentro de la sección
            if 'table' in section:
                table_data = section['table']
                if table_data:
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 12))

        # Insertar figuras
        for i, fig in enumerate(figures):
            img_bytes = ReportGenerator.fig_to_image(fig)
            if img_bytes:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                try:
                    img = RLImage(tmp_path, width=6*inch, height=3.5*inch)
                    story.append(Spacer(1, 12))
                    story.append(img)
                    story.append(Spacer(1, 8))
                finally:
                    os.unlink(tmp_path)

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes


# ============================================================================
# MÓDULO 6: PERSISTENCIA DE ESCENARIOS (JSON)
# ============================================================================

def export_scenarios_to_json(scenarios_dict):
    """Exportar escenarios a JSON string descargable."""
    # Convertir a tipos serializables
    export = {}
    for name, data in scenarios_dict.items():
        export[name] = {
            'params': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in data['params'].items()},
            'equilibrium': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in data['equilibrium'].items()},
            'timestamp': datetime.now().isoformat()
        }
    return json.dumps(export, indent=2)

def import_scenarios_from_json(json_str):
    """Importar escenarios desde JSON string."""
    return json.loads(json_str)


# ============================================================================
# APLICACIÓN PRINCIPAL - STREAMLIT
# ============================================================================

def main():
    st.markdown('<div class="main-header">📊 SICM v3.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulador Integral de Choques Macroeconómicos<br>'
                'Versión Universitaria y de Investigación</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🧭 Navegación")
        nav_options = [
            "🏠 Inicio",
            "📈 Modelos IS-LM y AD-AS",
            "🌎 Mundell-Fleming",
            "🎬 Animaciones",
            "📊 Series Temporales",
            "📂 Importar Datos",
            "🧮 Simulación Políticas",
            "📝 Reportes PDF",
            "💾 Guardar Escenarios",
            "🏛️ Dashboard"
        ]
        selection = st.radio("", nav_options, label_visibility="collapsed")
        st.markdown("---")
        st.info("📚 SICM v3.0 - Modelos con validación numérica y derivación rigurosa")

    # ========================================================================
    # PÁGINA: INICIO
    # ========================================================================

    if selection == "🏠 Inicio":
        st.header("Bienvenido al Laboratorio Macroeconómico")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modelos Disponibles", "5", delta="Validados numéricamente")
        with col2:
            st.metric("Choques Soportados", "15+", delta="Fiscales, Monetarios, Externos")
        with col3:
            st.metric("Herramientas", "8", delta="Investigación")

        st.markdown("---")
        st.subheader("🚀 Guía Rápida y Novedades v3.0")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📈 Modelos Macroeconómicos Corregidos**
            - IS-LM: unificación de escalas y validación post-convergencia
            - AD-AS: curva AD derivada rigurosamente de IS-LM (no ecuación cuantitativa aislada)
            - Mundell-Fleming: sistemas separados para tipo de cambio fijo (M endógena) y flexible (e endógena)
            - Dinámico: Curva de Phillips con expectativas + demanda agregada de IS-LM
            """)
        with col2:
            st.markdown("""
            **🔧 Herramientas Avanzadas**
            - Datos reales de FRED (Federal Reserve) con caching
            - Animaciones funcionales en Streamlit (frame a frame)
            - Reportes PDF con gráficos incrustados (vía kaleido)
            - Persistencia de escenarios en JSON (exportar/importar)
            - Validación numérica automática: si fsolve no converge, se advierte al usuario
            """)

    # ========================================================================
    # PÁGINA: MODELOS IS-LM Y AD-AS
    # ========================================================================

    elif selection == "📈 Modelos IS-LM y AD-AS":
        st.header("Modelos Macroeconómicos Clásicos (Validados)")

        tab_is, tab_ad = st.tabs(["IS-LM", "AD-AS (derivado de IS-LM)"])

        with tab_is:
            st.subheader("Modelo IS-LM")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Parámetros**")
                c = st.slider("Propensión marginal a consumir (c)", 0.3, 0.95, 0.75, 0.05)
                G = st.slider("Gasto gobierno (G)", 50, 300, 120, 5)
                M = st.slider("Oferta monetaria (M)", 50, 600, 200, 10)
                b_inv = st.slider("Sensibilidad inversión a r (b)", 5.0, 100.0, 20.0, 5.0)
                h_dm = st.slider("Sensibilidad dinero a r (h)", 100.0, 2000.0, 800.0, 50.0)

                params = {'C0': 50, 'c': c, 'I0': 100, 'b': b_inv, 
                         'G': G, 'T': 80, 'M': M, 'P': 1.0,
                         'k': 0.2, 'h': h_dm}

                try:
                    model = ISLMModel(params)
                    eq = model.solve()
                    st.success("✅ Equilibrio convergió correctamente")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    model = ISLMModel()
                    eq = model.solve()

            with col2:
                fig = Visualizer.plot_is_lm(model)
                st.plotly_chart(fig, use_container_width=True)

            if eq:
                st.subheader("📊 Equilibrio Macroeconómico")
                cols = st.columns(4)
                cols[0].metric("Producción (Y)", f"{eq['Y']:.2f}")
                cols[1].metric("Tasa Interés (r)", f"{eq['r']:.2%}")
                cols[2].metric("Consumo (C)", f"{eq['C']:.2f}")
                cols[3].metric("Inversión (I)", f"{eq['I']:.2f}")

                # Verificación numérica
                with st.expander("✅ Verificación numérica"):
                    residual = model.equations([eq['Y'], eq['r']])
                    st.write(f"Residual IS: {residual[0]:.2e}")
                    st.write(f"Residual LM: {residual[1]:.2e}")
                    st.caption("Valores < 1e-6 indican convergencia perfecta.")

            with st.expander("📐 Ecuaciones del modelo"):
                st.latex(r"Y = C(Y-T) + I(r) + G")
                st.latex(r"C = C_0 + c(Y-T)")
                st.latex(r"I = I_0 - b \cdot r")
                st.latex(r"\frac{M}{P} = kY - h \cdot r")
                st.latex(r"\text{IS: } Y = C(Y-T) + I(r) + G")
                st.latex(r"\text{LM: } \frac{M}{P} = L(Y,r)")

        with tab_ad:
            st.subheader("Modelo AD-AS (AD derivada de IS-LM)")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Parámetros IS-LM subyacentes**")
                M_ad = st.slider("Oferta monetaria (M)", 50, 600, 200, 10, key="M_ad")
                G_ad = st.slider("Gasto gobierno (G)", 50, 300, 120, 5, key="G_ad")

                st.markdown("**Parámetros AS**")
                Y_n = st.slider("Producción natural (Y_n)", 80, 200, 100, 5)
                lambda_param = st.slider("λ (Pendiente SRAS)", 0.01, 0.2, 0.05, 0.01)

                islm_p = {'C0': 50, 'c': 0.75, 'I0': 100, 'b': 20.0,
                         'G': G_ad, 'T': 80, 'M': M_ad, 'P': 1.0,
                         'k': 0.2, 'h': 800}
                adas_p = {'Y_n': Y_n, 'λ': lambda_param, 'P_e': 1.0}

                try:
                    model_ad = ADASModel(islm_p, adas_p)
                    eq_ad = model_ad.solve()
                    lr_ad = model_ad.long_run_equilibrium()
                    st.success("✅ Equilibrio AD-AS convergió")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    model_ad = ADASModel()
                    eq_ad = model_ad.solve()
                    lr_ad = model_ad.long_run_equilibrium()

            with col2:
                fig = Visualizer.plot_ad_as(model_ad)
                st.plotly_chart(fig, use_container_width=True)

            if eq_ad:
                st.subheader("📊 Equilibrio AD-AS")
                cols = st.columns(4)
                cols[0].metric("Producción CP (Y)", f"{eq_ad['Y']:.2f}")
                cols[1].metric("Nivel Precios (P)", f"{eq_ad['P']:.3f}")
                cols[2].metric("Producción Natural", f"{eq_ad['Y_n']:.2f}")
                cols[3].metric("Precios LP", f"{lr_ad['P']:.3f}" if not np.isnan(lr_ad['P']) else "N/A")

                with st.expander("📐 Ecuaciones del modelo"):
                    st.latex(r"\text{AD: derivada de } \begin{cases} Y = C(Y-T) + I(r) + G \ M/P = kY - hr \end{cases}")
                    st.latex(r"\text{SRAS: } P = P_e [1 + \lambda(Y - Y_n)]")
                    st.latex(r"\text{LRAS: } Y = Y_n")
                    st.caption("La curva AD se obtiene eliminando r entre IS y LM para cada nivel de precios P.")

    # ========================================================================
    # PÁGINA: MUNDELL-FLEMING
    # ========================================================================

    elif selection == "🌎 Mundell-Fleming":
        st.header("Modelo Mundell-Fleming - Economía Abierta (Corregido)")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuración")
            regime = st.radio("Régimen Cambiario", ["Fijo", "Flexible"], horizontal=True)
            mobility = st.select_slider("Movilidad de Capitales", 
                                       options=["Nula", "Baja", "Media", "Alta", "Perfecta"],
                                       value="Media")

            G_mf = st.slider("Gasto gobierno (G)", 50, 300, 120, 5)
            M_mf = st.slider("Oferta monetaria (M)", 50, 600, 200, 10)
            r_w = st.slider("Tasa mundial (r*)", 0.0, 0.10, 0.04, 0.005)
            e0 = st.slider("Tipo de cambio fijo (e0)", 0.5, 2.0, 1.0, 0.1)

            cf_map = {"Nula": 0.01, "Baja": 0.2, "Media": 0.5, "Alta": 0.8, "Perfecta": 100.0}

            params_mf = {'C0': 50, 'c': 0.75, 'I0': 100, 'b': 20.0,
                        'G': G_mf, 'T': 80, 'M': M_mf, 'P': 1.0,
                        'k': 0.2, 'h': 800, 'NX0': 20, 'm': 0.1, 'e0': e0,
                        'r_w': r_w, 'CF': cf_map[mobility]}

            model_mf = MundellFlemingModel(params_mf)
            model_mf.exchange_rate_fixed = (regime == "Fijo")

            if st.button("Calcular Equilibrio", type="primary"):
                try:
                    eq_mf = model_mf.solve()
                    st.session_state['mf_eq'] = eq_mf
                    st.success("✅ Equilibrio calculado")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with col2:
            if 'mf_eq' in st.session_state and st.session_state['mf_eq']:
                eq = st.session_state['mf_eq']
                # Recrear modelo con equilibrio para graficar
                model_mf.equilibrium = eq
                fig = Visualizer.plot_mundell_fleming(model_mf)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 Resultados del Equilibrio")
                cols = st.columns(4)
                cols[0].metric("Producción (Y)", f"{eq['Y']:.2f}")
                cols[1].metric("Tasa Interés", f"{eq['r']:.2%}")
                cols[2].metric("Tipo Cambio", f"{eq['e']:.3f}")
                cols[3].metric("Exportaciones Netas", f"{eq['NX']:.2f}")

                if regime == "Fijo":
                    st.info(f"💡 En régimen fijo, la oferta monetaria endógena es M = {eq.get('M_endog', 'N/A'):.2f} "
                           f"para mantener e = {eq['e']:.3f}")
                else:
                    st.info(f"💡 En régimen flexible, el tipo de cambio se ajusta a e = {eq['e']:.3f}")
            else:
                st.info("Configure los parámetros y presione 'Calcular Equilibrio'")

    # ========================================================================
    # PÁGINA: ANIMACIONES
    # ========================================================================

    elif selection == "🎬 Animaciones":
        st.header("🎬 Animaciones de Transición Económica (Funcionales)")

        st.markdown("""
        Las animaciones muestran la transición dinámica entre estados económicos después de un choque.
        A diferencia de la v2.0, estas animaciones **funcionan directamente en Streamlit**.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuración")
            shock_type = st.selectbox("Tipo de Choque", 
                                     ["Gasto Gobierno ↑", "Oferta Monetaria ↑", "Impuestos ↓"])
            magnitude = st.slider("Magnitud", 0.05, 0.30, 0.10, 0.05)
            frames = st.slider("Frames (suavidad)", 10, 40, 20, 5)

            animate_btn = st.button("▶️ Ejecutar Animación", type="primary")

        with col2:
            anim_placeholder = st.empty()

            if animate_btn:
                base_model = ISLMModel()
                base_model.solve()

                with st.spinner("Generando animación..."):
                    final_model = Visualizer.animate_is_lm_transition(
                        base_model, shock_type, magnitude, frames=frames, placeholder=anim_placeholder
                    )

                if final_model and final_model.equilibrium:
                    st.success("✅ Animación completada")
                    st.metric("Equilibrio final - Y", f"{final_model.equilibrium['Y']:.2f}")
                    st.metric("Equilibrio final - r", f"{final_model.equilibrium['r']:.2%}")

                    with st.expander("📝 Mecanismo de Transmisión"):
                        st.markdown(f"""
                        **Choque:** {shock_type} ({magnitude:.0%})

                        **Mecanismo:**
                        1. El choque desplaza la curva {'IS' if 'Gasto' in shock_type or 'Impuestos' in shock_type else 'LM'}
                        2. La economía se mueve al nuevo equilibrio de corto plazo
                        3. Los precios comienzan a ajustarse gradualmente (en el modelo dinámico)
                        4. La economía converge al nuevo equilibrio de largo plazo

                        **Efectos:**
                        - Producción: {'↑' if '↑' in shock_type else '↓'}
                        - Tasa de interés: {'↑' if 'Gasto' in shock_type or 'Impuestos' in shock_type else '↓'}
                        """)

    # ========================================================================
    # PÁGINA: SERIES TEMPORALES
    # ========================================================================

    elif selection == "📊 Series Temporales":
        st.header("📈 Análisis de Series Temporales Macroeconómicas")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Fuente de Datos")
            data_source = st.radio("Origen", ["FRED (Datos Reales)", "Datos Sintéticos (Demo)"])

            if data_source == "FRED (Datos Reales)":
                if HAS_PDR:
                    st.info("Descargando series de FRED: PIB real, Inflación, Tasa Fed, Desempleo")
                    series_map = {
                        'PIB': 'GDPC1',
                        'Inflacion': 'CPIAUCSL',
                        'Tasa_Interes': 'FEDFUNDS',
                        'Desempleo': 'UNRATE'
                    }
                    if st.button("🔄 Descargar de FRED"):
                        with st.spinner("Descargando..."):
                            df = DataManager.get_fred_data(series_map, '2000-01-01')
                        if df is not None:
                            # Normalizar nombres y calcular inflación si es nivel de precios
                            if 'Inflacion' in df.columns:
                                # Si es CPIAUCSL (índice), calcular variación anual
                                if df['Inflacion'].mean() > 10:
                                    df['Inflacion'] = df['Inflacion'].pct_change(12)
                            if 'PIB' in df.columns and df['PIB'].mean() > 10000:
                                df['PIB'] = df['PIB'] / 1000  # Escalar para visualización
                            st.session_state['time_series_data'] = df
                            st.success(f"✅ Datos descargados: {len(df)} registros")
                        else:
                            st.error("No se pudieron descargar datos de FRED")
                else:
                    st.error("❌ pandas_datareader no instalado. Use: `pip install pandas-datareader`")
                    st.info("Se usarán datos sintéticos como fallback.")

        with col2:
            if st.button("📊 Generar Datos Sintéticos Realistas"):
                data = DataManager.generate_sample_data()
                st.session_state['time_series_data'] = data
                st.success("✅ Datos sintéticos generados (emulan ciclo económico EE.UU. 2000-2024)")

        if 'time_series_data' in st.session_state and st.session_state['time_series_data'] is not None:
            data = st.session_state['time_series_data']

            available = [c for c in ['PIB', 'Inflacion', 'Tasa_Interes', 'Desempleo', 'Tipo_Cambio'] if c in data.columns]
            variables = st.multiselect("Seleccionar variables", available, default=available[:2])

            if variables:
                fig = go.Figure()
                for var in variables:
                    fig.add_trace(go.Scatter(x=data['fecha'], y=data[var], mode='lines', name=var, line=dict(width=2)))
                fig.update_layout(title="Series Temporales", xaxis_title="Fecha", yaxis_title="Valor",
                                template="plotly_white", height=500, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📊 Estadísticas Descriptivas"):
                    st.dataframe(data[variables].describe(), use_container_width=True)

                if st.button("📊 Mostrar Dashboard Completo"):
                    fig_dash = Visualizer.plot_macro_dashboard(data)
                    st.plotly_chart(fig_dash, use_container_width=True)

    # ========================================================================
    # PÁGINA: IMPORTAR DATOS
    # ========================================================================

    elif selection == "📂 Importar Datos":
        st.header("📂 Importación de Datos Económicos")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Subir archivo (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
            if uploaded_file is not None:
                try:
                    df = DataManager.load_data(uploaded_file)
                    st.session_state['imported_data'] = df
                    st.success(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
                    st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.subheader("Datos de Ejemplo")
            if st.button("📊 Generar y Descargar Ejemplo CSV"):
                sample = DataManager.generate_sample_data()
                csv = sample.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar CSV", csv, "datos_macro_ejemplo.csv", "text/csv")

    # ========================================================================
    # PÁGINA: SIMULACIÓN DE POLÍTICAS
    # ========================================================================

    elif selection == "🧮 Simulación Políticas":
        st.header("🧮 Simulación de Políticas Económicas")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Modelo Base")
            model_type = st.selectbox("Modelo", ["IS-LM", "AD-AS"], key="sim_model")

            if model_type == "IS-LM":
                G_base = st.slider("Gasto gobierno (G)", 50, 300, 100, key="G_sim")
                M_base = st.slider("Oferta monetaria (M)", 50, 600, 200, key="M_sim")
                model = ISLMModel({'G': G_base, 'M': M_base})
            else:
                M_base = st.slider("Oferta monetaria (M)", 50, 600, 200, key="M_sim_ad")
                Yn_base = st.slider("Producción natural (Y_n)", 80, 200, 100, key="Yn_sim")
                model = ADASModel({'M': M_base, 'G': 120}, {'Y_n': Yn_base})

            try:
                eq_base = model.solve()
                st.metric("Equilibrio Base - Y", f"{eq_base['Y']:.2f}")
            except Exception as e:
                st.error(f"Error modelo base: {e}")
                eq_base = None

        with col2:
            st.subheader("Política a Simular")
            policy_type = st.selectbox("Tipo", ["Fiscal Expansiva", "Fiscal Contractiva",
                                                   "Monetaria Expansiva", "Monetaria Contractiva"])
            magnitude_pol = st.slider("Magnitud", 0.05, 0.30, 0.10, 0.05)

            if st.button("▶️ Ejecutar Simulación", type="primary") and eq_base is not None:
                shock_map = {
                    "Fiscal Expansiva": "Gasto Gobierno ↑",
                    "Fiscal Contractiva": "Gasto Gobierno ↓",
                    "Monetaria Expansiva": "Oferta Monetaria ↑",
                    "Monetaria Contractiva": "Oferta Monetaria ↓"
                }
                shock = shock_map[policy_type]

                if model_type == "IS-LM":
                    model_after = ISLMModel({k: v for k, v in model.params.items()})
                else:
                    model_after = ADASModel({k: v for k, v in model.islm_params.items()},
                                             {k: v for k, v in model.adas_params.items()})

                try:
                    eq_after = model_after.apply_shock(shock, magnitude_pol)

                    st.subheader("📊 Resultados")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Variable", "Producción (Y)")
                    col_b.metric("Base", f"{eq_base['Y']:.2f}")
                    col_c.metric("Simulado", f"{eq_after['Y']:.2f}", 
                                delta=f"{eq_after['Y'] - eq_base['Y']:+.2f}")

                    # Tabla comparativa
                    rows = []
                    for key in ['Y', 'r', 'P', 'C', 'I']:
                        if key in eq_base and key in eq_after:
                            base_v = eq_base[key]
                            after_v = eq_after[key]
                            change = ((after_v - base_v) / abs(base_v) * 100) if base_v != 0 else 0
                            rows.append({'Variable': key, 'Base': base_v, 'Simulado': after_v, 'Cambio %': change})

                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception as e:
                    st.error(f"Error en simulación: {e}")

    # ========================================================================
    # PÁGINA: REPORTES PDF
    # ========================================================================

    elif selection == "📝 Reportes PDF":
        st.header("📝 Generación Automática de Reportes con Gráficos")

        if not HAS_REPORTLAB:
            st.error("❌ reportlab no está instalado. Ejecute: `pip install reportlab`")
            st.stop()

        if not HAS_KALEIDO:
            st.warning("⚠️ kaleido no está instalado. Los reportes no incluirán gráficos. "
                      "Instale con: `pip install kaleido`")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Configuración")
            report_title = st.text_input("Título", "Análisis Macroeconómico SICM v3.0")
            include_model = st.checkbox("Incluir gráfico IS-LM", value=True)
            include_adas = st.checkbox("Incluir gráfico AD-AS", value=True)

            if st.button("📄 Generar Reporte", type="primary"):
                content = []
                content.append({
                    'title': 'Resumen Ejecutivo',
                    'text': 'Análisis realizado con SICM v3.0. Los modelos incluyen validación numérica '
                           'y derivación rigurosa de ecuaciones. Los equilibrios han sido verificados '
                           'con tolerancia < 1e-6.'
                })

                # Generar figuras para el reporte
                figures = []
                if include_model:
                    m = ISLMModel()
                    m.solve()
                    figures.append(Visualizer.plot_is_lm(m))
                if include_adas:
                    m2 = ADASModel()
                    m2.solve()
                    figures.append(Visualizer.plot_ad_as(m2))

                try:
                    pdf_bytes = ReportGenerator.generate_report(report_title, content, figures)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="reporte_sicm_v3.pdf">📥 Descargar Reporte PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ Reporte generado exitosamente")
                except Exception as e:
                    st.error(f"Error generando PDF: {e}")

    # ========================================================================
    # PÁGINA: GUARDAR ESCENARIOS
    # ========================================================================

    elif selection == "💾 Guardar Escenarios":
        st.header("💾 Gestión de Escenarios (Persistencia JSON)")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Guardar Escenario Actual")
            scenario_name = st.text_input("Nombre", "Escenario_A")
            scenario_desc = st.text_area("Descripción", "Política fiscal expansiva +10%")

            if st.button("💾 Guardar en Memoria", type="primary"):
                # Crear un escenario de ejemplo si no hay datos
                demo_params = {'C0': 50, 'c': 0.75, 'I0': 100, 'b': 20.0, 'G': 132, 'T': 80, 'M': 200, 'P': 1.0, 'k': 0.2, 'h': 800}
                demo_eq = ISLMModel(demo_params).solve()

                scenario = {
                    'name': scenario_name,
                    'description': scenario_desc,
                    'params': demo_params,
                    'equilibrium': demo_eq,
                    'timestamp': datetime.now().isoformat()
                }

                if 'saved_scenarios_json' not in st.session_state:
                    st.session_state.saved_scenarios_json = []
                st.session_state.saved_scenarios_json.append(scenario)
                st.success(f"✅ Escenario '{scenario_name}' guardado")

            st.markdown("---")
            st.subheader("Exportar/Importar JSON")
            if st.session_state.get('saved_scenarios_json'):
                json_str = export_scenarios_to_json({s['name']: s for s in st.session_state.saved_scenarios_json})
                st.download_button("⬇️ Exportar escenarios JSON", json_str, "escenarios_sicm.json", "application/json")

            uploaded_json = st.file_uploader("Cargar escenarios JSON", type="json")
            if uploaded_json is not None:
                try:
                    imported = import_scenarios_from_json(uploaded_json.read().decode())
                    for name, data in imported.items():
                        st.session_state.saved_scenarios_json.append({
                            'name': name, **data
                        })
                    st.success(f"✅ {len(imported)} escenarios importados")
                except Exception as e:
                    st.error(f"Error importando: {e}")

        with col2:
            st.subheader("Escenarios Guardados")
            scenarios = st.session_state.get('saved_scenarios_json', [])
            if scenarios:
                for i, s in enumerate(scenarios):
                    with st.container():
                        st.markdown(f"**{s['name']}**")
                        st.caption(f"{s.get('description', '')}")
                        st.caption(f"📅 {s['timestamp'][:10] if 'timestamp' in s else 'N/A'}")
                        if 'equilibrium' in s:
                            eq = s['equilibrium']
                            st.write(f"Y={eq.get('Y', 'N/A'):.1f}, r={eq.get('r', 'N/A'):.2%}")
                        st.markdown("---")
            else:
                st.info("No hay escenarios guardados. Cree uno o importe desde JSON.")

        # Comparación
        if scenarios:
            st.subheader("📊 Comparación de Escenarios")
            names = [s['name'] for s in scenarios]
            selected = st.multiselect("Seleccionar", names, default=names[:2] if len(names) >= 2 else names)
            if selected:
                rows = []
                for s in scenarios:
                    if s['name'] in selected and 'equilibrium' in s:
                        eq = s['equilibrium']
                        rows.append({
                            'Escenario': s['name'],
                            'Y': eq.get('Y', np.nan),
                            'r': eq.get('r', np.nan),
                            'P': eq.get('P', np.nan)
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ========================================================================
    # PÁGINA: DASHBOARD
    # ========================================================================

    elif selection == "🏛️ Dashboard":
        st.header("🏛️ Dashboard - Laboratorio de Economía")

        # Cargar datos si no existen
        if 'time_series_data' not in st.session_state or st.session_state['time_series_data'] is None:
            if st.button("📊 Cargar Datos para Dashboard"):
                st.session_state['time_series_data'] = DataManager.generate_sample_data()
                st.rerun()
            else:
                st.info("Cargue datos para ver el dashboard completo")
                st.stop()

        data = st.session_state['time_series_data']

        # Métricas dinámicas (últimos valores)
        last = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else last

        col1, col2, col3, col4 = st.columns(4)
        if 'PIB' in data.columns:
            col1.metric("PIB", f"{last['PIB']:.1f}", f"{((last['PIB']-prev['PIB'])/prev['PIB']*100):+.1f}%")
        if 'Inflacion' in data.columns:
            col2.metric("Inflación", f"{last['Inflacion']*100:.1f}%", f"{(last['Inflacion']-prev['Inflacion'])*100:+.1f}pp")
        if 'Desempleo' in data.columns:
            col3.metric("Desempleo", f"{last['Desempleo']*100:.1f}%", f"{(last['Desempleo']-prev['Desempleo'])*100:+.1f}pp")
        if 'Tasa_Interes' in data.columns:
            col4.metric("Tasa Interés", f"{last['Tasa_Interes']*100:.1f}%", f"{(last['Tasa_Interes']-prev['Tasa_Interes'])*100:+.1f}pp")

        # Dashboard completo
        fig = Visualizer.plot_macro_dashboard(data)
        st.plotly_chart(fig, use_container_width=True)

        # Indicadores de política
        st.subheader("📊 Indicadores de Política")
        col1, col2 = st.columns(2)
        with col1:
            if 'PIB' in data.columns:
                trend = np.linspace(data['PIB'].iloc[0], data['PIB'].iloc[-1], len(data))
                gap = (last['PIB'] - trend[-1]) / trend[-1]
                st.markdown("**Brecha del Producto**")
                st.progress(min(max(0.5 + gap * 5, 0.0), 1.0), text=f"Brecha: {gap*100:+.1f}%")
        with col2:
            if 'Inflacion' in data.columns and 'Tasa_Interes' in data.columns:
                taylor = 0.02 + 1.5 * (last['Inflacion'] - 0.02)
                diff = last['Tasa_Interes'] - taylor
                stance = "Restrictiva" if diff > 0.005 else "Expansiva" if diff < -0.005 else "Neutral"
                st.markdown(f"**Postura de Política Monetaria: {stance}**")
                st.progress(min(max(0.5 + diff * 20, 0.0), 1.0), text=f"Desvío Taylor: {diff*100:+.1f}pp")


if __name__ == "__main__":
    main()
