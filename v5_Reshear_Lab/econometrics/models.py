"""
SICM v5 Research Lab — Econometría
===================================
ARIMA, VAR, VECM, Monte Carlo.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class ForecastResult:
    """Resultado de pronóstico"""
    forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence: float = 0.95


class ARIMAModel:
    """
    Modelo ARIMA para pronóstico de series temporales.
    
    ARIMA(p,d,q): AutoRegressive Integrated Moving Average
    """
    
    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        self.p = p  # Orden AR
        self.d = d  # Orden de diferenciación
        self.q = q  # Orden MA
        self.params = None
        self.residuals = None
    
    def _difference(self, series: np.ndarray, d: int = 1) -> np.ndarray:
        """Aplica diferenciación d veces"""
        diff = series.copy()
        for _ in range(d):
            diff = np.diff(diff)
        return diff
    
    def _fit_ar(self, series: np.ndarray) -> np.ndarray:
        """
        Estima parámetros AR(p) usando Yule-Walker.
        
        Returns:
            Array de coeficientes AR
        """
        n = len(series)
        r = np.correlate(series - series.mean(), series - series.mean(), mode='full')
        r = r[n-1:] / n  # Autocorrelaciones
        
        # Matriz de Toeplitz
        R = np.array([[r[abs(i-j)] for j in range(self.p)] for i in range(self.p)])
        rho = r[1:self.p+1]
        
        try:
            phi = np.linalg.solve(R, rho)
        except np.linalg.LinAlgError:
            phi = np.zeros(self.p)
        
        return phi
    
    def fit(self, series: np.ndarray) -> 'ARIMAModel':
        """
        Ajusta el modelo ARIMA a los datos.
        
        Args:
            series: Serie temporal univariada
        
        Returns:
            self
        """
        # Diferenciación
        if self.d > 0:
            diff_series = self._difference(series, self.d)
        else:
            diff_series = series
        
        # Estimar AR
        if self.p > 0:
            phi = self._fit_ar(diff_series)
        else:
            phi = np.array([])
        
        # Calcular residuos (simplificado para MA)
        residuals = diff_series[self.p:] - np.mean(diff_series)
        if len(phi) > 0:
            for i in range(self.p, len(diff_series)):
                pred = np.dot(phi, diff_series[i-self.p:i][::-1])
                residuals[i-self.p] = diff_series[i] - pred
        
        self.params = {
            'phi': phi,
            'mu': np.mean(diff_series),
            'sigma': np.std(residuals)
        }
        self.residuals = residuals
        
        return self
    
    def forecast(self, steps: int = 10, confidence: float = 0.95) -> ForecastResult:
        """
        Genera pronóstico con intervalos de confianza.
        
        Args:
            steps: Número de períodos a pronosticar
            confidence: Nivel de confianza
        
        Returns:
            ForecastResult con pronóstico e intervalos
        """
        if self.params is None:
            raise ValueError("Modelo no ajustado. Ejecute fit() primero.")
        
        phi = self.params['phi']
        mu = self.params['mu']
        sigma = self.params['sigma']
        
        # Pronóstico simple (AR)
        forecast = np.full(steps, mu)
        
        # Intervalos de confianza
        z = stats.norm.ppf((1 + confidence) / 2)
        se = sigma * np.sqrt(np.arange(1, steps + 1))
        
        lower = forecast - z * se
        upper = forecast + z * se
        
        return ForecastResult(
            forecast=forecast,
            lower_bound=lower,
            upper_bound=upper,
            confidence=confidence
        )
    
    def summary(self) -> Dict:
        """Resumen del modelo ARIMA"""
        return {
            'orden': f"ARIMA({self.p},{self.d},{self.q})",
            'parametros': self.params,
            'sigma_residuals': self.params['sigma'] if self.params else None
        }


class VARModel:
    """
    Modelo Vector Autoregresivo para análisis de transmisión dinámica.
    
    VAR(p): Sistema de ecuaciones AR interconectadas
    """
    
    def __init__(self, p: int = 1):
        self.p = p  # Orden del VAR
        self.coefs = None
        self.residuals = None
        self.var_names = None
    
    def fit(self, data: pd.DataFrame) -> 'VARModel':
        """
        Ajusta VAR(p) usando OLS ecuación por ecuación.
        
        Args:
            data: DataFrame con series temporales multivariadas
        
        Returns:
            self
        """
        self.var_names = list(data.columns)
        n_vars = len(self.var_names)
        T = len(data)
        
        # Construir matriz de rezagos
        Y = data.values[self.p:]  # Observaciones contemporáneas
        Z = np.zeros((T - self.p, n_vars * self.p + 1))
        Z[:, 0] = 1  # Constante
        
        for i in range(self.p):
            Z[:, 1 + i*n_vars:(i+1)*n_vars + 1] = data.values[self.p-1-i:T-1-i]
        
        # OLS ecuación por ecuación
        self.coefs = np.zeros((n_vars, n_vars * self.p + 1))
        self.residuals = np.zeros((T - self.p, n_vars))
        
        for i in range(n_vars):
            y_i = Y[:, i]
            beta = np.linalg.lstsq(Z, y_i, rcond=None)[0]
            self.coefs[i] = beta
            self.residuals[:, i] = y_i - Z @ beta
        
        return self
    
    def impulse_response(self, variable: str, steps: int = 10) -> pd.DataFrame:
        """
        Calcula función de respuesta al impulso (IRF).
        
        Args:
            variable: Variable que recibe el choque
            steps: Horizonte de respuesta
        
        Returns:
            DataFrame con respuestas
        """
        if self.coefs is None:
            raise ValueError("Modelo no ajustado")
        
        idx = self.var_names.index(variable)
        n_vars = len(self.var_names)
        
        # Simular respuesta a un choque unitario
        response = np.zeros((steps, n_vars))
        shock = np.zeros(n_vars)
        shock[idx] = 1  # Choque unitario
        
        # Simulación recursiva
        history = np.zeros((self.p, n_vars))
        for t in range(steps):
            # Construir vector de rezagos
            lags = history.flatten()
            pred = self.coefs[:, 1:] @ lags + self.coefs[:, 0]
            if t == 0:
                pred += shock
            response[t] = pred
            # Actualizar historia
            history = np.vstack([history[1:], pred])
        
        return pd.DataFrame(response, columns=self.var_names)
    
    def forecast(self, data: pd.DataFrame, steps: int = 10) -> pd.DataFrame:
        """
        Pronóstico fuera de muestra.
        
        Args:
            data: Datos históricos
            steps: Períodos a pronosticar
        
        Returns:
            DataFrame con pronósticos
        """
        if self.coefs is None:
            raise ValueError("Modelo no ajustado")
        
        n_vars = len(self.var_names)
        forecast = np.zeros((steps, n_vars))
        
        # Últimos p valores
        history = data.values[-self.p:].copy()
        
        for t in range(steps):
            lags = history.flatten()
            pred = self.coefs[:, 1:] @ lags + self.coefs[:, 0]
            forecast[t] = pred
            history = np.vstack([history[1:], pred])
        
        return pd.DataFrame(forecast, columns=self.var_names)


class VECMModel:
    """
    Vector Error Correction Model para cointegración.
    
    VECM: Captura relaciones de largo plazo entre variables cointegradas
    """
    
    def __init__(self, r: int = 1):
        self.r = r  # Rango de cointegración
        self.pi = None  # Matriz de cointegración
        self.gamma = None  # Coeficientes de corto plazo
        self.alpha = None  # Velocidad de ajuste
        self.beta = None  # Vectores de cointegración
    
    def _johansen_test(self, data: pd.DataFrame) -> Dict:
        """
        Test de cointegración de Johansen (simplificado).
        
        Returns:
            Dict con resultado del test
        """
        # Implementación simplificada
        # En producción usar statsmodels.tsa.vector_ar.vecm.coint_johansen
        
        n_vars = len(data.columns)
        
        # Diferenciación
        diff = data.diff().dropna()
        levels = data.iloc[:-1]
        
        # Regresión auxiliar
        X = levels.values
        Y = diff.values
        
        # Estimación simplificada de Pi
        Pi = np.linalg.lstsq(X, Y, rcond=None)[0].T
        
        # Eigenvalores
        eigenvalues = np.linalg.eigvals(Pi @ Pi.T)
        
        return {
            'rango_estimado': min(self.r, n_vars),
            'eigenvalues': eigenvalues,
            'Pi': Pi
        }
    
    def fit(self, data: pd.DataFrame) -> 'VECMModel':
        """
        Ajusta el modelo VECM.
        
        Args:
            data: DataFrame con variables cointegradas
        
        Returns:
            self
        """
        # Test de cointegración
        johansen = self._johansen_test(data)
        self.pi = johansen['Pi']
        
        # Descomposición Pi = alpha * beta'
        # Simplificación: usar SVD
        U, S, Vt = np.linalg.svd(self.pi)
        self.alpha = U[:, :self.r] * S[:self.r]
        self.beta = Vt[:self.r].T
        
        # Estimar gamma (coeficientes de corto plazo)
        diff = data.diff().dropna()
        lags = diff.iloc[:-1].values
        dep = diff.iloc[1:].values
        
        # OLS para gamma
        ecm = (data.iloc[:-2].values @ self.beta)  # Error de corrección
        X = np.column_stack([ecm, lags])
        gamma = np.linalg.lstsq(X, dep, rcond=None)[0]
        
        self.gamma = gamma
        
        return self
    
    def error_correction_term(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el término de corrección de error.
        
        Args:
            data: Datos de nivel
        
        Returns:
            DataFrame con ECT
        """
        if self.beta is None:
            raise ValueError("Modelo no ajustado")
        
        ect = data.values @ self.beta
        
        return pd.DataFrame(ect, columns=[f'ECT_{i+1}' for i in range(self.r)])


class MonteCarloSimulation:
    """
    Simulación de Monte Carlo para análisis de incertidumbre.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.results = None
    
    def simulate_islm(self, config: 'EconomyConfig', 
                     param_shocks: Optional[Dict] = None) -> pd.DataFrame:
        """
        Simula incertidumbre en parámetros del modelo IS-LM.
        
        Args:
            config: Configuración base
            param_shocks: Dict con desviaciones estándar de parámetros
        
        Returns:
            DataFrame con distribución de resultados
        """
        from ..core.equilibrium import EquilibriumSolver
        
        if param_shocks is None:
            param_shocks = {
                'c1': 0.05,
                'b': 2.0,
                'k': 0.05,
                'h': 1.0
            }
        
        results = []
        
        for _ in range(self.n_simulations):
            # Perturbar parámetros
            import copy
            sim_config = copy.deepcopy(config)
            
            if 'c1' in param_shocks:
                sim_config.consumo.c1 += np.random.normal(0, param_shocks['c1'])
                sim_config.consumo.c1 = np.clip(sim_config.consumo.c1, 0, 1)
            
            if 'b' in param_shocks:
                sim_config.inversion.b += np.random.normal(0, param_shocks['b'])
                sim_config.inversion.b = max(0.1, sim_config.inversion.b)
            
            if 'k' in param_shocks:
                sim_config.dinero.k += np.random.normal(0, param_shocks['k'])
                sim_config.dinero.k = max(0.1, sim_config.dinero.k)
            
            if 'h' in param_shocks:
                sim_config.dinero.h += np.random.normal(0, param_shocks['h'])
                sim_config.dinero.h = max(0.1, sim_config.dinero.h)
            
            # Calcular equilibrio
            try:
                solver = EquilibriumSolver(sim_config)
                result = solver.solve_islm_cerrado()
                results.append({
                    'Y': result.Y,
                    'r': result.r,
                    'C': result.C,
                    'I': result.I
                })
            except Exception:
                continue
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def get_confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Calcula intervalos de confianza de los resultados.
        
        Args:
            confidence: Nivel de confianza
        
        Returns:
            DataFrame con percentiles
        """
        if self.results is None:
            raise ValueError("Ejecute simulate primero")
        
        alpha = 1 - confidence
        lower = alpha / 2
        upper = 1 - alpha / 2
        
        intervals = {}
        for col in self.results.columns:
            intervals[col] = {
                'mean': self.results[col].mean(),
                'median': self.results[col].median(),
                'std': self.results[col].std(),
                'lower': self.results[col].quantile(lower),
                'upper': self.results[col].quantile(upper)
            }
        
        return pd.DataFrame(intervals).T
    
    def plot_distribution(self, variable: str = 'Y') -> go.Figure:
        """
        Genera histograma de la distribución simulada.
        
        Args:
            variable: Variable a graficar
        
        Returns:
            Figura de Plotly
        """
        import plotly.graph_objects as go
        
        if self.results is None:
            raise ValueError("Ejecute simulate primero")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.results[variable],
            nbinsx=50,
            name=f'Distribución de {variable}',
            marker_color='#2E86AB',
            opacity=0.75
        ))
        
        # Línea de media
        mean_val = self.results[variable].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Media: {mean_val:.2f}")
        
        fig.update_layout(
            title=f'Distribución de {variable} - Simulación Monte Carlo (n={self.n_simulations})',
            xaxis_title=variable,
            yaxis_title='Frecuencia',
            template='plotly_white'
        )
        
        return fig
