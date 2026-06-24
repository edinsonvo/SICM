"""
SICM v5 Research Lab — Econometría
====================================
Fase 6: ARIMA, VAR, VECM, Monte Carlo.

Modelos implementados:
- ARIMA(p,d,q): Pronóstico univariado
- VAR(p): Transmisión dinámica multivariada
- VECM: Cointegración
- Monte Carlo: Simulación de incertidumbre
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class ForecastResult:
    """Resultado de pronóstico con intervalos de confianza"""
    forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence: float = 0.95

    def to_dataframe(self, index: Optional[pd.Index] = None) -> pd.DataFrame:
        """Convierte a DataFrame"""
        df = pd.DataFrame({
            'forecast': self.forecast,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound
        })
        if index is not None:
            df.index = index
        return df


class ARIMAModel:
    """
    Modelo ARIMA para pronóstico de series temporales.

    ARIMA(p,d,q): AutoRegressive Integrated Moving Average

    Ecuación:
        (1 - Σφ_i L^i)(1 - L)^d y_t = c + (1 + Σθ_i L^i)ε_t

    Args:
        p: Orden autorregresivo
        d: Orden de diferenciación
        q: Orden de media móvil
    """

    def __init__(self, p: int = 1, d: int = 1, q: int = 0):
        self.p = p
        self.d = d
        self.q = q
        self.phi = None       # Coeficientes AR
        self.theta = None     # Coeficientes MA
        self.mu = None        # Media
        self.sigma = None     # Desv. estándar de residuos
        self.residuals = None
        self.series_diff = None

    def _difference(self, series: np.ndarray, d: int = 1) -> np.ndarray:
        """Aplica diferenciación d veces"""
        diff = series.copy().astype(float)
        for _ in range(d):
            diff = np.diff(diff)
        return diff

    def _fit_ar(self, series: np.ndarray) -> np.ndarray:
        """
        Estima coeficientes AR(p) usando ecuaciones de Yule-Walker.

        Resuelve el sistema: R * φ = r
        donde R es la matriz de autocovarianzas y r el vector de autocovarianzas.
        """
        n = len(series)
        series_c = series - np.mean(series)

        # Autocovarianzas
        gamma = np.array([
            np.mean(series_c[i:] * series_c[:-i]) if i > 0 else np.var(series_c)
            for i in range(self.p + 1)
        ])

        # Matriz de Toeplitz
        R = np.array([[gamma[abs(i-j)] for j in range(self.p)] for i in range(self.p)])
        r = gamma[1:self.p+1]

        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            phi = np.zeros(self.p)

        return phi

    def fit(self, series: np.ndarray) -> 'ARIMAModel':
        """
        Ajusta el modelo ARIMA a los datos.

        Args:
            series: Serie temporal univariada (array 1D)

        Returns:
            self
        """
        series = np.asarray(series, dtype=float)

        # Diferenciación
        if self.d > 0:
            self.series_diff = self._difference(series, self.d)
        else:
            self.series_diff = series.copy()

        # Estimar componente AR
        if self.p > 0:
            self.phi = self._fit_ar(self.series_diff)
        else:
            self.phi = np.array([])

        # Calcular residuos (aproximación para MA simplificada)
        y = self.series_diff
        n = len(y)

        if len(self.phi) > 0:
            # Predicción AR
            y_pred = np.full(n, np.mean(y))
            for i in range(self.p, n):
                y_pred[i] = np.mean(y) + np.dot(self.phi, y[i-self.p:i][::-1] - np.mean(y))
            self.residuals = y - y_pred
        else:
            self.residuals = y - np.mean(y)

        self.mu = np.mean(self.series_diff)
        self.sigma = np.std(self.residuals)

        return self

    def forecast(self, steps: int = 10, confidence: float = 0.95) -> ForecastResult:
        """
        Genera pronóstico con intervalos de confianza.

        Args:
            steps: Número de períodos a pronosticar
            confidence: Nivel de confianza (ej. 0.95 para 95%)

        Returns:
            ForecastResult con pronóstico e intervalos
        """
        if self.phi is None:
            raise ValueError("Modelo no ajustado. Ejecute fit() primero.")

        # Pronóstico recursivo AR
        y_hist = self.series_diff.copy()
        forecast = np.zeros(steps)

        for t in range(steps):
            if len(self.phi) > 0:
                pred = self.mu + np.dot(self.phi, y_hist[-self.p:][::-1] - self.mu)
            else:
                pred = self.mu
            forecast[t] = pred
            y_hist = np.append(y_hist, pred)

        # Reintegrar diferenciación
        if self.d > 0:
            # Simplificación: asumimos que el último valor conocido es el punto de partida
            last_val = self.series_diff[-1] if len(self.series_diff) > 0 else self.mu
            forecast = np.cumsum(forecast) + last_val

        # Intervalos de confianza
        z = stats.norm.ppf((1 + confidence) / 2)
        se = self.sigma * np.sqrt(np.arange(1, steps + 1))

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
            'phi': self.phi.tolist() if self.phi is not None else None,
            'mu': float(self.mu) if self.mu is not None else None,
            'sigma': float(self.sigma) if self.sigma is not None else None,
            'n_obs': len(self.series_diff) if self.series_diff is not None else 0
        }


class VARModel:
    """
    Vector Autoregresivo para análisis de transmisión dinámica.

    VAR(p): Sistema de ecuaciones AR interconectadas

    Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + ε_t

    Args:
        p: Orden del VAR (número de rezagos)
    """

    def __init__(self, p: int = 1):
        self.p = p
        self.coefs = None          # Coeficientes [n_vars x (n_vars*p + 1)]
        self.residuals = None      # Residuos
        self.var_names = None      # Nombres de variables
        self.sigma_u = None        # Matriz de covarianza de residuos

    def fit(self, data: pd.DataFrame) -> 'VARModel':
        """
        Ajusta VAR(p) usando MCO ecuación por ecuación.

        Args:
            data: DataFrame con series temporales multivariadas
                  (filas=tiempo, columnas=variables)

        Returns:
            self
        """
        self.var_names = list(data.columns)
        n_vars = len(self.var_names)
        T = len(data)

        if T <= self.p:
            raise ValueError(f"Número de observaciones ({T}) debe ser > orden del modelo ({self.p})")

        # Construir matriz de rezagos
        Y = data.values[self.p:]  # Observaciones contemporáneas (T-p x n_vars)

        # Matriz de regresores: [1, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
        n_regressors = n_vars * self.p + 1
        Z = np.zeros((T - self.p, n_regressors))
        Z[:, 0] = 1  # Constante

        for lag in range(self.p):
            start_col = 1 + lag * n_vars
            end_col = start_col + n_vars
            Z[:, start_col:end_col] = data.values[self.p - 1 - lag:T - 1 - lag]

        # MCO ecuación por ecuación
        self.coefs = np.zeros((n_vars, n_regressors))
        self.residuals = np.zeros((T - self.p, n_vars))

        for i in range(n_vars):
            y_i = Y[:, i]
            beta = np.linalg.lstsq(Z, y_i, rcond=None)[0]
            self.coefs[i] = beta
            self.residuals[:, i] = y_i - Z @ beta

        # Matriz de covarianza de residuos
        self.sigma_u = np.cov(self.residuals, rowvar=False)

        return self

    def impulse_response(self, variable: str, steps: int = 10, 
                         shock_size: float = 1.0) -> pd.DataFrame:
        """
        Calcula Función de Respuesta al Impulso (IRF).

        Simula el efecto de un choque unitario en una variable
        sobre todas las variables del sistema.

        Args:
            variable: Variable que recibe el choque
            steps: Horizonte de respuesta
            shock_size: Magnitud del choque

        Returns:
            DataFrame con respuestas (filas=tiempo, cols=variables)
        """
        if self.coefs is None:
            raise ValueError("Modelo no ajustado. Ejecute fit() primero.")

        idx = self.var_names.index(variable)
        n_vars = len(self.var_names)

        # Simular respuesta a choque unitario
        response = np.zeros((steps, n_vars))

        # Historia de rezagos (inicialmente cero)
        history = np.zeros((self.p, n_vars))

        for t in range(steps):
            # Construir vector de rezagos aplanado
            lags_flat = history.flatten()

            # Predicción base
            pred = self.coefs[:, 0] + self.coefs[:, 1:] @ lags_flat

            # Aplicar choque en t=0
            if t == 0:
                pred[idx] += shock_size

            response[t] = pred

            # Actualizar historia (desplazar y agregar nuevo valor)
            history = np.vstack([history[1:], pred])

        return pd.DataFrame(response, columns=self.var_names)

    def forecast(self, data: pd.DataFrame, steps: int = 10) -> pd.DataFrame:
        """
        Pronóstico fuera de muestra.

        Args:
            data: Datos históricos (últimos p valores se usan como inicialización)
            steps: Períodos a pronosticar

        Returns:
            DataFrame con pronósticos
        """
        if self.coefs is None:
            raise ValueError("Modelo no ajustado")

        n_vars = len(self.var_names)
        forecast = np.zeros((steps, n_vars))

        # Últimos p valores como historia inicial
        history = data.values[-self.p:].copy()

        for t in range(steps):
            lags_flat = history.flatten()
            pred = self.coefs[:, 0] + self.coefs[:, 1:] @ lags_flat
            forecast[t] = pred
            history = np.vstack([history[1:], pred])

        # Crear índice de fechas
        last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else len(data) - 1
        if isinstance(last_date, (int, np.integer)):
            idx = range(last_date + 1, last_date + 1 + steps)
        else:
            idx = pd.date_range(start=last_date, periods=steps + 1, freq='Q')[1:]

        return pd.DataFrame(forecast, columns=self.var_names, index=idx)

    def granger_causality(self, cause: str, effect: str, max_lag: int = None) -> Dict:
        """
        Test de causalidad de Granger (simplificado).

        Args:
            cause: Variable causal
            effect: Variable efecto
            max_lag: Máximo rezago a probar

        Returns:
            Dict con resultado del test
        """
        if max_lag is None:
            max_lag = self.p

        # Simplificación: usar correlación cruzada como proxy
        cause_idx = self.var_names.index(cause)
        effect_idx = self.var_names.index(effect)

        # Correlación contemporánea y con rezagos
        corr_cont = np.corrcoef(self.residuals[:, cause_idx], 
                                self.residuals[:, effect_idx])[0, 1]

        return {
            'cause': cause,
            'effect': effect,
            'correlation': float(corr_cont),
            'interpretation': f"{'Posible' if abs(corr_cont) > 0.3 else 'Débil'} causalidad de Granger"
        }


class VECMModel:
    """
    Vector Error Correction Model para cointegración.

    VECM captura relaciones de largo plazo entre variables cointegradas:
        ΔY_t = αβ'Y_{t-1} + ΣΓ_i ΔY_{t-i} + ε_t

    Args:
        r: Rango de cointegración (número de relaciones de largo plazo)
    """

    def __init__(self, r: int = 1):
        self.r = r
        self.pi = None           # Matriz Pi = αβ'
        self.alpha = None        # Velocidad de ajuste
        self.beta = None         # Vectores de cointegración
        self.gamma = None        # Coeficientes de corto plazo
        self.residuals = None

    def _johansen_lr_test(self, data: pd.DataFrame) -> Dict:
        """
        Test de cointegración de Johansen (implementación simplificada).

        En producción usar: statsmodels.tsa.vector_ar.vecm.coint_johansen

        Args:
            data: DataFrame con variables en niveles

        Returns:
            Dict con eigenvalores y rango estimado
        """
        n_vars = len(data.columns)
        T = len(data)

        # Diferenciación
        diff = data.diff().dropna()
        levels = data.iloc[:-1]

        # Regresión auxiliar: ΔY_t = Π Y_{t-1} + ε_t
        X = levels.values
        Y = diff.values

        # Estimación de Pi (simplificada)
        Pi = np.linalg.lstsq(X, Y, rcond=None)[0].T  # (n_vars x n_vars)

        # Eigenvalores de Pi
        eigenvalues = np.linalg.eigvals(Pi @ Pi.T)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Ordenar descendente

        return {
            'rango_estimado': min(self.r, n_vars),
            'eigenvalues': eigenvalues,
            'Pi': Pi
        }

    def fit(self, data: pd.DataFrame) -> 'VECMModel':
        """
        Ajusta el modelo VECM.

        Args:
            data: DataFrame con variables en niveles (deben ser I(1) y cointegradas)

        Returns:
            self
        """
        # Test de cointegración
        johansen = self._johansen_lr_test(data)
        self.pi = johansen['Pi']

        # Descomposición Pi = α * β' vía SVD
        # SVD: Pi = U S V' => α = U[:, :r] * sqrt(S[:r]), β = V[:, :r] * sqrt(S[:r])
        U, S, Vt = np.linalg.svd(self.pi)

        sqrt_S = np.sqrt(S[:self.r])
        self.alpha = U[:, :self.r] * sqrt_S
        self.beta = Vt[:self.r].T * sqrt_S

        # Estimar gamma (coeficientes de corto plazo)
        diff = data.diff().dropna()

        # Variable dependiente: ΔY_t
        dep = diff.iloc[1:].values

        # Variables independientes: ECT_{t-1}, ΔY_{t-1}
        ecm = (data.iloc[:-2].values @ self.beta)  # Error de corrección
        lags = diff.iloc[:-1].values

        X = np.column_stack([ecm, lags])

        # MCO
        gamma = np.linalg.lstsq(X, dep, rcond=None)[0]
        self.gamma = gamma

        self.residuals = dep - X @ gamma

        return self

    def error_correction_term(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el término de corrección de error (ECT).

        ECT = β'Y_t representa la desviación del equilibrio de largo plazo.

        Args:
            data: Datos en niveles

        Returns:
            DataFrame con ECT
        """
        if self.beta is None:
            raise ValueError("Modelo no ajustado")

        ect = data.values @ self.beta

        cols = [f'ECT_{i+1}' for i in range(self.r)]
        return pd.DataFrame(ect, columns=cols, index=data.index)

    def long_run_equilibrium(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Relación de largo plazo: β'Y = 0

        Args:
            data: Datos en niveles

        Returns:
            DataFrame con la relación de largo plazo
        """
        ect = self.error_correction_term(data)
        return ect

    def summary(self) -> Dict:
        """Resumen del modelo VECM"""
        return {
            'rango_cointegracion': self.r,
            'alpha': self.alpha.tolist() if self.alpha is not None else None,
            'beta': self.beta.tolist() if self.beta is not None else None,
            'n_obs': len(self.residuals) if self.residuals is not None else 0
        }


class MonteCarloSimulation:
    """
    Simulación de Monte Carlo para análisis de incertidumbre.

    Perturba parámetros del modelo IS-LM según distribuciones
    especificadas y genera distribuciones de resultados de equilibrio.
    """

    def __init__(self, n_simulations: int = 1000, seed: Optional[int] = None):
        """
        Args:
            n_simulations: Número de simulaciones Monte Carlo
            seed: Semilla para reproducibilidad
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self.results = None

        if seed is not None:
            np.random.seed(seed)

    def simulate_islm(self, config: 'EconomyConfig', 
                      param_shocks: Optional[Dict] = None,
                      shock_type: str = 'normal') -> pd.DataFrame:
        """
        Simula incertidumbre en parámetros del modelo IS-LM.

        Args:
            config: Configuración base del modelo
            param_shocks: Dict con desviaciones estándar de parámetros.
                         Ej: {'c1': 0.05, 'b': 2.0, 'k': 0.05, 'h': 1.0}
            shock_type: Tipo de distribución ('normal', 'uniform')

        Returns:
            DataFrame con distribución de resultados de equilibrio
        """
        from ..core.equilibrium import EquilibriumSolver

        if param_shocks is None:
            param_shocks = {
                'c1': 0.05,
                'b': 2.0,
                'k': 0.05,
                'h': 1.0,
                'c0': 10.0,
                'I0': 15.0
            }

        results = []

        for i in range(self.n_simulations):
            # Perturbar parámetros
            import copy
            sim_config = copy.deepcopy(config)

            # Función para generar choque según tipo de distribución
            def get_shock(std):
                if shock_type == 'uniform':
                    return np.random.uniform(-std * np.sqrt(3), std * np.sqrt(3))
                return np.random.normal(0, std)

            if 'c1' in param_shocks:
                sim_config.consumo.c1 += get_shock(param_shocks['c1'])
                sim_config.consumo.c1 = np.clip(sim_config.consumo.c1, 0.01, 0.99)

            if 'c0' in param_shocks:
                sim_config.consumo.c0 += get_shock(param_shocks['c0'])

            if 'b' in param_shocks:
                sim_config.inversion.b += get_shock(param_shocks['b'])
                sim_config.inversion.b = max(0.1, sim_config.inversion.b)

            if 'I0' in param_shocks:
                sim_config.inversion.I0 += get_shock(param_shocks['I0'])

            if 'k' in param_shocks:
                sim_config.dinero.k += get_shock(param_shocks['k'])
                sim_config.dinero.k = max(0.01, sim_config.dinero.k)

            if 'h' in param_shocks:
                sim_config.dinero.h += get_shock(param_shocks['h'])
                sim_config.dinero.h = max(0.1, sim_config.dinero.h)

            # Calcular equilibrio
            try:
                solver = EquilibriumSolver(sim_config)
                result = solver.solve_islm_cerrado()
                results.append({
                    'Y': result.Y,
                    'r': result.r,
                    'C': result.C,
                    'I': result.I,
                    'P': result.P,
                    'L': result.L
                })
            except Exception:
                continue

        self.results = pd.DataFrame(results)
        return self.results

    def get_confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Calcula intervalos de confianza de los resultados simulados.

        Args:
            confidence: Nivel de confianza (ej. 0.95)

        Returns:
            DataFrame con estadísticos descriptivos
        """
        if self.results is None:
            raise ValueError("Ejecute simulate primero")

        alpha = 1 - confidence
        lower_p = alpha / 2
        upper_p = 1 - alpha / 2

        intervals = {}
        for col in self.results.columns:
            intervals[col] = {
                'mean': float(self.results[col].mean()),
                'median': float(self.results[col].median()),
                'std': float(self.results[col].std()),
                'min': float(self.results[col].min()),
                'max': float(self.results[col].max()),
                f'p{int(lower_p*100)}': float(self.results[col].quantile(lower_p)),
                f'p{int(upper_p*100)}': float(self.results[col].quantile(upper_p))
            }

        return pd.DataFrame(intervals).T

    def plot_distribution(self, variable: str = 'Y', bins: int = 50) -> 'go.Figure':
        """
        Genera histograma de la distribución simulada.

        Args:
            variable: Variable a graficar ('Y', 'r', 'C', 'I')
            bins: Número de bins del histograma

        Returns:
            Figura de Plotly
        """
        import plotly.graph_objects as go

        if self.results is None:
            raise ValueError("Ejecute simulate primero")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=self.results[variable],
            nbinsx=bins,
            name=f'Distribución de {variable}',
            marker_color='#2E86AB',
            opacity=0.75,
            histnorm='probability density'
        ))

        # Línea de media
        mean_val = self.results[variable].mean()
        fig.add_vline(
            x=mean_val, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"μ = {mean_val:.2f}"
        )

        # Línea de mediana
        median_val = self.results[variable].median()
        fig.add_vline(
            x=median_val, 
            line_dash="dot", 
            line_color="green",
            annotation_text=f"med = {median_val:.2f}"
        )

        fig.update_layout(
            title=f'Distribución de {variable} — Monte Carlo (n={self.n_simulations})',
            xaxis_title=variable,
            yaxis_title='Densidad',
            template='plotly_white',
            showlegend=False
        )

        return fig

    def sensitivity_analysis(self, config: 'EconomyConfig',
                            param: str, 
                            values: List[float]) -> pd.DataFrame:
        """
        Análisis de sensibilidad: varía un parámetro y observa el efecto.

        Args:
            config: Configuración base
            param: Parámetro a variar ('c1', 'b', 'k', 'h', 'G', 'M')
            values: Lista de valores a probar

        Returns:
            DataFrame con resultados para cada valor
        """
        from ..core.equilibrium import EquilibriumSolver

        results = []

        for val in values:
            import copy
            sim_config = copy.deepcopy(config)

            if param == 'c1':
                sim_config.consumo.c1 = val
            elif param == 'b':
                sim_config.inversion.b = val
            elif param == 'k':
                sim_config.dinero.k = val
            elif param == 'h':
                sim_config.dinero.h = val
            elif param == 'G':
                sim_config.G = val
            elif param == 'M':
                sim_config.dinero.M = val
            else:
                raise ValueError(f"Parámetro no soportado: {param}")

            try:
                solver = EquilibriumSolver(sim_config)
                result = solver.solve_islm_cerrado()
                results.append({
                    param: val,
                    'Y': result.Y,
                    'r': result.r,
                    'C': result.C,
                    'I': result.I
                })
            except Exception:
                continue

        return pd.DataFrame(results)
