import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from arch import arch_model
import pmdarima as pm

class EconometricLab:
    """Laboratorio econométrico integrado"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        
    def load_real_data(self, source: str = 'fred', indicators: List[str] = None) -> pd.DataFrame:
        """Cargar datos reales de diversas fuentes"""
        
        if source == 'fred':
            # Usar yfinance como alternativa a FRED (gratuito)
            if indicators is None:
                indicators = ['GDP', 'CPIAUCSL', 'FEDFUNDS']
            
            # Mapeo de indicadores a tickers de Yahoo Finance
            tickers = {
                'GDP': 'GDP',  # Necesita API de FRED, alternativa usar datos simulados
                'CPIAUCSL': 'CPI',
                'FEDFUNDS': '^IRX',
                'UNRATE': 'UNRATE'
            }
            
            data = {}
            for ind in indicators:
                try:
                    # Para demostración, generamos datos sintéticos realistas
                    np.random.seed(42)
                    dates = pd.date_range(start='2000-01-01', periods=240, freq='M')
                    
                    if ind == 'GDP':
                        data[ind] = 10000 + np.cumsum(np.random.normal(50, 20, 240))
                    elif ind == 'CPIAUCSL':
                        data[ind] = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.003, 240)))
                    elif ind == 'FEDFUNDS':
                        data[ind] = np.maximum(0, 0.05 + np.random.normal(0, 0.01, 240).cumsum())
                    else:
                        data[ind] = np.random.normal(0, 1, 240)
                except:
                    data[ind] = np.random.normal(100, 10, 240)
            
            self.data = pd.DataFrame(data, index=dates)
            
        elif source == 'simulated':
            # Datos simulados para pruebas
            dates = pd.date_range(start='2000-01-01', periods=200, freq='Q')
            self.data = pd.DataFrame({
                'gdp': 100 + np.cumsum(np.random.normal(0.5, 1, 200)),
                'inflation': 0.02 + np.random.normal(0, 0.005, 200),
                'interest': 0.05 + np.random.normal(0, 0.01, 200),
                'unemployment': 0.06 + np.random.normal(0, 0.005, 200)
            }, index=dates)
        
        return self.data
    
    def estimate_arima(self, variable: str, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """Estimar modelo ARIMA"""
        
        if self.data is None or variable not in self.data.columns:
            raise ValueError(f"Datos no disponibles para {variable}")
        
        series = self.data[variable].dropna()
        
        # Auto-selección de órdenes
        auto_model = pm.auto_arima(series, seasonal=False, stepwise=True, trace=False)
        
        # Estimar modelo
        model = ARIMA(series, order=auto_model.order)
        results = model.fit()
        
        # Pronósticos
        forecast = results.forecast(steps=12)
        forecast_ci = results.get_forecast(steps=12).conf_int()
        
        self.models['arima'] = results
        
        return {
            'model': results,
            'order': auto_model.order,
            'params': results.params.to_dict(),
            'aic': results.aic,
            'bic': results.bic,
            'forecast': forecast,
            'forecast_ci': forecast_ci,
            'residuals': results.resid
        }
    
    def estimate_sarima(self, variable: str, seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 4)) -> Dict:
        """Estimar modelo SARIMA con estacionalidad"""
        
        series = self.data[variable].dropna()
        
        model = SARIMAX(series, 
                        order=(1, 1, 1),
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        results = model.fit(disp=False)
        
        forecast = results.forecast(steps=12)
        
        self.models['sarima'] = results
        
        return {
            'model': results,
            'params': results.params.to_dict(),
            'aic': results.aic,
            'bic': results.bic,
            'forecast': forecast,
            'residuals': results.resid
        }
    
    def estimate_var(self, variables: List[str], maxlags: int = 5) -> Dict:
        """Estimar modelo VAR multivariado"""
        
        data_vars = self.data[variables].dropna()
        
        model = VAR(data_vars)
        results = model.fit(maxlags=maxlags, ic='aic')
        
        # IRF
        irf = results.irf(20)
        
        self.models['var'] = results
        
        return {
            'model': results,
            'coeffs': results.coefs,
            'irf': irf.irfs,
            'irf_plot': irf.plot(),
            'forecast': results.forecast(data_vars.values[-results.k_ar:], steps=12),
            'granger_causality': results.test_causality(variables[0], variables[1:])
        }
    
    def estimate_garch(self, variable: str, p: int = 1, q: int = 1) -> Dict:
        """Estimar modelo GARCH para volatilidad"""
        
        returns = self.data[variable].pct_change().dropna() * 100
        
        model = arch_model(returns, vol='Garch', p=p, q=q)
        results = model.fit(update_freq=5)
        
        # Pronósticos de volatilidad
        forecasts = results.forecast(horizon=12)
        
        self.models['garch'] = results
        
        return {
            'model': results,
            'params': results.params.to_dict(),
            'conditional_volatility': results.conditional_volatility,
            'volatility_forecast': forecasts.variance,
            'aic': results.aic,
            'bic': results.bic
        }
    
    def fiscal_multiplier_estimation(self, output_var: str = 'gdp', 
                                     fiscal_var: str = 'gov_spending') -> float:
        """Estimar multiplicador fiscal usando VAR o métodos de series de tiempo"""
        
        # Preparar datos
        data_fiscal = self.data[[output_var, fiscal_var]].dropna()
        
        # Estimar VAR
        model = VAR(data_fiscal)
        results = model.fit(maxlags=4)
        
        # IRF de gasto gobierno a PIB
        irf = results.irf(10)
        
        # Multiplicador acumulado (suma de respuestas)
        cumulative_multiplier = irf.irfs[1:6, 0, 1].sum()  # Cambio acumulado en PIB / shock
        
        return {
            'multiplier': cumulative_multiplier,
            'irf_response': irf.irfs[:, 0, 1],
            'confidence_intervals': irf.irfs_cum_effects
        }
    
    def compare_theoretical_empirical(self, model_theoretical, shock_type: str, 
                                     empirical_var: str) -> Dict:
        """Comparar IRF teórica vs empírica"""
        
        # IRF teórica del modelo
        from dynamics.simulator import DynamicSimulator
        simulator = DynamicSimulator(model_theoretical)
        theoretical_irf = simulator.impulse_response_function(shock_type)
        
        # IRF empírica de VAR
        var_results = self.estimate_var([empirical_var, 'interest'])
        empirical_irf = var_results['irf'][:20, 0, 1] if len(var_results['irf'].shape) > 2 else np.zeros(20)
        
        # Correlación y similitud
        correlation = np.corrcoef(theoretical_irf.get('Y', np.zeros(20))[:20], 
                                 empirical_irf[:20])[0, 1]
        
        return {
            'theoretical_irf': theoretical_irf,
            'empirical_irf': empirical_irf,
            'correlation': correlation,
            'rmse': np.sqrt(np.mean((theoretical_irf.get('Y', np.zeros(20))[:20] - empirical_irf[:20])**2))
        }