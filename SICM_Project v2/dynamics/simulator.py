import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import signal

class DynamicSimulator:
    """Simulador dinámico y funciones impulso-respuesta"""
    
    def __init__(self, model):
        self.model = model
        self.history = []
        
    def time_series_simulation(self, shock_type: str, magnitude: float, 
                               periods: int = 20, persistence: float = 0.8) -> pd.DataFrame:
        """Simular trayectoria temporal después de un choque"""
        
        # Guardar estado original
        original_params = self.model.params.copy()
        
        # Inicializar matrices para almacenar resultados
        Y_path = [self.model.equilibrium['Y']]
        i_path = [self.model.equilibrium['i']]
        π_path = [self.model.equilibrium.get('π', 0.02)]
        
        # Aplicar choque con persistencia
        shock_t = magnitude
        for t in range(1, periods):
            # Choque decae con persistencia
            shock_t = shock_t * persistence
            
            # Aplicar choque del periodo
            self.model.apply_shock(shock_type, shock_t)
            eq = self.model.solve(initial_guess=np.array([Y_path[-1], i_path[-1], π_path[-1]]))
            
            Y_path.append(eq['Y'])
            i_path.append(eq['i'])
            π_path.append(eq.get('π', π_path[-1]))
            
            # Pequeño ruido para realismo (opcional)
            if t > 5:
                Y_path[-1] += np.random.normal(0, 0.5)
        
        # Restaurar parámetros originales
        self.model.params = original_params
        self.model.solve()
        
        return pd.DataFrame({
            'periodo': list(range(periods)),
            'PIB': Y_path,
            'tasa_interes': i_path,
            'inflacion': π_path
        })
    
    def impulse_response_function(self, shock_type: str, magnitude: float = 0.1, 
                                  periods: int = 20, variables: List[str] = None) -> Dict:
        """Calcular funciones impulso-respuesta (IRF)"""
        
        if variables is None:
            variables = ['Y', 'i', 'π']
        
        # Simular con choque
        df_shock = self.time_series_simulation(shock_type, magnitude, periods)
        
        # Simular sin choque (baseline)
        baseline = self.time_series_simulation(shock_type, 0, periods)
        
        # Calcular IRF como diferencia
        irf = {}
        for var in variables:
            var_map = {'Y': 'PIB', 'i': 'tasa_interes', 'π': 'inflacion'}
            col = var_map.get(var, var)
            irf[var] = df_shock[col].values - baseline[col].values
        
        return irf
    
    def var_irf_from_data(self, data: pd.DataFrame, variables: List[str], 
                          shocks: List[str], periods: int = 20) -> Dict:
        """Estimar IRF empíricas usando VAR sobre datos reales"""
        from statsmodels.tsa.api import VAR
        
        # Preparar datos
        data_clean = data[variables].dropna()
        
        # Estimar VAR
        model_var = VAR(data_clean)
        results = model_var.fit(maxlags=5, ic='aic')
        
        # Calcular IRF
        irf = results.irf(periods)
        
        return {
            'irf_plot': irf.plot(),
            'irf_table': irf.irfs,
            'cumulative': irf.cum_effects
        }
    
    def monte_carlo_simulation(self, shock_type: str, magnitude: float, 
                              n_simulations: int = 100, periods: int = 20) -> pd.DataFrame:
        """Simulación Monte Carlo para incertidumbre"""
        
        all_simulations = []
        
        for _ in range(n_simulations):
            # Añadir ruido aleatorio a la magnitud
            noisy_magnitude = magnitude * (1 + np.random.normal(0, 0.2))
            df = self.time_series_simulation(shock_type, noisy_magnitude, periods)
            all_simulations.append(df['PIB'].values)
        
        # Calcular estadísticos
        sim_array = np.array(all_simulations)
        
        return pd.DataFrame({
            'periodo': list(range(periods)),
            'media': sim_array.mean(axis=0),
            'percentil_5': np.percentile(sim_array, 5, axis=0),
            'percentil_95': np.percentile(sim_array, 95, axis=0),
            'std': sim_array.std(axis=0)
        })
    
    def convergence_dynamics(self, target_variable: str = 'Y', tolerance: float = 0.01) -> Dict:
        """Analizar dinámica de convergencia al equilibrio"""
        
        initial = self.model.equilibrium[target_variable]
        steady = self.model.steady_state().get(target_variable, initial)
        
        # Simular convergencia
        path = [initial]
        current = initial
        
        while abs((current - steady) / steady) > tolerance:
            # Ajuste gradual
            current = current + 0.1 * (steady - current)
            path.append(current)
            
            if len(path) > 100:  # Evitar loops infinitos
                break
        
        half_life = None
        for i, val in enumerate(path):
            if abs((val - steady) / steady) <= 0.5 * abs((initial - steady) / steady):
                half_life = i
                break
        
        return {
            'initial': initial,
            'steady_state': steady,
            'convergence_path': path,
            'half_life_periods': half_life,
            'speed_of_convergence': abs(np.log(0.5) / half_life) if half_life else None
        }
