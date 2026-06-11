# models/growth.py
import numpy as np
from typing import Dict, List
from .base import MacroModel

class Solow(MacroModel):
    """Modelo de crecimiento de Solow"""
    
    def get_default_params(self) -> Dict:
        return {
            's': 0.3,      # Tasa de ahorro
            'δ': 0.05,     # Depreciación capital
            'n': 0.01,     # Tasa crecimiento población
            'g': 0.02,     # Progreso tecnológico
            'α': 0.3,      # Elasticidad capital
            'k0': 1.0,     # Capital inicial per cápita efectivo
            'A0': 1.0,     # Tecnología inicial
            'L0': 100      # Población inicial
        }
    
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        k, = vars  # Capital por unidad de trabajo efectivo
        
        # Ecuación fundamental de Solow
        # Δk = s*f(k) - (δ + n + g)*k
        f_k = k ** params['α']
        k_dot = params['s'] * f_k - (params['δ'] + params['n'] + params['g']) * k
        
        return np.array([k_dot])
    
    def solve(self, initial_guess=None):
        """Resolver estado estacionario"""
        if initial_guess is None:
            initial_guess = np.array([1.0])
        
        from scipy.optimize import fsolve
        solution = fsolve(self.equations, initial_guess, args=(self.params,))
        
        self.equilibrium = {"k": float(solution[0])}
        return self.equilibrium
    
    def steady_state(self) -> Dict:
        """Calcular estado estacionario analíticamente"""
        s = self.params['s']
        δ = self.params['δ']
        n = self.params['n']
        g = self.params['g']
        α = self.params['α']
        
        k_ss = (s / (δ + n + g)) ** (1 / (1 - α))
        y_ss = k_ss ** α
        
        return {
            'k_ss': k_ss,
            'y_ss': y_ss,
            'c_ss': (1 - s) * y_ss
        }
    
    def time_path(self, periods: int = 50) -> Dict:
        """Simular trayectoria dinámica hacia estado estacionario"""
        k = self.params['k0']
        path = []
        
        for t in range(periods):
            path.append(k)
            f_k = k ** self.params['α']
            k_dot = (self.params['s'] * f_k - 
                    (self.params['δ'] + self.params['n'] + self.params['g']) * k)
            k += k_dot * 0.5  # Factor de ajuste para estabilidad
        
        return {'capital': path, 'periods': list(range(periods))}
    
    def apply_shock(self, shock_type: str, magnitude: float) -> Dict:
        if shock_type == "↑ s (Ahorro)":
            self.params['s'] = min(0.9, self.params['s'] * (1 + magnitude))
        elif shock_type == "↑ n (Población)":
            self.params['n'] *= (1 + magnitude)
        elif shock_type == "↑ g (Tecnología)":
            self.params['g'] *= (1 + magnitude)
        
        return self.steady_state()
    
    def get_transmission_mechanism(self, shock_type: str) -> List[str]:
        if "Ahorro" in shock_type:
            return [
                "1. Aumento de la tasa de ahorro (s)",
                "2. Mayor inversión por trabajador",
                "3. Acumulación de capital durante transición",
                "4. Aumento del producto per cápita en nuevo estado estacionario",
                "5. Efecto nivel (no crecimiento de largo plazo)"
            ]
        elif "Población" in shock_type:
            return [
                "1. Aumento de la tasa de crecimiento poblacional (n)",
                "2. Menor capital por trabajador efectivo",
                "3. Reducción del producto per cápita en estado estacionario",
                "4. Efecto negativo en niveles de bienestar"
            ]
        else:
            return [
                "1. Shock tecnológico positivo (g↑)",
                "2. Aumento del trabajo efectivo",
                "3. Mayor nivel de producto en estado estacionario",
                "4. Posible aumento de crecimiento durante transición",
                "5. Efecto permanente en niveles de bienestar"
            ]


class Ramsey:
    """Modelo de Ramsey-Cass-Koopmans (versión simplificada)"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.params = {
            'β': 0.99,     # Factor de descuento
            'σ': 1.0,      # Elasticidad de sustitución intertemporal
            'δ': 0.05,     # Depreciación
            'α': 0.3,      # Elasticidad capital
            'n': 0.01,     # Crecimiento población
            'g': 0.02,     # Progreso tecnológico
            'k0': 1.0      # Capital inicial
        }
    
    def steady_state(self) -> Dict:
        """Regla de oro modificada de Ramsey"""
        α = self.params['α']
        δ = self.params['δ']
        n = self.params['n']
        g = self.params['g']
        β = self.params['β']
        
        # Tasa de interés de estado estacionario
        r_ss = (1/β) - 1
        
        # Capital de estado estacionario
        k_ss = (α / (r_ss + δ)) ** (1 / (1 - α))
        
        return {
            'k_ss': k_ss,
            'r_ss': r_ss,
            'c_ss': k_ss ** α - (n + g + δ) * k_ss
        }
    
    def apply_shock(self, shock_type: str, magnitude: float) -> Dict:
        if shock_type == "↑ β (Paciencia)":
            self.params['β'] = min(0.999, self.params['β'] + magnitude * 0.01)
        return self.steady_state()

class Ramsey(Ramsey):
    """Modelo de Ramsey-Cass-Koopmans"""
    pass  # Implementación simplificada, similar a Solow pero con optimización
