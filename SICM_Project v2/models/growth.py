import numpy as np
from .base import MacroModel
from typing import Dict, List

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
    
    def steady_state(self) -> Dict:
        """Calcular estado estacionario"""
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
        else:
            return [
                "1. Shock tecnológico positivo (g↑)",
                "2. Aumento del trabajo efectivo",
                "3. Mayor nivel de producto en estado estacionario",
                "4. Posible aumento de crecimiento durante transición",
                "5. Efecto permanente en niveles de bienestar"
            ]

class Ramsey(Ramsey):
    """Modelo de Ramsey-Cass-Koopmans"""
    pass  # Implementación simplificada, similar a Solow pero con optimización
