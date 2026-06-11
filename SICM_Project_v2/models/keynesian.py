# models/keynesian.py
import numpy as np
from typing import Dict, List
from .base import MacroModel

class ISLM(MacroModel):
    """Modelo IS-LM para economía cerrada"""
    
    def get_default_params(self):
        return {
            'C0': 50,      # Consumo autónomo
            'c': 0.7,      # Propensión marginal a consumir
            'I0': 100,     # Inversión autónoma
            'b': 0.5,      # Sensibilidad inversión a tasa interés
            'G': 100,      # Gasto gobierno
            'T': 80,       # Impuestos
            'M': 200,      # Oferta monetaria
            'P': 1,        # Nivel de precios
            'k': 0.2,      # Sensibilidad demanda dinero a Y
            'h': 100       # Sensibilidad demanda dinero a i
        }
    
    def equations(self, vars, params):
        Y, i = vars
        
        # Curva IS: Y = C + I + G
        C = params['C0'] + params['c'] * (Y - params['T'])
        I = params['I0'] - params['b'] * i * 100
        IS = Y - (C + I + params['G'])
        
        # Curva LM: M/P = L(Y,i)
        Ld = params['k'] * Y - params['h'] * i
        LM = params['M']/params['P'] - Ld
        
        return np.array([IS, LM])
    
    def solve(self, initial_guess=None):
        from scipy.optimize import fsolve
        
        if initial_guess is None:
            initial_guess = np.array([100, 0.05])
        
        solution = fsolve(self.equations, initial_guess, args=(self.params,))
        self.equilibrium = {"Y": float(solution[0]), "i": float(solution[1])}
        return self.equilibrium
    
    def apply_shock(self, shock_type: str, magnitude: float):
        if shock_type == "↑ G":
            self.params['G'] *= (1 + magnitude)
        elif shock_type == "↓ G":
            self.params['G'] *= (1 - magnitude)
        elif shock_type == "↑ M":
            self.params['M'] *= (1 + magnitude)
        elif shock_type == "↓ M":
            self.params['M'] *= (1 - magnitude)
        elif shock_type == "↑ T":
            self.params['T'] *= (1 + magnitude)
        elif shock_type == "↓ T":
            self.params['T'] *= (1 - magnitude)
        
        return self.solve()
    
    def get_transmission_mechanism(self, shock_type: str):
        if shock_type in ["↑ G", "↓ G"]:
            direction = "expansivo" if "↑" in shock_type else "contractivo"
            return [
                f"1. Choque fiscal {direction}",
                "2. Desplazamiento de la curva IS",
                "3. Aumento del PIB (Y)",
                "4. Aumento de la demanda de dinero",
                "5. Aumento de la tasa de interés (i)",
                "6. Efecto crowding out sobre inversión"
            ]
        else:
            direction = "expansiva" if "↑" in shock_type else "contractiva"
            return [
                f"1. Política monetaria {direction}",
                "2. Desplazamiento de la curva LM",
                "3. Caída de la tasa de interés (i)",
                "4. Aumento de la inversión (I)",
                "5. Aumento del PIB (Y)"
            ]
    
    def steady_state(self):
        return self.equilibrium if self.equilibrium else self.solve()


class MundellFleming(ISLM):
    """Modelo Mundell-Fleming para economía abierta"""
    
    def get_default_params(self):
        params = super().get_default_params()
        params.update({
            'NX0': 20,     # Exportaciones netas autónomas
            'm': 0.1,      # Propensión marginal a importar
            'e': 10,       # Tipo de cambio nominal
            'r': 0.02      # Tasa de interés internacional
        })
        return params
