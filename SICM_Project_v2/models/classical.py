# models/classical.py
import numpy as np
from typing import Dict, List
from .base import MacroModel

class ADASClassical(MacroModel):
    """Modelo AD-AS Clásico con LRAS vertical"""
    
    def get_default_params(self) -> Dict:
        return {
            'M': 200,      # Oferta monetaria
            'V': 5,        # Velocidad del dinero
            'Y_potential': 100,  # Producción potencial
            'P0': 1.0,     # Nivel de precios inicial
        }
    
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        """
        Sistema de ecuaciones del modelo AD-AS
        
        Args:
            vars: [Y, P] - Producción y nivel de precios
            params: Diccionario con parámetros
        
        Returns:
            [AD, AS] - Exceso de demanda y oferta
        """
        Y, P = vars
        
        # AD: MV = PY (Ecuación cuantitativa)
        AD = params['M'] * params['V'] - P * Y
        
        # AS Clásica: Y = Y_potential (LRAS vertical)
        AS = Y - params['Y_potential']
        
        return np.array([AD, AS])
    
    def solve(self, initial_guess=None) -> Dict:
        from scipy.optimize import fsolve
        
        if initial_guess is None:
            initial_guess = np.array([100, 1.0])  # [Y, P]
        
        try:
            solution = fsolve(self.equations, initial_guess, args=(self.params,))
            self.equilibrium = {
                "Y": float(solution[0]), 
                "P": float(solution[1])
            }
        except Exception as e:
            # Fallback: solución analítica
            Y = self.params['Y_potential']
            P = (self.params['M'] * self.params['V']) / Y
            self.equilibrium = {"Y": Y, "P": P}
        
        return self.equilibrium
    
    def apply_shock(self, shock_type: str, magnitude: float) -> Dict:
        if shock_type in ["↑ M", "↓ M"]:
            self.params['M'] *= (1 + magnitude)
        elif shock_type in ["↑ Productividad", "↓ Productividad"]:
            self.params['Y_potential'] *= (1 + magnitude)
        
        return self.solve()
    
    def get_transmission_mechanism(self, shock_type: str) -> List[str]:
        if "M" in shock_type:
            direction = "expansiva" if "↑" in shock_type else "contractiva"
            return [
                f"1. Política monetaria {direction}",
                "2. Aumento de la oferta monetaria (M)",
                "3. Exceso de oferta monetaria → aumento de precios (P)",
                "4. Neutralidad monetaria: P↑ pero Y no cambia",
                "5. Solo efectos nominales, no reales"
            ]
        else:
            direction = "positivo" if "↑" in shock_type else "negativo"
            return [
                f"1. Shock de productividad {direction}",
                "2. Desplazamiento de la LRAS",
                f"3. Nueva producción potencial: {self.params['Y_potential']:.1f}",
                "4. Ajuste completo de precios",
                "5. Efectos reales permanentes"
            ]
    
    def steady_state(self) -> Dict:
        return self.equilibrium if self.equilibrium else self.solve()


class LoanableFunds(MacroModel):
    """Modelo de mercado de fondos prestables"""
    
    def get_default_params(self) -> Dict:
        return {
            'S0': 50,      # Ahorro autónomo
            's': 0.3,      # Propensión marginal a ahorrar
            'I0': 100,     # Inversión autónoma
            'd': 0.8,      # Sensibilidad inversión a tasa
            'G': 80,       # Gasto gobierno
            'T': 80,       # Impuestos
            'Y': 100       # Producción fija en el largo plazo
        }
    
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        r, = vars  # tasa de interés real
        
        # Ahorro: S = S0 + s(Y - T) + (T - G)
        S_private = params['S0'] + params['s'] * (params['Y'] - params['T'])
        S_public = params['T'] - params['G']
        S = S_private + S_public
        
        # Inversión: I = I0 - d*r
        I = params['I0'] - params['d'] * r * 100
        
        return np.array([S - I])
    
    def solve(self, initial_guess=None) -> Dict:
        from scipy.optimize import fsolve
        
        if initial_guess is None:
            initial_guess = np.array([0.05])  # 5% tasa interés
        
        solution = fsolve(self.equations, initial_guess, args=(self.params,))
        self.equilibrium = {"r": float(solution[0])}
        return self.equilibrium
    
    def apply_shock(self, shock_type: str, magnitude: float) -> Dict:
        if shock_type == "↑ G":
            self.params['G'] *= (1 + magnitude)
        elif shock_type == "↓ G":
            self.params['G'] *= (1 - magnitude)
        elif shock_type == "↑ T":
            self.params['T'] *= (1 + magnitude)
        elif shock_type == "↓ T":
            self.params['T'] *= (1 - magnitude)
        
        return self.solve()
    
    def get_transmission_mechanism(self, shock_type: str) -> List[str]:
        if "G" in shock_type:
            direction = "expansivo" if "↑" in shock_type else "contractivo"
            return [
                f"1. Choque fiscal {direction}",
                "2. Reducción del ahorro público",
                "3. Desplazamiento de la curva de ahorro",
                "4. Aumento de la tasa de interés real",
                "5. Crowding out de inversión privada"
            ]
        else:
            direction = "expansiva" if "↑" in shock_type else "contractiva"
            return [
                f"1. Política fiscal {direction} vía impuestos",
                "2. Cambio en ahorro privado",
                "3. Ajuste en tasa de interés",
                "4. Cambio en inversión"
            ]
    
    def steady_state(self) -> Dict:
        return self.equilibrium if self.equilibrium else self.solve()
