import numpy as np
from .base import MacroModel
from typing import Dict, List

class ADASClassical(MacroModel):
    """Modelo AD-AS Clásico con LRAS vertical"""
    
    def get_default_params(self) -> Dict:
        return {
            'M': 200,      # Oferta monetaria
            'V': 5,        # Velocidad del dinero
            'Y_potential': 100,  # Producción potencial
            'α': 0.5,      # Sensibilidad de precios
            'P0': 1.0,     # Nivel de precios inicial
            'expectations': 1.0  # Expectativas de precios
        }
    
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        Y, P = vars
        
        # AD: MV = PY (Ecuación cuantitativa)
        AD = params['M'] * params['V'] - P * Y
        
        # AS Clásica: Y = Y_potencial (LRAS vertical)
        AS = Y - params['Y_potential']
        
        return np.array([AD, AS])
    
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
                "3. Nuevo nivel de producción potencial",
                "4. Ajuste completo de precios",
                "5. Efectos reales permanentes"
            ]

class LoanableFunds(MacroModel):
    """Modelo de mercado de fondos prestables"""
    
    def get_default_params(self) -> Dict:
        return {
            'S0': 50,      # Ahorro autónomo
            's': 0.3,      # Propensión marginal a ahorrar
            'I0': 100,     # Inversión autónoma
            'd': 0.8,      # Sensibilidad inversión a tasa
            'G': 80,       # Gasto gobierno
            'T': 80        # Impuestos
        }
    
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        r, = vars  # tasa de interés real
        
        # Ahorro: S = S0 + s(Y - T) + (T - G)
        Y = 100  # Producción fija en el largo plazo
        S_private = params['S0'] + params['s'] * (Y - params['T'])
        S_public = params['T'] - params['G']
        S = S_private + S_public
        
        # Inversión: I = I0 - d*r
        I = params['I0'] - params['d'] * r * 100
        
        return np.array([S - I])
    
    def apply_shock(self, shock_type: str, magnitude: float) -> Dict:
        if shock_type == "↑ G":
            self.params['G'] *= (1 + magnitude)
        elif shock_type == "↓ G":
            self.params['G'] *= (1 - magnitude)
        
        return self.solve()
    
    def get_transmission_mechanism(self, shock_type: str) -> List[str]:
        return [
            "1. Aumento del gasto público (G)",
            "2. Reducción del ahorro público",
            "3. Desplazamiento de la curva de ahorro",
            "4. Aumento de la tasa de interés real",
            "5. Crowding out de inversión privada"
        ]
