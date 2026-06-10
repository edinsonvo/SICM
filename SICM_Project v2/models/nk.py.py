import numpy as np
from .base import MacroModel
from typing import Dict, List

class NewKeynesian(MacroModel):
    """Modelo Nuevo Keynesiano de 3 ecuaciones"""
    
    def get_default_params(self) -> Dict:
        return {
            # Parámetros IS
            'σ': 1.0,      # Elasticidad de sustitución intertemporal
            'β': 0.99,     # Factor de descuento
            'r_natural': 0.02,  # Tasa natural
            
            # Parámetros Phillips
            'κ': 0.1,      # Pendiente de Phillips
            'θ': 0.75,     # Rigidez de precios
            
            # Parámetros Taylor
            'φ_π': 1.5,    # Respuesta a inflación
            'φ_y': 0.5,    # Respuesta a output gap
            
            # Estado inicial
            'π_target': 0.02,  # Meta de inflación 2%
            'Y_potential': 100,
            
            # Shocks
            'ε_demand': 0.0,
            'ε_supply': 0.0
        }
    
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        output_gap, π, i = vars
        
        # Curva IS Nuevo Keynesiana
        # output_gap = E[output_gap_next] - σ*(i - E[π_next] - r_natural)
        # Simplificada con expectativas adaptativas
        IS = output_gap + params['σ'] * (i - π - params['r_natural'])
        
        # Curva de Phillips Nuevo Keynesiana
        # π = β*E[π_next] + κ*output_gap + ε_supply
        Phillips = π - (params['β'] * π + params['κ'] * output_gap + params['ε_supply'])
        
        # Regla de Taylor
        # i = r_natural + π_target + φ_π*(π - π_target) + φ_y*output_gap
        Taylor = i - (params['r_natural'] + params['π_target'] + 
                      params['φ_π'] * (π - params['π_target']) + 
                      params['φ_y'] * output_gap + params['ε_demand'])
        
        return np.array([IS, Phillips, Taylor])
    
    def apply_shock(self, shock_type: str, magnitude: float) -> Dict:
        if shock_type == "Demanda":
            self.params['ε_demand'] = magnitude
        elif shock_type == "Oferta (Costos)":
            self.params['ε_supply'] = magnitude
        elif shock_type == "↑ M":
            # Política monetaria expansiva se refleja en regla de Taylor
            self.params['π_target'] *= (1 + magnitude)
        
        return self.solve()
    
    def get_transmission_mechanism(self, shock_type: str) -> List[str]:
        if shock_type == "Demanda":
            return [
                "1. Shock positivo de demanda",
                "2. Aumento del output gap",
                "3. Presión al alza sobre inflación",
                "4. Banco central aumenta tasa de interés (Regla de Taylor)",
                "5. Parcialmente estabiliza output e inflación",
                "6. Efecto riqueza y consumo intertemporal"
            ]
        elif shock_type == "Oferta (Costos)":
            return [
                "1. Shock de oferta negativo (aumento costos)",
                "2. Inflación aumenta (cost-push)",
                "3. Output gap se reduce",
                "4. Dilema del banco central: ¿estabilizar inflación o output?",
                "5. Compensación (trade-off) en el corto plazo"
            ]
        else:
            return [
                "1. Cambio en la meta de inflación",
                "2. Banco central modifica tasa de interés",
                "3. Impacto en demanda agregada vía IS",
                "4. Efecto gradual sobre inflación (Phillips)",
                "5. Nuevo equilibrio con expectativas ancladas"
            ]

class TaylorRule(NewKeynesian):
    """Versión simplificada solo con Regla de Taylor"""
    
    def get_default_params(self) -> Dict:
        params = super().get_default_params()
        params.update({
            'i_current': 0.05,
            'π_current': 0.02,
            'Y_current': 100
        })
        return params
    
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        i, = vars
        
        # Tasa recomendada por Regla de Taylor
        output_gap = (params['Y_current'] - params['Y_potential']) / params['Y_potential']
        inflation_gap = params['π_current'] - params['π_target']
        
        i_recommended = (params['r_natural'] + params['π_target'] + 
                         params['φ_π'] * inflation_gap + 
                         params['φ_y'] * output_gap)
        
        return np.array([i - i_recommended])