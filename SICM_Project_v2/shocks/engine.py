# shocks/engine.py
from typing import Dict, List, Tuple, Any

class ShockEngine:
    """Motor de aplicación de choques económicos"""
    
    AVAILABLE_SHOCKS = {
        "Fiscales": ["↑ G", "↓ G", "↑ T", "↓ T"],
        "Monetarios": ["↑ M", "↓ M"],
        "Externos": ["↑ Exportaciones", "↓ Importaciones", "Mejora términos intercambio"],
        "Oferta": ["↑ Productividad", "↓ Productividad", "↑ Salarios"]
    }
    
    def __init__(self, model: Any):
        """
        Inicializa el motor de choques
        
        Args:
            model: Instancia de un modelo macroeconómico
        """
        self.model = model
        self.shock_history = []
    
    def apply(self, shock_type: str, magnitude: float = 0.1) -> Tuple[Dict, List[str]]:
        """
        Aplica un choque al modelo
        
        Args:
            shock_type: Tipo de choque (ej. "↑ G", "↑ M")
            magnitude: Magnitud del choque (ej. 0.1 = 10%)
        
        Returns:
            Tuple con (nuevo_equilibrio, mecanismo_de_transmisión)
        """
        # Validar que el choque existe
        if shock_type not in self._flatten_shocks():
            raise ValueError(f"Choque {shock_type} no disponible. Opciones: {self._flatten_shocks()}")
        
        # Verificar que el modelo tiene los métodos necesarios
        if not hasattr(self.model, 'apply_shock'):
            raise AttributeError("El modelo no tiene método 'apply_shock'")
        
        if not hasattr(self.model, 'get_transmission_mechanism'):
            raise AttributeError("El modelo no tiene método 'get_transmission_mechanism'")
        
        # Aplicar choque
        new_eq = self.model.apply_shock(shock_type, magnitude)
        mechanism = self.model.get_transmission_mechanism(shock_type)
        
        # Registrar en historial
        self.shock_history.append({
            "shock": shock_type,
            "magnitude": magnitude,
            "equilibrium": new_eq
        })
        
        return new_eq, mechanism
    
    def _flatten_shocks(self) -> List[str]:
        """Retorna lista plana de todos los choques disponibles"""
        return [s for shocks in self.AVAILABLE_SHOCKS.values() for s in shocks]
    
    def compare_scenarios(self, shock_type: str, magnitudes: List[float]) -> List[Dict]:
        """
        Compara el mismo choque con diferentes magnitudes
        
        Args:
            shock_type: Tipo de choque
            magnitudes: Lista de magnitudes a probar
        
        Returns:
            Lista de resultados para cada magnitud
        """
        results = []
        
        # Guardar estado original
        if hasattr(self.model, 'equilibrium'):
            original_eq = self.model.equilibrium.copy() if self.model.equilibrium else None
        else:
            original_eq = None
        
        original_params = self.model.params.copy() if hasattr(self.model, 'params') else None
        
        for mag in magnitudes:
            try:
                eq, _ = self.apply(shock_type, mag)
                results.append({
                    "magnitud": mag,
                    "Y": eq.get('Y', 0),
                    "i": eq.get('i', 0),
                    "π": eq.get('π', 0)
                })
            except Exception as e:
                results.append({
                    "magnitud": mag,
                    "error": str(e)
                })
            
            # Restaurar modelo original
            if original_params:
                self.model.params = original_params.copy()
            if original_eq:
                self.model.equilibrium = original_eq
        
        return results
    
    def get_history(self) -> List[Dict]:
        """Retorna el historial de choques aplicados"""
        return self.shock_history
    
    def clear_history(self):
        """Limpia el historial de choques"""
        self.shock_history = []
    
    def get_available_shocks(self) -> Dict[str, List[str]]:
        """Retorna los choques disponibles por categoría"""
        return self.AVAILABLE_SHOCKS
