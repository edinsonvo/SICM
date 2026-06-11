from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, List, Optional

class MacroModel(ABC):
    """Clase base para todos los modelos macroeconómicos"""
    
    def __init__(self, settings):
        self.settings = settings
        self.params = self.get_default_params()
        self.equilibrium: Optional[Dict] = None
        self.history = []
        
    @abstractmethod
    def get_default_params(self) -> Dict:
        pass
    
    @abstractmethod
    def equations(self, vars: np.ndarray, params: Dict) -> np.ndarray:
        pass
    
    def solve(self, initial_guess: Optional[np.ndarray] = None) -> Dict:
        from scipy.optimize import fsolve
        
        if initial_guess is None:
            initial_guess = np.array([100, 0.05, 0.02])  # Y, i, π
        
        solution = fsolve(self.equations, initial_guess, args=(self.params,))
        
        self.equilibrium = {
            "Y": float(solution[0]),
            "i": float(solution[1]) if len(solution) > 1 else 0.05,
            "π": float(solution[2]) if len(solution) > 2 else 0.02
        }
        return self.equilibrium
    
    def apply_shock(self, shock_type: str, magnitude: float) -> Dict:
        pass
    
    def get_transmission_mechanism(self, shock_type: str) -> List[str]:
        pass
    
    def steady_state(self) -> Dict:
        return self.equilibrium if self.equilibrium else self.solve()
