"""
SICM v5 Research Lab — Importación y Calibración de Datos
==========================================================
Carga datos reales y calibra parámetros del modelo.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from scipy.optimize import minimize
from ..core.parameters import EconomyConfig, ParametrosConsumo, ParametrosInversion, ParametrosDinero


class DataLoader:
    """
    Carga datos macroeconómicos desde CSV o Excel.
    """
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
    
    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """Carga datos desde CSV"""
        return pd.read_csv(filepath, **kwargs)
    
    @staticmethod
    def load_excel(filepath: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """Carga datos desde Excel"""
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
        """
        Valida que el DataFrame tenga las columnas requeridas.
        
        Columnas esperadas: Y, C, I, G, T, M, r, P
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        return True


class ModelCalibrator:
    """
    Calibra parámetros del modelo a partir de datos reales.
    
    Estima: c, b, k, h desde series temporales.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.params = {}
    
    def calibrate_consumo(self) -> Dict:
        """
        Estima función de consumo: C = c0 + c1*(Y-T)
        
        Returns:
            Dict con c0, c1 estimados
        """
        # Variables
        Y = self.data['Y'].values
        C = self.data['C'].values
        T = self.data.get('T', np.zeros(len(Y))).values
        
        Yd = Y - T  # Ingreso disponible
        
        # Regresión OLS: C = c0 + c1*Yd
        X = np.column_stack([np.ones(len(Yd)), Yd])
        beta = np.linalg.lstsq(X, C, rcond=None)[0]
        
        c0, c1 = beta[0], beta[1]
        
        # Calcular R²
        C_pred = c0 + c1 * Yd
        ss_res = np.sum((C - C_pred) ** 2)
        ss_tot = np.sum((C - np.mean(C)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        self.params['consumo'] = {'c0': c0, 'c1': c1, 'R2': r2}
        
        return self.params['consumo']
    
    def calibrate_inversion(self) -> Dict:
        """
        Estima función de inversión: I = I0 - b*r
        
        Returns:
            Dict con I0, b estimados
        """
        I = self.data['I'].values
        r = self.data['r'].values
        
        # Regresión: I = I0 - b*r  =>  I = I0 + (-b)*r
        X = np.column_stack([np.ones(len(r)), r])
        beta = np.linalg.lstsq(X, I, rcond=None)[0]
        
        I0, b = beta[0], -beta[1]
        
        # Calcular R²
        I_pred = I0 - b * r
        ss_res = np.sum((I - I_pred) ** 2)
        ss_tot = np.sum((I - np.mean(I)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        self.params['inversion'] = {'I0': I0, 'b': b, 'R2': r2}
        
        return self.params['inversion']
    
    def calibrate_dinero(self) -> Dict:
        """
        Estima función de demanda de dinero: M/P = k*Y - h*r
        
        Returns:
            Dict con k, h estimados
        """
        M = self.data['M'].values
        P = self.data.get('P', np.ones(len(M))).values
        Y = self.data['Y'].values
        r = self.data['r'].values
        
        Md_real = M / P
        
        # Regresión: Md/P = k*Y - h*r
        X = np.column_stack([Y, r])
        beta = np.linalg.lstsq(X, Md_real, rcond=None)[0]
        
        k, h = beta[0], -beta[1]
        
        # Calcular R²
        Md_pred = k * Y - h * r
        ss_res = np.sum((Md_real - Md_pred) ** 2)
        ss_tot = np.sum((Md_real - np.mean(Md_real)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        self.params['dinero'] = {'k': k, 'h': h, 'R2': r2}
        
        return self.params['dinero']
    
    def calibrate_all(self) -> Dict:
        """
        Calibra todos los parámetros del modelo.
        
        Returns:
            Dict completo con todos los parámetros estimados
        """
        self.calibrate_consumo()
        self.calibrate_inversion()
        self.calibrate_dinero()
        
        return self.params
    
    def to_economy_config(self) -> EconomyConfig:
        """
        Crea una EconomyConfig con los parámetros calibrados.
        
        Returns:
            EconomyConfig calibrada
        """
        if not self.params:
            self.calibrate_all()
        
        config = EconomyConfig()
        
        # Aplicar parámetros calibrados
        if 'consumo' in self.params:
            config.consumo.c0 = self.params['consumo']['c0']
            config.consumo.c1 = self.params['consumo']['c1']
        
        if 'inversion' in self.params:
            config.inversion.I0 = self.params['inversion']['I0']
            config.inversion.b = self.params['inversion']['b']
        
        if 'dinero' in self.params:
            config.dinero.k = self.params['dinero']['k']
            config.dinero.h = self.params['dinero']['h']
        
        # Calcular G promedio
        if 'G' in self.data.columns:
            config.G = self.data['G'].mean()
        
        return config
    
    def get_fit_statistics(self) -> pd.DataFrame:
        """
        Devuelve estadísticas de ajuste de las calibraciones.
        
        Returns:
            DataFrame con R² y otros estadísticos
        """
        stats = []
        for key, vals in self.params.items():
            stats.append({
                'Ecuación': key,
                'R²': vals.get('R2', np.nan),
                'Parámetros': ', '.join([f"{k}={v:.4f}" for k, v in vals.items() if k != 'R2'])
            })
        
        return pd.DataFrame(stats)
