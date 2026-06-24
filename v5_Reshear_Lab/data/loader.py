"""
SICM v5 Research Lab — Importación y Calibración de Datos
==========================================================
Fase 5: Carga datos reales y calibra parámetros del modelo.

Módulos:
- DataLoader: Importa CSV/Excel
- ModelCalibrator: Estima c, b, k, h desde datos reales vía OLS
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from ..core.parameters import EconomyConfig, ParametrosConsumo, ParametrosInversion, ParametrosDinero


class DataLoader:
    """
    Carga datos macroeconómicos desde CSV o Excel.

    Columnas esperadas:
        Y: Producto (PIB)
        C: Consumo
        I: Inversión
        G: Gasto público
        T: Impuestos
        M: Oferta monetaria
        r: Tipo de interés
        P: Nivel de precios (opcional, default=1)
    """

    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.parquet']

    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """Carga datos desde CSV"""
        return pd.read_csv(filepath, **kwargs)

    @staticmethod
    def load_excel(filepath: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """Carga datos desde Excel"""
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def load_parquet(filepath: str, **kwargs) -> pd.DataFrame:
        """Carga datos desde Parquet"""
        return pd.read_parquet(filepath, **kwargs)

    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Valida que el DataFrame tenga las columnas requeridas.

        Args:
            df: DataFrame a validar
            required_columns: Lista de columnas requeridas (default: ['Y','C','I','G','r'])

        Returns:
            True si pasa validación

        Raises:
            ValueError: Si faltan columnas
        """
        if required_columns is None:
            required_columns = ['Y', 'C', 'I', 'G', 'r']

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}. Disponibles: {list(df.columns)}")

        # Verificar que no haya NaNs en columnas requeridas
        na_counts = df[required_columns].isna().sum()
        if na_counts.any():
            cols_with_na = na_counts[na_counts > 0].index.tolist()
            raise ValueError(f"Columnas con valores faltantes: {cols_with_na}")

        return True

    @staticmethod
    def generate_sample_data(n_periods: int = 40, seed: int = 42) -> pd.DataFrame:
        """
        Genera datos sintéticos de ejemplo para pruebas.

        Args:
            n_periods: Número de períodos
            seed: Semilla aleatoria

        Returns:
            DataFrame con datos macro sintéticos
        """
        np.random.seed(seed)

        # Parámetros verdaderos (para generar datos)
        c0_true, c1_true = 100, 0.75
        I0_true, b_true = 150, 20
        k_true, h_true = 0.5, 10
        G_bar, T_bar = 200, 100
        M_bar, P_bar = 500, 1.0

        # Simular choques aleatorios
        u_c = np.random.normal(0, 10, n_periods)
        u_i = np.random.normal(0, 15, n_periods)
        u_m = np.random.normal(0, 20, n_periods)

        # Generar Y, r de equilibrio IS-LM
        # Sistema: Y = (c0 + c1*(Y-T) + I0 - b*r + G)  =>  Y(1-c1) = c0 - c1*T + I0 - b*r + G
        #          M/P = k*Y - h*r  =>  r = (k*Y - M/P)/h
        # Sustituyendo: Y = [c0 - c1*T + I0 + G - b*(k*Y - M/P)/h] / (1-c1)
        # Y * [(1-c1) + b*k/h] = c0 - c1*T + I0 + G + b*M/(P*h)

        Y = np.zeros(n_periods)
        r = np.zeros(n_periods)
        C = np.zeros(n_periods)
        I = np.zeros(n_periods)

        for t in range(n_periods):
            G_t = G_bar + np.random.normal(0, 20)
            T_t = T_bar + np.random.normal(0, 10)
            M_t = M_bar + u_m[t]

            denom = (1 - c1_true) + (b_true * k_true) / h_true
            numer = c0_true + u_c[t] - c1_true * T_t + I0_true + u_i[t] + G_t + (b_true * M_t) / (P_bar * h_true)

            Y[t] = numer / denom
            r[t] = (k_true * Y[t] - M_t / P_bar) / h_true

            C[t] = c0_true + u_c[t] + c1_true * (Y[t] - T_t)
            I[t] = I0_true + u_i[t] - b_true * r[t]

        df = pd.DataFrame({
            'periodo': range(1, n_periods + 1),
            'Y': Y,
            'C': C,
            'I': I,
            'G': G_bar + np.random.normal(0, 20, n_periods),
            'T': T_bar + np.random.normal(0, 10, n_periods),
            'M': M_bar + u_m,
            'r': r,
            'P': P_bar
        })

        return df


class ModelCalibrator:
    """
    Calibra parámetros del modelo a partir de datos reales vía MCO (OLS).

    Estima:
        - c0, c1: función de consumo C = c0 + c1*(Y-T)
        - I0, b: función de inversión I = I0 - b*r
        - k, h: demanda de dinero M/P = k*Y - h*r
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame con columnas Y, C, I, G, T, M, r, P
        """
        self.data = data.copy()
        self.params: Dict = {}
        self.fit_stats: Dict = {}

    def calibrate_consumo(self) -> Dict:
        """
        Estima función de consumo: C = c0 + c1*(Y-T) + u

        Returns:
            Dict con c0, c1 estimados y estadísticos
        """
        Y = self.data['Y'].values
        C = self.data['C'].values
        T = self.data.get('T', pd.Series(np.zeros(len(Y)))).values

        Yd = Y - T  # Ingreso disponible

        # MCO: C = c0 + c1*Yd
        X = np.column_stack([np.ones(len(Yd)), Yd])
        beta = np.linalg.lstsq(X, C, rcond=None)[0]

        c0, c1 = beta[0], beta[1]
        C_pred = c0 + c1 * Yd

        # Estadísticos
        residuals = C - C_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((C - np.mean(C)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(C)
        k_params = 2
        s2 = ss_res / (n - k_params) if n > k_params else np.inf

        # Error estándar de c1
        var_c1 = s2 / np.sum((Yd - np.mean(Yd)) ** 2)
        se_c1 = np.sqrt(var_c1) if var_c1 > 0 else np.inf

        self.params['consumo'] = {
            'c0': float(c0),
            'c1': float(c1),
            'R2': float(r2),
            'RMSE': float(np.sqrt(ss_res / n)),
            'SE_c1': float(se_c1)
        }

        return self.params['consumo']

    def calibrate_inversion(self) -> Dict:
        """
        Estima función de inversión: I = I0 - b*r + u

        Returns:
            Dict con I0, b estimados y estadísticos
        """
        I = self.data['I'].values
        r = self.data['r'].values

        # MCO: I = I0 + (-b)*r
        X = np.column_stack([np.ones(len(r)), r])
        beta = np.linalg.lstsq(X, I, rcond=None)[0]

        I0, b = beta[0], -beta[1]
        I_pred = I0 - b * r

        residuals = I - I_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((I - np.mean(I)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(I)

        self.params['inversion'] = {
            'I0': float(I0),
            'b': float(b),
            'R2': float(r2),
            'RMSE': float(np.sqrt(ss_res / n))
        }

        return self.params['inversion']

    def calibrate_dinero(self) -> Dict:
        """
        Estima demanda de dinero: M/P = k*Y - h*r + u

        Returns:
            Dict con k, h estimados y estadísticos
        """
        M = self.data['M'].values
        P = self.data.get('P', pd.Series(np.ones(len(M)))).values
        Y = self.data['Y'].values
        r = self.data['r'].values

        Md_real = M / P

        # MCO: Md/P = k*Y - h*r
        X = np.column_stack([Y, r])
        beta = np.linalg.lstsq(X, Md_real, rcond=None)[0]

        k, h = beta[0], -beta[1]
        Md_pred = k * Y - h * r

        residuals = Md_real - Md_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Md_real - np.mean(Md_real)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(Md_real)

        self.params['dinero'] = {
            'k': float(k),
            'h': float(h),
            'R2': float(r2),
            'RMSE': float(np.sqrt(ss_res / n))
        }

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
            EconomyConfig calibrada lista para simulación
        """
        if not self.params:
            self.calibrate_all()

        config = EconomyConfig()

        if 'consumo' in self.params:
            config.consumo.c0 = self.params['consumo']['c0']
            config.consumo.c1 = self.params['consumo']['c1']

        if 'inversion' in self.params:
            config.inversion.I0 = self.params['inversion']['I0']
            config.inversion.b = self.params['inversion']['b']

        if 'dinero' in self.params:
            config.dinero.k = self.params['dinero']['k']
            config.dinero.h = self.params['dinero']['h']

        if 'G' in self.data.columns:
            config.G = float(self.data['G'].mean())

        if 'T' in self.data.columns:
            config.consumo.T = float(self.data['T'].mean())

        if 'M' in self.data.columns:
            config.dinero.M = float(self.data['M'].mean())

        if 'P' in self.data.columns:
            config.dinero.P = float(self.data['P'].mean())

        return config

    def get_fit_statistics(self) -> pd.DataFrame:
        """
        Devuelve estadísticas de ajuste de las calibraciones.

        Returns:
            DataFrame con R², RMSE y parámetros
        """
        if not self.params:
            self.calibrate_all()

        stats = []
        for key, vals in self.params.items():
            row = {
                'Ecuación': key,
                'R²': round(vals.get('R2', np.nan), 4),
                'RMSE': round(vals.get('RMSE', np.nan), 4),
                'Parámetros': ', '.join([f"{k}={round(v, 4)}" 
                    for k, v in vals.items() if k not in ['R2', 'RMSE', 'SE_c1']])
            }
            if 'SE_c1' in vals:
                row['SE(c1)'] = round(vals['SE_c1'], 4)
            stats.append(row)

        return pd.DataFrame(stats)

    def summary(self) -> str:
        """Resumen textual de la calibración"""
        stats = self.get_fit_statistics()
        lines = [
            "=" * 60,
            "CALIBRACIÓN DEL MODELO IS-LM",
            "=" * 60,
            "",
            stats.to_string(index=False),
            "",
            "Interpretación:",
            f"  - c1 (PMC) = {self.params.get('consumo', {}).get('c1', 'N/A'):.4f}",
            f"  - b (Sens. inversión) = {self.params.get('inversion', {}).get('b', 'N/A'):.4f}",
            f"  - k (Sens. dinero a Y) = {self.params.get('dinero', {}).get('k', 'N/A'):.4f}",
            f"  - h (Sens. dinero a r) = {self.params.get('dinero', {}).get('h', 'N/A'):.4f}",
            "",
            "=" * 60
        ]
        return "\n".join(lines)
