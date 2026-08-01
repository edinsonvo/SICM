"""Gestión de datos del SICM v2.0.

La clase :class:`DataManager` genera datos sintéticos realistas, importa
series desde CSV/Excel con validación y ofrece herramientas de análisis:
estadística descriptiva, brecha del producto, regla de Taylor y ciclo
económico (filtro HP).
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["fecha", "PIB", "Inflacion", "Tasa_Interes",
                    "Desempleo", "Tipo_Cambio"]

VARIABLE_LABELS = {
    "PIB": "PIB",
    "Inflacion": "Inflación (%)",
    "Tasa_Interes": "Tasa de interés (%)",
    "Desempleo": "Desempleo (%)",
    "Tipo_Cambio": "Tipo de cambio",
    "PIB_Potencial": "PIB potencial",
}

VARIABLE_COLORS = {
    "PIB": "#1f77b4",
    "Inflacion": "#d62728",
    "Tasa_Interes": "#2ca02c",
    "Desempleo": "#ff7f0e",
    "Tipo_Cambio": "#9467bd",
    "PIB_Potencial": "#bcbd22",
}


class DataManager:
    """Genera e importa series macroeconómicas y calcula indicadores."""

    # ------------------------------------------------------------------
    # Generación de datos sintéticos
    # ------------------------------------------------------------------
    @staticmethod
    def generate_sample_data(n=240, start="2000-01-01"):
        """Genera 240 observaciones mensuales con estructura realista.

        Cada serie combina tendencia, ciclo y ruido, con vínculos
        coherentes entre variables (ley de Okun, regla de Taylor, etc.).

        Parámetros
        ----------
        n : int
            Número de observaciones (por defecto 240 meses).
        start : str
            Fecha inicial de la serie.

        Devuelve
        --------
        pandas.DataFrame
            Con columnas fecha, PIB, PIB_Potencial, Inflacion,
            Tasa_Interes, Desempleo y Tipo_Cambio.
        """
        rng = np.random.default_rng(42)
        dates = pd.date_range(start=start, periods=n, freq="MS")
        t = np.arange(n)

        # Tendencia + ciclo del PIB
        trend = 100 + 0.20 * t  # crecimiento de ~2.4 % anual
        cycle_amp = 4.5
        cycle = cycle_amp * np.sin(2 * np.pi * t / 96.0 + 0.8) + \
            cycle_amp * 0.35 * np.sin(2 * np.pi * t / 28.0)
        # Ruido moderado (el ciclo domina al ruido)
        gdp_noise = rng.normal(0, 0.9, n)
        pib = trend + cycle + gdp_noise
        pib_potencial = trend

        # Brecha del producto (%)
        gap = (pib - pib_potencial) / pib_potencial * 100.0

        # Inflación anualizada (AR(1) + ciclo + ruido), alrededor de 3.5 %
        infl_noise = rng.normal(0, 0.25, n)
        inflacion = np.empty(n)
        inflacion[0] = 3.5
        for i in range(1, n):
            inflacion[i] = (0.94 * (inflacion[i - 1] - 3.5) + 3.5
                            + 0.18 * gap[i - 1] + infl_noise[i])
        inflacion = np.clip(inflacion, 0.5, 9.0)

        # Tasa de interés: regla de Taylor con suavizado
        tasa = 3.5 + 1.5 * (inflacion - 3.5) + 0.5 * gap
        tasa = tasa + rng.normal(0, 0.2, n)

        # Desempleo: ley de Okun (u = u* - 0.5·gap)
        u_star = 5.5
        desempleo = u_star - 0.5 * gap + rng.normal(0, 0.15, n)
        desempleo = np.clip(desempleo, 2.5, 12.0)

        # Tipo de cambio: caminata aleatoria con reversión a la media
        tipo_cambio = np.empty(n)
        tipo_cambio[0] = 1.0
        for i in range(1, n):
            tipo_cambio[i] = (tipo_cambio[i - 1]
                              + 0.02 * (1.0 - tipo_cambio[i - 1])
                              + rng.normal(0, 0.012))
        tipo_cambio = np.clip(tipo_cambio, 0.6, 1.6)

        return pd.DataFrame({
            "fecha": dates,
            "PIB": pib,
            "PIB_Potencial": pib_potencial,
            "Inflacion": inflacion,
            "Tasa_Interes": tasa,
            "Desempleo": desempleo,
            "Tipo_Cambio": tipo_cambio,
        })

    # ------------------------------------------------------------------
    # Importación de datos
    # ------------------------------------------------------------------
    @staticmethod
    def load_data(file, file_name=""):
        """Carga datos CSV o Excel con validación de formato.

        Parámetros
        ----------
        file : archivo cargado por Streamlit (o ruta).
        file_name : nombre del archivo (para detectar la extensión).

        Devuelve
        --------
        pandas.DataFrame
            Datos normalizados con columna ``fecha`` tipo datetime.

        Lanza
        -----
        ValueError
            Si el formato, las columnas o los tipos de dato son inválidos.
        """
        name = (file_name or getattr(file, "name", "")).lower()
        try:
            if name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
            elif name.endswith(".csv") or not name:
                df = pd.read_csv(file)
            else:
                raise ValueError(
                    "Formato no soportado. Use un archivo CSV o Excel (.xlsx).")
        except ValueError as exc:
            raise exc
        except Exception as exc:
            raise ValueError(
                f"No se pudo leer el archivo: {exc}. Verifique que esté "
                "bien formado y que tenga extensión .csv o .xlsx.") from exc

        df = DataManager._validate(df)
        return df

    @staticmethod
    def _validate(df):
        """Valida columnas y tipos; normaliza la fecha y datos numéricos."""
        df = df.copy()
        if df.empty:
            raise ValueError("El archivo está vacío.")

        # Normalizar nombres de columna (mayúsculas / espacios)
        rename = {}
        for col in df.columns:
            key = str(col).strip().replace(" ", "_")
            rename[col] = key
        df = df.rename(columns=rename)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "Faltan columnas obligatorias: "
                + ", ".join(missing)
                + ". Columnas requeridas: " + ", ".join(REQUIRED_COLUMNS) + ".")

        # Fecha
        try:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="raise")
        except Exception as exc:
            raise ValueError(
                "La columna 'fecha' no tiene un formato de fecha válido. "
                f"Detalle: {exc}") from exc

        # Datos numéricos
        for col in ["PIB", "Inflacion", "Tasa_Interes", "Desempleo",
                    "Tipo_Cambio"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().all():
                raise ValueError(
                    f"La columna '{col}' no contiene valores numéricos válidos.")

        df = df.sort_values("fecha").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------------
    @staticmethod
    def descriptive_stats(df):
        """Estadística descriptiva de las series + cambios de los últimos 12 meses."""
        numeric = [c for c in REQUIRED_COLUMNS[1:] if c in df.columns]
        stats = df[numeric].describe().T
        stats["Var_12m_abs"] = [
            DataManager.recent_changes(df, c, 12)[0] for c in numeric
        ]
        stats["Var_12m_%"] = [
            DataManager.recent_changes(df, c, 12)[1] for c in numeric
        ]
        return stats

    @staticmethod
    def annual_change(df, column):
        """Variación porcentual anual de una columna (últimos 12 meses)."""
        if column not in df.columns or len(df) < 13:
            return np.nan
        current = df[column].iloc[-1]
        previous = df[column].iloc[-13]
        if previous == 0 or np.isnan(previous):
            return np.nan
        return (current - previous) / previous * 100.0

    @staticmethod
    def output_gap(df):
        """Brecha del producto (%) respecto al PIB potencial (filtro HP)."""
        if "PIB" not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        trend = DataManager.hp_trend(df, "PIB")
        gap = (df["PIB"] - trend) / trend * 100.0
        return gap

    @staticmethod
    def hp_trend(df, column, lamb=129600):
        """Tendencia de Hodrick-Prescott de una serie mensual.

        Si ``statsmodels`` no está disponible, usa una media móvil centrada
        de 13 términos como aproximación.
        """
        series = pd.Series(df[column].values, index=df["fecha"]).dropna()
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(series, lamb=lamb)
            return pd.Series(trend.values, index=series.index)
        except Exception:
            trend = series.rolling(window=13, center=True, min_periods=1).mean()
            return pd.Series(trend.values, index=series.index)

    @staticmethod
    def taylor_recommendation(df, r_neutral=2.5, pi_target=3.5,
                              w_inflation=1.5, w_output=0.5):
        """Tasa de política recomendada por la regla de Taylor.

        r_taylor = r_neutral + 1.5·(inflación - objetivo) + 0.5·(brecha del producto).
        """
        if len(df) == 0:
            return np.nan, np.nan, np.nan
        inflation = df["Inflacion"].iloc[-1]
        gap = DataManager.output_gap(df).iloc[-1]
        if np.isnan(inflation) or np.isnan(gap):
            return np.nan, np.nan, np.nan
        rate = r_neutral + w_inflation * (inflation - pi_target) + w_output * gap
        return rate, inflation, gap

    @staticmethod
    def cyclical_analysis(df):
        """Descompone el PIB en tendencia y ciclo (filtro HP)."""
        trend = DataManager.hp_trend(df, "PIB")
        cycle = pd.Series(df["PIB"].values, index=df["fecha"]) - trend
        out = pd.DataFrame({
            "fecha": df["fecha"],
            "PIB": df["PIB"].values,
            "PIB_Potencial": trend.values,
            "Ciclo": cycle.values,
        })
        return out

    @staticmethod
    def recent_changes(df, column, periods=12):
        """Cambio absoluto y porcentual reciente de una serie."""
        if column not in df.columns or len(df) < 2:
            return 0.0, 0.0
        last = df[column].iloc[-1]
        prev = df[column].iloc[-1 - min(periods, len(df) - 1)]
        diff = last - prev
        pct = (diff / prev * 100.0) if prev else 0.0
        return diff, pct
