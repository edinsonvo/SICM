"""Modelos macroeconómicos del Simulador Integral de Choques Macroeconómicos (SICM) v2.0.

Contiene las clases :class:`ISLMModel`, :class:`ADASModel` y
:class:`MundellFlemingModel`, cada una con su sistema de ecuaciones,
resolución numérica (SciPy ``fsolve``), curvas para visualización,
multiplicadores y aplicación de choques exógenos.

Convenciones de escala:
- La producción ``Y`` se mide en unidades de índice (u. i.).
- La tasa de interés ``r`` se almacena como fracción (0.05 = 5 %).
- ``100 * b * r`` convierte la sensibilidad de la inversión a un
  cambio de 1 punto porcentual.
"""

import numpy as np
from scipy.optimize import fsolve


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _safe_solve(func, guess):
    """Resuelve ``func`` = 0 con ``fsolve`` devolviendo ``None`` si no converge.

    Parámetros
    ----------
    func : callable
        Función del sistema de ecuaciones.
    guess : list[float]
        Valor inicial para el solver.

    Devuelve
    --------
    numpy.ndarray | None
        Solución numérica o ``None`` si hubo fallo / no convergencia.
    """
    try:
        sol, info, ier, _ = fsolve(func, guess, full_output=True)
        if ier == 1 and np.all(np.isfinite(sol)):
            return sol
    except Exception:
        pass
    return None


def _merge_params(defaults, params):
    """Fusiona parámetros por defecto con los proporcionados por el usuario."""
    merged = dict(defaults)
    if params:
        merged.update(params)
    return merged


# ---------------------------------------------------------------------------
# Catálogo de choques (usado por la UI, simulación y reportes)
# ---------------------------------------------------------------------------
SHOCK_CATALOG = {
    "IS-LM": [
        {
            "id": "G_up",
            "label": "Gasto Gobierno ↑",
            "param": "G",
            "sign": +1,
            "mechanism": (
                "Aumento del gasto público (política fiscal expansiva). La curva IS se "
                "desplaza a la derecha: sube la demanda agregada, la producción y la tasa "
                "de interés. El alza de la tasa expulsa parcialmente la inversión privada "
                "(efecto crowding out)."
            ),
        },
        {
            "id": "G_down",
            "label": "Gasto Gobierno ↓",
            "param": "G",
            "sign": -1,
            "mechanism": (
                "Reducción del gasto público (política fiscal contractiva). La IS se "
                "desplaza a la izquierda: cae la demanda agregada, la producción y la "
                "tasa de interés, lo que estimula parcialmente la inversión privada."
            ),
        },
        {
            "id": "M_up",
            "label": "Oferta Monetaria ↑",
            "param": "M",
            "sign": +1,
            "mechanism": (
                "Expansión monetaria (política monetaria expansiva). La LM se desplaza a "
                "la derecha: aumentan los saldos reales, baja la tasa de interés y sube la "
                "producción (la inversión responde a tasas más bajas)."
            ),
        },
        {
            "id": "M_down",
            "label": "Oferta Monetaria ↓",
            "param": "M",
            "sign": -1,
            "mechanism": (
                "Contracción monetaria (política monetaria contractiva). La LM se "
                "desplaza a la izquierda: caen los saldos reales, sube la tasa de interés "
                "y se reduce la producción e inversión."
            ),
        },
        {
            "id": "T_up",
            "label": "Impuestos ↑",
            "param": "T",
            "sign": +1,
            "mechanism": (
                "Aumento de impuestos. La renta disponible cae, lo que reduce el consumo "
                "y desplaza la IS a la izquierda: menor producción y menor tasa de interés."
            ),
        },
        {
            "id": "T_down",
            "label": "Impuestos ↓",
            "param": "T",
            "sign": -1,
            "mechanism": (
                "Reducción de impuestos. La renta disponible sube, el consumo aumenta y "
                "la IS se desplaza a la derecha: mayor producción y mayor tasa de interés."
            ),
        },
    ],
    "AD-AS": [
        {
            "id": "M_up",
            "label": "Oferta Monetaria ↑",
            "param": "M",
            "sign": +1,
            "mechanism": (
                "Choque de demanda positivo. La AD se desplaza a la derecha: en el corto "
                "plazo suben la producción y el nivel de precios por encima del producto "
                "natural (brecha positiva)."
            ),
        },
        {
            "id": "M_down",
            "label": "Oferta Monetaria ↓",
            "param": "M",
            "sign": -1,
            "mechanism": (
                "Choque de demanda negativo. La AD se desplaza a la izquierda: cae la "
                "producción y el nivel de precios en el corto plazo (brecha negativa)."
            ),
        },
        {
            "id": "Pe_up",
            "label": "Precios Esperados ↑",
            "param": "Pe_factor",
            "sign": +1,
            "mechanism": (
                "Choque de expectativas. La SRAS se desplaza hacia arriba: sube el nivel "
                "de precios y cae la producción en el corto plazo (estanflación moderada)."
            ),
        },
        {
            "id": "Pe_down",
            "label": "Precios Esperados ↓",
            "param": "Pe_factor",
            "sign": -1,
            "mechanism": (
                "Revisión a la baja de las expectativas de precios. La SRAS se desplaza "
                "hacia abajo: cae el nivel de precios y sube la producción en el corto plazo."
            ),
        },
        {
            "id": "Yn_up",
            "label": "Productividad ↑",
            "param": "Yn",
            "sign": +1,
            "mechanism": (
                "Choque de oferta positivo (mejora tecnológica). Aumenta la producción "
                "natural: la LRAS y la SRAS se desplazan a la derecha, sube Y y cae P."
            ),
        },
        {
            "id": "Yn_down",
            "label": "Productividad ↓",
            "param": "Yn",
            "sign": -1,
            "mechanism": (
                "Choque de oferta adverso (alza del precio del petróleo, desastres "
                "naturales). Cae la producción natural: la LRAS y la SRAS se desplazan a "
                "la izquierda, sube P y cae Y (estanflación)."
            ),
        },
    ],
    "Mundell-Fleming": [
        {
            "id": "G_up",
            "label": "Gasto Gobierno ↑",
            "param": "G",
            "sign": +1,
            "mechanism": (
                "Política fiscal expansiva. Con tipo de cambio flexible y alta movilidad "
                "de capitales, la IS* se desplaza a la derecha, el tipo de cambio se "
                "aprecia y la producción no cambia (crowding out vía exportaciones netas). "
                "Con tipo fijo, el banco central expande la oferta monetaria y la "
                "producción sí aumenta."
            ),
        },
        {
            "id": "G_down",
            "label": "Gasto Gobierno ↓",
            "param": "G",
            "sign": -1,
            "mechanism": (
                "Política fiscal contractiva. Con tipo flexible, el tipo de cambio se "
                "deprecia y la producción no cambia; con tipo fijo cae la producción y la "
                "oferta monetaria."
            ),
        },
        {
            "id": "M_up",
            "label": "Oferta Monetaria ↑",
            "param": "M",
            "sign": +1,
            "mechanism": (
                "Política monetaria expansiva. Con tipo de cambio flexible, la LM* se "
                "desplaza a la derecha: sube la producción y el tipo de cambio se deprecia. "
                "Con tipo fijo la política monetaria es endógena y no puede aplicarse."
            ),
        },
        {
            "id": "M_down",
            "label": "Oferta Monetaria ↓",
            "param": "M",
            "sign": -1,
            "mechanism": (
                "Política monetaria contractiva. Con tipo flexible, la LM* se desplaza a "
                "la izquierda: cae la producción y el tipo de cambio se aprecia. Con tipo "
                "fijo no es aplicable."
            ),
        },
        {
            "id": "rw_up",
            "label": "Tasa Mundial ↑",
            "param": "r_w",
            "sign": +1,
            "mechanism": (
                "Alza de la tasa de interés mundial. La cuenta financiera se vuelve menos "
                "atractiva (salida de capitales), el tipo de cambio se deprecia y, vía "
                "exportaciones netas, la producción puede aumentar."
            ),
        },
        {
            "id": "rw_down",
            "label": "Tasa Mundial ↓",
            "param": "r_w",
            "sign": -1,
            "mechanism": (
                "Baja de la tasa de interés mundial. Entran capitales, el tipo de cambio "
                "se aprecia y las exportaciones netas caen, reduciendo la producción."
            ),
        },
    ],
}

# Mapa modelo -> catálogo de choques aplicable
SHOCK_KEYS = {
    "IS-LM": [s["label"] for s in SHOCK_CATALOG["IS-LM"]],
    "AD-AS": [s["label"] for s in SHOCK_CATALOG["AD-AS"]],
    "Mundell-Fleming": [s["label"] for s in SHOCK_CATALOG["Mundell-Fleming"]],
}

_ISLM_SHOCK_MAP = {s["label"]: s for s in SHOCK_CATALOG["IS-LM"]}
_ADAS_SHOCK_MAP = {s["label"]: s for s in SHOCK_CATALOG["AD-AS"]}
_MF_SHOCK_MAP = {s["label"]: s for s in SHOCK_CATALOG["Mundell-Fleming"]}


def get_shock_mechanism(model_name, shock_label):
    """Devuelve la descripción del mecanismo de transmisión de un choque."""
    catalog = {"IS-LM": _ISLM_SHOCK_MAP, "AD-AS": _ADAS_SHOCK_MAP,
               "Mundell-Fleming": _MF_SHOCK_MAP}.get(model_name, {})
    shock = catalog.get(shock_label)
    return shock["mechanism"] if shock else ""


def apply_shock_to_params(params, shock_map, shock_label, magnitude):
    """Aplica un choque sobre una copia de los parámetros (sin mutar el original)."""
    shock = shock_map.get(shock_label)
    if shock is None:
        return params
    new_params = dict(params)
    sign = shock["sign"]
    param = shock["param"]
    if param == "Pe_factor":
        new_params["Pe_factor"] = new_params["Pe_factor"] * (1 + sign * magnitude)
    else:
        new_params[param] = new_params[param] * (1 + sign * magnitude)
    return new_params


# ---------------------------------------------------------------------------
# Modelo IS-LM
# ---------------------------------------------------------------------------
class ISLMModel:
    """Modelo IS-LM de economía cerrada con ecuaciones completas.

    Mercado de bienes (IS):   Y = C + I + G
    Mercado de dinero (LM):   M/P = k·Y - h·r

    con  C = C0 + c·(Y - T)   y   I = I0 - 100·b·r.

    Métodos principales: :meth:`solve`, :meth:`apply_shock`,
    :meth:`get_is_curve`, :meth:`get_lm_curve`, :meth:`multipliers`.
    """

    DEFAULTS = {
        "C0": 50.0,   # Consumo autónomo
        "c": 0.75,    # Propensión marginal a consumir
        "I0": 100.0,  # Inversión autónoma
        "b": 0.4,     # Sensibilidad de la inversión a la tasa (por p.p.)
        "G": 120.0,   # Gasto del gobierno
        "T": 80.0,    # Impuestos
        "M": 200.0,   # Oferta monetaria nominal
        "P": 1.0,     # Nivel de precios
        "k": 0.5,     # Sensibilidad de la demanda de dinero al ingreso
        "h": 1500.0,  # Sensibilidad de la demanda de dinero a la tasa
    }

    def __init__(self, params=None):
        """Inicializa el modelo con parámetros dados (o por defecto)."""
        self.params = _merge_params(self.DEFAULTS, params)
        self.equilibrium = None

    # --- Sistema de ecuaciones -------------------------------------------
    def equations(self, vars_):
        """Devuelve [IS, LM] evaluadas en (Y, r)."""
        Y, r = vars_
        p = self.params
        C = p["C0"] + p["c"] * (Y - p["T"])
        I = p["I0"] - 100.0 * p["b"] * r
        IS = Y - (C + I + p["G"])
        LM = p["M"] / p["P"] - (p["k"] * Y - p["h"] * r)
        return [IS, LM]

    # --- Solución ----------------------------------------------------------
    def solve(self, initial_guess=None):
        """Resuelve el equilibrio general (Y*, r*).

        Devuelve un diccionario con Y, r, C, I, G y los multiplicadores.
        Si el solver no converge, aplica la solución analítica (fallback).
        """
        if initial_guess is None:
            initial_guess = [800.0, 0.10]
        sol = _safe_solve(self.equations, initial_guess)
        if sol is None:
            sol = np.array(self._analytical_solution())
        Y, r = float(sol[0]), float(sol[1])
        p = self.params
        C = p["C0"] + p["c"] * (Y - p["T"])
        I = p["I0"] - 100.0 * p["b"] * r
        self.equilibrium = {
            "Y": Y,
            "r": r,
            "C": C,
            "I": I,
            "G": p["G"],
            "T": p["T"],
            "mult_dYdG": self.multipliers()["dY_dG"],
            "mult_dYdM": self.multipliers()["dY_dM"],
        }
        return self.equilibrium

    def _analytical_solution(self):
        """Solución analítica del sistema IS-LM (respaldo del solver)."""
        p = self.params
        A0 = p["C0"] - p["c"] * p["T"] + p["I0"] + p["G"]
        den = p["h"] * (1 - p["c"]) + 100.0 * p["b"] * p["k"]
        if abs(den) < 1e-9:
            return [100.0, 0.05]
        Y = (p["h"] * A0 + 100.0 * p["b"] * p["M"] / p["P"]) / den
        r = (p["k"] * Y - p["M"] / p["P"]) / p["h"]
        return [Y, r]

    # --- Multiplicadores ----------------------------------------------------
    def multipliers(self):
        """Multiplicadores de política del modelo IS-LM.

        dY/dG : multiplicador fiscal con reacción endógena de r.
        dY/dM : multiplicador monetario.

        Derivados de la solución general:  Y = [h·A0 + 100·b·(M/P)] /
                                             [h(1-c) + 100·b·k]
        """
        p = self.params
        den = p["h"] * (1 - p["c"]) + 100.0 * p["b"] * p["k"]
        if abs(den) < 1e-9:
            return {"dY_dG": 0.0, "dY_dM": 0.0}
        dY_dG = p["h"] / den
        dY_dM = 100.0 * p["b"] / (p["P"] * den)
        return {"dY_dG": dY_dG, "dY_dM": dY_dM}

    # --- Curvas para visualización ------------------------------------------
    def get_is_curve(self, Y_range):
        """Devuelve (Y, r) de la curva IS para un rango de producción."""
        p = self.params
        A0 = p["C0"] - p["c"] * p["T"] + p["I0"] + p["G"]
        b = max(p["b"], 1e-6)
        r = (A0 - (1 - p["c"]) * np.asarray(Y_range, dtype=float)) / (100.0 * b)
        return np.asarray(Y_range, dtype=float), r

    def get_lm_curve(self, Y_range):
        """Devuelve (Y, r) de la curva LM para un rango de producción."""
        p = self.params
        h = max(p["h"], 1e-6)
        r = (p["k"] * np.asarray(Y_range, dtype=float) - p["M"] / p["P"]) / h
        return np.asarray(Y_range, dtype=float), r

    # --- Choques ------------------------------------------------------------
    def apply_shock(self, shock_label, magnitude):
        """Aplica un choque (por nombre) con la magnitud dada y resuelve de nuevo.

        La magnitud se interpreta como proporción (0.05 = 5 %). Devuelve el
        nuevo equilibrio y modifica los parámetros internos del modelo.
        """
        self.params = apply_shock_to_params(self.params, _ISLM_SHOCK_MAP,
                                            shock_label, magnitude)
        return self.solve()

    def clone(self):
        """Devuelve una copia independiente del modelo."""
        return ISLMModel(dict(self.params))


# ---------------------------------------------------------------------------
# Modelo AD-AS
# ---------------------------------------------------------------------------
class ADASModel:
    """Modelo de oferta y demanda agregadas.

    Demanda agregada:  Y = M·V / P
    Oferta de corto:   P = P_e · [1 + λ·(Y - Y_n)]
    Oferta de largo:   Y = Y_n

    El nivel de precios esperado se modela como  P_e = factor · P_e_base,
    donde ``P_e_base = M_base·V_base/Y_n_base`` (parámetros por defecto) es
    un *ancla* fija. De este modo, con ``factor = 1`` la economía parte de
    su equilibrio de largo plazo y los choques monetarios son **no neutrales**
    en el corto plazo (los precios esperados no se ajustan de inmediato). El
    factor corresponde al rango 0.5-2.0 de la interfaz.
    """

    DEFAULTS = {
        "M": 200.0,     # Oferta monetaria nominal
        "V": 5.0,       # Velocidad del dinero
        "Yn": 100.0,    # Producción natural
        "lambda": 0.05, # Pendiente de la SRAS
        "Pe_factor": 1.0,  # Multiplicador de precios esperados (ancla fija)
    }

    # Precios esperados base: ancla del equilibrio de largo plazo (M·V/Yn
    # evaluados con los parámetros por defecto). Se mantiene fija para que
    # los choques de demanda tengan efectos reales en el corto plazo.
    BASE_PE = DEFAULTS["M"] * DEFAULTS["V"] / DEFAULTS["Yn"]

    def __init__(self, params=None):
        """Inicializa el modelo AD-AS."""
        self.params = _merge_params(self.DEFAULTS, params)
        self.equilibrium = None

    @property
    def pe(self):
        """Nivel de precios esperado (anclado, no ligado a M/V/Yn actuales)."""
        return self.params["Pe_factor"] * ADASModel.BASE_PE

    def equations(self, vars_):
        """Devuelve [AD, SRAS] evaluadas en (Y, P)."""
        Y, P = vars_
        p = self.params
        AD = Y - p["M"] * p["V"] / P
        SRAS = P - self.pe * (1 + p["lambda"] * (Y - p["Yn"]))
        return [AD, SRAS]

    def solve(self, initial_guess=None):
        """Resuelve el equilibrio de corto plazo (Y, P)."""
        if initial_guess is None:
            initial_guess = [self.params["Yn"], self.pe]
        sol = _safe_solve(self.equations, initial_guess)
        if sol is None:
            sol = np.array([self.params["Yn"], self.pe])
        Y, P = float(sol[0]), float(sol[1])
        p = self.params
        gap = (Y - p["Yn"]) / p["Yn"] * 100.0
        self.equilibrium = {
            "Y": Y,
            "P": P,
            "Yn": p["Yn"],
            "Pe": self.pe,
            "gap": gap,          # brecha del producto (%)
            "long_run_P": self.pe,
        }
        return self.equilibrium

    def long_run_equilibrium(self):
        """Equilibrio de largo plazo (Y_n, P*)."""
        p = self.params
        return {"Y": p["Yn"], "P": self.pe}

    def get_ad_curve(self, Y_range):
        """Devuelve (Y, P) de la demanda agregada."""
        Y = np.asarray(Y_range, dtype=float)
        Y_safe = np.where(Y <= 1e-9, 1e-9, Y)
        P = self.params["M"] * self.params["V"] / Y_safe
        return Y, P

    def get_sras_curve(self, Y_range):
        """Devuelve (Y, P) de la oferta agregada de corto plazo."""
        p = self.params
        Y = np.asarray(Y_range, dtype=float)
        P = self.pe * (1 + p["lambda"] * (Y - p["Yn"]))
        return Y, P

    def apply_shock(self, shock_label, magnitude):
        """Aplica un choque (por nombre) y resuelve de nuevo."""
        self.params = apply_shock_to_params(self.params, _ADAS_SHOCK_MAP,
                                            shock_label, magnitude)
        return self.solve()

    def clone(self):
        """Devuelve una copia independiente del modelo."""
        return ADASModel(dict(self.params))


# ---------------------------------------------------------------------------
# Modelo Mundell-Fleming
# ---------------------------------------------------------------------------
class MundellFlemingModel:
    """Modelo Mundell-Fleming de economía abierta pequeña.

    IS*:  Y = C(Y - T) + I(r*) + G + NX(e)
    LM*:  M/P = L(Y, r*)
    BP:   NX(e) + CF(r* - r_w) = 0

    con  NX(e) = NX0 - θ·e,   CF = κ·(r* - r_w).

    Regímenes:
    - "Flexible": se resuelve (Y, e) con M exógeno.
    - "Fijo":     se resuelve (Y, M) con e = e_bar (M endógeno para defender el ancla).

    Movilidad de capitales ``kappa``:
    - 0  : nula  (la BP es vertical / el balance externo fija e).
    - 1-4: imperfecta (BP creciente).
    - 1e9: perfecta (r* = r_w; la BP es horizontal en (Y, e)).
    """

    DEFAULTS = {
        "C0": 50.0, "c": 0.75, "I0": 100.0, "b": 0.4,
        "G": 120.0, "T": 80.0,
        "M": 200.0, "P": 1.0, "k": 0.5, "h": 1500.0,
        "NX0": 30.0,     # Exportaciones netas autónomas
        "theta": 0.5,    # Sensibilidad de NX al tipo de cambio
        "r_star": 0.05,  # Tasa mundial (r*)
        "r_w": 0.05,     # Tasa mundial relevante para la BP
        "kappa": 1e9,    # Movilidad de capitales (1e9 = perfecta)
        "regime": "Flexible",  # "Flexible" | "Fijo"
        "e_bar": 200.0,  # Ancla cambiaria (régimen fijo)
    }

    def __init__(self, params=None):
        """Inicializa el modelo Mundell-Fleming."""
        self.params = _merge_params(self.DEFAULTS, params)
        self.equilibrium = None

    def _is_fixed(self):
        return str(self.params.get("regime", "Flexible")).lower() == "fijo"

    def equations(self, vars_):
        """Sistema de ecuaciones según régimen.

        - Flexible: [IS*, LM*] en (Y, e).
        - Fijo:     [IS*, LM*] en (Y, M) con e = e_bar.
        """
        p = self.params
        Y, x = vars_
        C = p["C0"] + p["c"] * (Y - p["T"])
        I = p["I0"] - 100.0 * p["b"] * p["r_star"]
        if self._is_fixed():
            e = p["e_bar"]
            M = x
        else:
            e = x
            M = p["M"]
        NX = p["NX0"] - p["theta"] * e
        IS = Y - (C + I + p["G"] + NX)
        LM = M / p["P"] - (p["k"] * Y - p["h"] * p["r_star"])
        return [IS, LM]

    def solve(self, initial_guess=None):
        """Resuelve el equilibrio según el régimen cambiario.

        Devuelve un diccionario con Y, e, M, NX, BP y el régimen.
        """
        p = self.params
        if self._is_fixed():
            if initial_guess is None:
                initial_guess = [150.0, 200.0]
            sol = _safe_solve(self.equations, initial_guess)
            if sol is None:
                # Solución directa: IS* despeja Y, LM* despeja M.
                Y = (p["C0"] - p["c"] * p["T"] + p["I0"]
                     - 100.0 * p["b"] * p["r_star"] + p["G"]
                     + p["NX0"] - p["theta"] * p["e_bar"]) / (1 - p["c"])
                M = p["P"] * (p["k"] * Y - p["h"] * p["r_star"])
                sol = np.array([Y, M])
            Y, M = float(sol[0]), float(sol[1])
            e = p["e_bar"]
        else:
            if initial_guess is None:
                initial_guess = [250.0, 150.0]
            sol = _safe_solve(self.equations, initial_guess)
            if sol is None:
                # Solución directa: LM* despeja Y, IS* despeja e.
                Y = (p["M"] / p["P"] + p["h"] * p["r_star"]) / p["k"]
                e = (p["C0"] - p["c"] * p["T"] + p["I0"]
                     - 100.0 * p["b"] * p["r_star"] + p["G"]
                     + p["NX0"] - (1 - p["c"]) * Y) / p["theta"]
                M = p["M"]
                sol = np.array([Y, e])
            Y, e = float(sol[0]), float(sol[1])
            M = p["M"]

        NX = p["NX0"] - p["theta"] * e
        CF = p["kappa"] * (p["r_star"] - p["r_w"])
        self.equilibrium = {
            "Y": Y,
            "e": e,
            "M": M,
            "NX": NX,
            "CF": CF,
            "BP": NX + CF,
            "r_star": p["r_star"],
            "regime": p["regime"],
            "e_bar": p["e_bar"],
        }
        return self.equilibrium

    def get_is_curve(self, Y_range):
        """Devuelve (Y, e) de la IS* en el plano (Y, e)."""
        p = self.params
        Y = np.asarray(Y_range, dtype=float)
        e = (p["C0"] - p["c"] * p["T"] + p["I0"]
             - 100.0 * p["b"] * p["r_star"] + p["G"]
             + p["NX0"] - (1 - p["c"]) * Y) / p["theta"]
        return Y, e

    def get_lm_curve(self, Y_range):
        """Devuelve (Y, e) de la LM* (vertical en el plano (Y, e))."""
        p = self.params
        Y_eq = (p["M"] / p["P"] + p["h"] * p["r_star"]) / p["k"]
        Y = np.asarray(Y_range, dtype=float)
        return Y, np.full_like(Y, Y_eq)

    def get_bp_curve(self, Y_range):
        """Devuelve (Y, e) de la BP (horizontal en el plano (Y, e))."""
        p = self.params
        kappa = max(p["kappa"], 1e-9)
        e_bp = (p["NX0"] + kappa * (p["r_star"] - p["r_w"])) / p["theta"]
        Y = np.asarray(Y_range, dtype=float)
        return Y, np.full_like(Y, e_bp)

    def apply_shock(self, shock_label, magnitude):
        """Aplica un choque (por nombre) y resuelve de nuevo."""
        self.params = apply_shock_to_params(self.params, _MF_SHOCK_MAP,
                                            shock_label, magnitude)
        return self.solve()

    def clone(self):
        """Devuelve una copia independiente del modelo."""
        return MundellFlemingModel(dict(self.params))


# ---------------------------------------------------------------------------
# Resolvers genéricos de políticas (usados por policy.py y app.py)
# ---------------------------------------------------------------------------
def build_model(model_name, params):
    """Fabrica la instancia de modelo indicada por ``model_name``."""
    if model_name == "IS-LM":
        return ISLMModel(params)
    if model_name == "AD-AS":
        return ADASModel(params)
    if model_name == "Mundell-Fleming":
        return MundellFlemingModel(params)
    raise ValueError(f"Modelo desconocido: {model_name}")
