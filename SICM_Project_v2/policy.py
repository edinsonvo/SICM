"""Simulación de políticas económicas del SICM v2.0.

Proporciona :func:`simulate_policy` que aplica las cuatro políticas
clásicas (fiscal/monetaria, expansiva/contractiva) sobre un modelo base
(IS-LM o AD-AS) y compara el equilibrio antes y después.
"""

from models import (build_model, get_shock_mechanism, SHOCK_KEYS)

# Política -> (choque equivalente en IS-LM, choque equivalente en AD-AS)
POLICY_SHOCKS = {
    "Fiscal Expansiva": ("Gasto Gobierno ↑", "Oferta Monetaria ↑"),
    "Fiscal Contractiva": ("Gasto Gobierno ↓", "Oferta Monetaria ↓"),
    "Monetaria Expansiva": ("Oferta Monetaria ↑", "Oferta Monetaria ↑"),
    "Monetaria Contractiva": ("Oferta Monetaria ↓", "Oferta Monetaria ↓"),
}

POLICY_NAMES = list(POLICY_SHOCKS.keys())

FISCAL_POLICIES = {"Fiscal Expansiva", "Fiscal Contractiva"}


def simulate_policy(model_name, params, policy_type, magnitude):
    """Simula una política económica sobre el modelo base indicado.

    Parámetros
    ----------
    model_name : str
        "IS-LM" o "AD-AS".
    params : dict
        Parámetros del modelo base.
    policy_type : str
        Una de :data:`POLICY_NAMES`.
    magnitude : float
        Proporción de la magnitud (0.05 = 5 %).

    Devuelve
    --------
    dict
        Con claves: policy, model, magnitude, before, after, deltas,
        shock_label, mechanism y ok (bool).
    """
    result = {
        "policy": policy_type,
        "model": model_name,
        "magnitude": magnitude,
        "before_params": dict(params),
        "before": None,
        "after": None,
        "deltas": {},
        "shock_label": "",
        "mechanism": "",
        "ok": False,
        "message": "",
    }

    if model_name not in ("IS-LM", "AD-AS"):
        result["message"] = "Modelo base no soportado para políticas."
        return result

    # La política fiscal solo existe en el modelo IS-LM
    if policy_type in FISCAL_POLICIES and model_name == "AD-AS":
        result["message"] = (
            "El modelo AD-AS (ecuación cuantitativa Y = M·V/P) no incluye el "
            "gasto público. La política fiscal se puede analizar con el modelo "
            "IS-LM; para AD-AS use políticas monetarias o choques de demanda/oferta."
        )
        return result

    shock_is, shock_ad = POLICY_SHOCKS[policy_type]
    shock_label = shock_is if model_name == "IS-LM" else shock_ad

    model = build_model(model_name, dict(params))
    before = model.solve()

    model_after = model.clone()
    after = model_after.apply_shock(shock_label, magnitude)

    result["before"] = before
    result["after"] = after
    result["shock_label"] = shock_label
    result["mechanism"] = get_shock_mechanism(model_name, shock_label)
    result["ok"] = True
    result["message"] = "Simulación completada."

    # Diferencias relevantes según el modelo
    if model_name == "IS-LM":
        keys = ["Y", "r", "C", "I", "G"]
    else:
        keys = ["Y", "P", "gap", "Yn"]
    result["deltas"] = {
        k: after[k] - before[k] for k in keys if k in before and k in after
    }
    return result


def sensitivity_analysis(model_name, params, policy_type, magnitudes):
    """Analiza la sensibilidad de la política ante varias magnitudes.

    Devuelve listas paralelas ``magnitudes`` y ``results`` (equilibrios
    posteriores) para cada magnitud.
    """
    results = []
    for mag in magnitudes:
        res = simulate_policy(model_name, params, policy_type, mag)
        results.append(res["after"] if res["ok"] else None)
    return list(magnitudes), results


def policy_result_metrics(res):
    """Extrae métricas de comparación (antes/después) de un resultado."""
    if not res or not res["ok"]:
        return None
    before, after, deltas = res["before"], res["after"], res["deltas"]
    model = res["model"]
    if model == "IS-LM":
        rows = [
            ("Producción (Y)", before["Y"], after["Y"], deltas.get("Y", 0.0)),
            ("Tasa de interés (r, %)", before["r"] * 100, after["r"] * 100,
             deltas.get("r", 0.0) * 100),
            ("Consumo (C)", before["C"], after["C"], deltas.get("C", 0.0)),
            ("Inversión (I)", before["I"], after["I"], deltas.get("I", 0.0)),
            ("Gasto público (G)", before["G"], after["G"], deltas.get("G", 0.0)),
        ]
    else:
        rows = [
            ("Producción (Y)", before["Y"], after["Y"], deltas.get("Y", 0.0)),
            ("Nivel de precios (P)", before["P"], after["P"],
             deltas.get("P", 0.0)),
            ("Brecha del producto (% del PIB)", before["gap"], after["gap"],
             deltas.get("gap", 0.0)),
        ]
    return rows
