from .equilibrium import Equilibrium
from .result import ModelResult
from .curve_set import CurveSet
from .shock import ModelShock
from .transmission import TransmissionMechanism

def build_result(
    model_name: str,
    model_family: str,
    initial_equilibrium: Equilibrium,
    final_equilibrium: Equilibrium,
    *,
    curves_before: CurveSet = CurveSet(),
    curves_after: CurveSet = CurveSet(),
    shock: ModelShock | None = None,
    transmission: TransmissionMechanism | None = None,
    interpretation: str = "",
    diagnostics: tuple[str, ...] = (),
) -> ModelResult:
    if shock is None and transmission is not None:
        raise ValueError("Transmission cannot exist without a shock.")
    if shock is not None and transmission is not None and transmission.shock_id != shock.shock_id:
        raise ValueError("Shock and transmission IDs must match.")
    return ModelResult(
        model_name=model_name,
        model_family=model_family,
        initial_equilibrium=initial_equilibrium,
        final_equilibrium=final_equilibrium,
        curves_before=curves_before,
        curves_after=curves_after,
        shock=shock,
        transmission=transmission,
        interpretation=interpretation,
        diagnostics=diagnostics,
    )
