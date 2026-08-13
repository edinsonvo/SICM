from sicm_core.models import (
    Equilibrium, ModelShock, ShockDirection, ShockResult, ShockTarget,
    TransmissionMechanism
)

def test_shock_result_delta():
    shock = ModelShock("s1", "fiscal", ShockTarget.PARAMETER, 10, ShockDirection.POSITIVE, parameter="G")
    initial = Equilibrium({"Y": 100})
    final = Equilibrium({"Y": 110})
    result = ShockResult(shock, initial, final, TransmissionMechanism("s1", ()))
    assert result.delta("Y") == 10
    assert result.delta("i") is None
