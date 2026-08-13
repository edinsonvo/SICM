import pytest
from sicm_core.models import (
    Equilibrium, ModelResult, ModelShock, ShockDirection, ShockTarget,
    TransmissionMechanism, TransmissionStep, build_result
)

def shock():
    return ModelShock("s1", "fiscal", ShockTarget.PARAMETER, 10,
                      ShockDirection.POSITIVE, parameter="G")

def test_model_result_delta_and_serialization():
    r = ModelResult("Demo", "Keynesian",
                    Equilibrium({"Y": 100, "i": 5}),
                    Equilibrium({"Y": 110, "i": 6}))
    assert r.delta("Y") == 10
    assert r.deltas() == {"Y": 10, "i": 1}
    assert not r.shocked
    assert r.to_dict()["model_name"] == "Demo"

def test_build_result_requires_matching_transmission():
    s = shock()
    t = TransmissionMechanism("other", (
        TransmissionStep(1, "G", "fiscal", "Y", "increase"),
    ))
    with pytest.raises(ValueError):
        build_result("Demo", "Keynesian", Equilibrium({"Y": 100}),
                     Equilibrium({"Y": 110}), shock=s, transmission=t)

def test_build_result_shocked():
    s = shock()
    t = TransmissionMechanism("s1", ())
    r = build_result("Demo", "Keynesian", Equilibrium({"Y": 100}),
                     Equilibrium({"Y": 110}), shock=s, transmission=t)
    assert r.shocked
