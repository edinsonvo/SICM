import pytest
from sicm_core.models import (
    Equilibrium, ModelShock, ShockDirection, ShockTarget,
    TransmissionMechanism, TransmissionStep
)

def test_signed_shock():
    shock = ModelShock("s1", "fiscal", ShockTarget.PARAMETER, 10, ShockDirection.POSITIVE, parameter="G")
    assert shock.signed_magnitude == 10

def test_negative_shock():
    shock = ModelShock("s2", "monetary", ShockTarget.PARAMETER, 5, ShockDirection.NEGATIVE, parameter="M")
    assert shock.signed_magnitude == -5

def test_transmission_order():
    mechanism = TransmissionMechanism("s1", (
        TransmissionStep(2, "i", "interest", "I", "decrease"),
        TransmissionStep(1, "M", "liquidity", "i", "decrease"),
    ))
    assert [s.order for s in mechanism.ordered_steps()] == [1, 2]
    assert mechanism.channels() == ("liquidity", "interest")

def test_parameter_target_requires_parameter():
    with pytest.raises(ValueError):
        ModelShock("s", "fiscal", ShockTarget.PARAMETER, 1, ShockDirection.POSITIVE)
