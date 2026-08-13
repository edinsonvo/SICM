from sicm_core.models import Equilibrium

def test_equilibrium():
    eq = Equilibrium({"Y": 100.0, "i": 5.0}, {"eq1": .001})
    assert eq.value("Y") == 100.0
    assert eq.max_residual() == .001
    assert eq.converged
