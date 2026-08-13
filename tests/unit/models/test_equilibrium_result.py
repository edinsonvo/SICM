from sicm_core.models import Curve, CurveSet, Equilibrium, EquilibriumResult

def test_result():
    c = Curve("is", "IS", "Y", "i", [1], [2])
    r = EquilibriumResult(Equilibrium({"Y": 100}), CurveSet((c,)))
    assert r.to_dict()["curves"][0]["curve_id"] == "is"
