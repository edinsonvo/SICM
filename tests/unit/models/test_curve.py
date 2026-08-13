import pytest
from sicm_core.models import Curve, CurveSet

def test_curve():
    c = Curve("is", "IS", "Y", "i", [1, 2], [4, 3])
    assert c.to_dict()["name"] == "IS"

def test_curve_lengths():
    with pytest.raises(ValueError):
        Curve("is", "IS", "Y", "i", [1], [2, 3])

def test_curve_ids():
    c = Curve("is", "IS", "Y", "i", [1], [2])
    with pytest.raises(ValueError):
        CurveSet((c, c))
