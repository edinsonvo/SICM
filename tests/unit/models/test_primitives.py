import pytest
from sicm_core.models import Equation, Parameter, Variable

def test_variable_bounds():
    v = Variable("PIB", "Y", "u.m.", "Producto", lower_bound=0)
    assert v.validate_value(10)
    assert not v.validate_value(-1)

def test_parameter_bounds():
    with pytest.raises(ValueError):
        Parameter("PMC", "c", 1.2, "-", "Propensión", 0, 1).validate()

def test_equation_requires_expression():
    with pytest.raises(ValueError):
        Equation("e", "Vacía", "", ("Y",))
