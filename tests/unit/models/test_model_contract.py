from sicm_core.models import Equation, EquationType, ModelContract, Parameter, Variable

class DemoModel(ModelContract):
    name = "demo"
    family = "test"
    def variables(self):
        return (Variable("Producto", "Y", "u.m.", "PIB", lower_bound=0),)
    def parameters(self):
        return (Parameter("PMC", "c", .8, "-", "Propensión", 0, 1),)
    def equations(self):
        return (Equation("eq1", "Consumo", "C=cY", ("Y",), ("c",), EquationType.BEHAVIORAL),)

def test_contract():
    model = DemoModel()
    model.validate_contract()
    assert model.variable_map()["Y"].name == "Producto"
    assert model.parameter_map()["c"].value == .8
    assert model.equation_map()["eq1"].name == "Consumo"
