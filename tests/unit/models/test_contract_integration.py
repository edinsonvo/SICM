from sicm_core.models import (
    Equation, EquationType, ModelContract, Parameter, Variable, Equilibrium, ModelResult
)

class DemoModel(ModelContract):
    name = "Demo"
    family = "Keynesian"

    def variables(self):
        return (Variable("PIB", "Y", "u.m.", "Producto", lower_bound=0),)

    def parameters(self):
        return (Parameter("Gasto", "G", 20, "u.m.", "Gasto público", lower_bound=0),)

    def equations(self):
        return (Equation("eq1", "Demanda", "Y=C+G", ("Y",), ("G",),
                         EquationType.EQUILIBRIUM),)

    def solve(self, scenario):
        return ModelResult(self.name, self.family,
                           Equilibrium({"Y": 100}), Equilibrium({"Y": 105}))

def test_contract_returns_model_result():
    m = DemoModel()
    m.validate_contract()
    result = m.solve(None)
    assert isinstance(result, ModelResult)
    assert result.delta("Y") == 5
