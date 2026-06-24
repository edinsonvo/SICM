from core.parameters import (
    EconomyConfig
)

from solvers.islm_solver import (
    ISLMSolver
)

def test_islm_solver():

    config = EconomyConfig()

    result = (
        ISLMSolver()
        .solve(config)
    )

    assert result.Y > 0
