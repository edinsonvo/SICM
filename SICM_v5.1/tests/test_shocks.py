from core.parameters import (
    EconomyConfig
)

from core.shocks import (
    FiscalShock
)

def test_fiscal_shock():

    config = EconomyConfig()

    shock = FiscalShock(
        delta_g=50
    )

    shock.apply(config)

    assert config.G == 200
