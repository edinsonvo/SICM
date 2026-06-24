from core.parameters import (
    EconomyConfig
)

def test_default_config():

    config = EconomyConfig()

    assert config.c == 0.80
