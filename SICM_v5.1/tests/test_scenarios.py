from core.scenarios import (
    ScenarioManager
)

def test_manager():

    manager = (
        ScenarioManager()
    )

    assert (
        len(
            manager.list_scenarios()
        )
        == 0
    )
