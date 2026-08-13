"""Public convenience factories."""
from __future__ import annotations
from sicm_core.config.enums import ModelType
from sicm_core.domain.experiment import Experiment
from sicm_core.domain.metadata import Metadata
from sicm_core.domain.parameters import EconomyParameters
from sicm_core.domain.scenario import Scenario

def create_experiment(name: str, model: ModelType, author: str, institution: str = "", parameters: EconomyParameters | None = None) -> Experiment:
    params = parameters or EconomyParameters()
    return Experiment(name=name, scenario=Scenario(model=model, parameters=params), metadata=Metadata(author=author, institution=institution))
