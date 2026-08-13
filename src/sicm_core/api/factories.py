"""
Factories for public domain objects.
"""

from __future__ import annotations

from sicm_core.config.enums import ModelType
from sicm_core.domain.experiment import Experiment
from sicm_core.domain.metadata import Metadata
from sicm_core.domain.parameters import EconomyParameters
from sicm_core.domain.scenario import Scenario


def create_experiment(
    name: str,
    model: ModelType,
    author: str,
    institution: str = "",
    parameters: EconomyParameters | None = None,
) -> Experiment:
    """
    Create a basic experiment.

    Args:
        name: Experiment name.
        model: Economic model.
        author: Experiment author.
        institution: Institution.
        parameters: Economic parameters.

    Returns:
        Configured experiment.
    """

    if parameters is None:
        parameters = EconomyParameters()

    scenario = Scenario(
        model=model,
        parameters=parameters,
    )

    metadata = Metadata(
        author=author,
        institution=institution,
    )

    return Experiment(
        name=name,
        scenario=scenario,
        metadata=metadata,
    )
