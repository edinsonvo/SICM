from abc import ABC
from abc import abstractmethod

from core.equilibrium import EquilibriumResult


class BaseModel(ABC):

    name = ""

    description = ""

    def __init__(self, config):

        self.config = config

    @abstractmethod
    def solve(self) -> EquilibriumResult:
        ...

    @abstractmethod
    def interpret(self, result):

        ...

    @abstractmethod
    def default_plot(self):

        ...

    def metrics(self, result):

        return result.__dict__
