from abc import ABC

from abc import abstractmethod


class BaseModel(ABC):

    def __init__(

        self,

        config

    ):

        self.config = config

    @abstractmethod
    def solve(self):

        ...

    @abstractmethod
    def interpret(self):

        ...

    @abstractmethod
    def metrics(self):

        ...
