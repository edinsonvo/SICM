from abc import ABC, abstractmethod


class Shock(ABC):

    @abstractmethod
    def apply(self, config):
        pass


class FiscalShock(Shock):

    def __init__(
        self,
        delta_g: float = 0.0,
        delta_t: float = 0.0
    ):
        self.delta_g = delta_g
        self.delta_t = delta_t

    def apply(self, config):

        config.G += self.delta_g
        config.T += self.delta_t

        return config


class MonetaryShock(Shock):

    def __init__(
        self,
        delta_m: float = 0.0
    ):
        self.delta_m = delta_m

    def apply(self, config):

        config.M += self.delta_m

        return config


class SupplyShock(Shock):

    def __init__(
        self,
        delta_a: float = 0.0
    ):
        self.delta_a = delta_a

    def apply(self, config):

        config.A += self.delta_a

        return config


class ExternalShock(Shock):

    def __init__(
        self,
        delta_nx: float = 0.0
    ):
        self.delta_nx = delta_nx

    def apply(self, config):

        config.NX0 += self.delta_nx

        return config
