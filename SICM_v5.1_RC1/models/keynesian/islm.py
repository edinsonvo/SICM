from core.registry import Registry

from models.base_model import BaseModel

from core.equilibrium import EquilibriumResult


@Registry.register("islm")
class ISLMModel(BaseModel):

    name = "IS-LM"

    description = "Modelo Keynesiano Cerrado"

    def solve(self):

        c = self.config.c

        denominator = (

            (1-c)

            + self.config.b

            * self.config.k

            / self.config.h

        )

        Y = (

            self.config.C0

            - c*self.config.T

            + self.config.I0

            + self.config.G

            + (self.config.b/self.config.h)

            * (self.config.M/self.config.P)

        ) / denominator

        r = (

            self.config.k*Y

            - self.config.M/self.config.P

        ) / self.config.h

        return EquilibriumResult(

            Y=Y,

            r=r,

            inflation=2,

            employment=.95*Y,

            unemployment=max(

                0,

                10-Y/100

            ),

            exchange_rate=1,

            nx=0,

            model=self.name

        )

    def interpret(self, result):

        return (

            "El equilibrio aumenta cuando "

            "la demanda agregada se expande."

        )

    def default_plot(self):

        return "islm"
