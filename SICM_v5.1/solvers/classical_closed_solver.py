from core.equilibrium import EquilibriumResult
from core.production import CobbDouglas


class ClassicalClosedSolver:

    def solve(self, config):

        N = 100

        Y = CobbDouglas.output(

            config.A,

            config.K,

            N,

            config.alpha
        )

        r = 5.0

        return EquilibriumResult(

            Y=Y,

            r=r,

            inflation=2.0,

            employment=N,

            unemployment=0,

            exchange_rate=1.0,

            nx=0,

            model="Classical Closed",

            notes="Producción determinada por oferta."
        )
