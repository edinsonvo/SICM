from core.equilibrium import EquilibriumResult
from core.production import CobbDouglas


class ClassicalOpenSolver:

    def solve(self, config):

        N = 100

        Y = CobbDouglas.output(

            config.A,

            config.K,

            N,

            config.alpha
        )

        exchange_rate = (

            1
            - 0.001 * config.NX0
        )

        return EquilibriumResult(

            Y=Y,

            r=5.0,

            inflation=2.0,

            employment=N,

            unemployment=0,

            exchange_rate=exchange_rate,

            nx=config.NX0,

            model="Classical Open",

            notes="Economía abierta clásica."
        )
