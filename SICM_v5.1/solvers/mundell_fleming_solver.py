from core.equilibrium import EquilibriumResult


class MundellFlemingSolver:

    def solve(self, config):

        config.validate()

        Y = (
            config.C0
            + config.I0
            + config.G
            + config.NX0
        )

        r = 3.0

        exchange_rate = 1.0

        if config.exchange_rate_regime.value == "flexible":

            exchange_rate = (
                1.0
                - 0.001 * config.NX0
            )

        elif config.exchange_rate_regime.value == "fixed":

            exchange_rate = 1.0

        return EquilibriumResult(

            Y=Y,

            r=r,

            inflation=2.5,

            employment=0.95 * Y,

            unemployment=max(
                0,
                10 - Y / 100
            ),

            exchange_rate=exchange_rate,

            nx=config.NX0,

            model="Mundell-Fleming",

            notes="Economía abierta."
        )
