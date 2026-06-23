from core.equilibrium import (
    EquilibriumResult
)


class ISLMSolver:

    def solve(self, config):

        config.validate()

        c = config.c

        C0 = config.C0

        T = config.T

        I0 = config.I0

        b = config.b

        G = config.G

        k = config.k

        h = config.h

        M = config.M

        P = config.P

        denominator = (
            (1 - c)
            + (b * k / h)
        )

        Y = (
            C0
            - c * T
            + I0
            + G
            + (b / h) * (M / P)
        ) / denominator

        r = (
            k * Y
            - (M / P)
        ) / h

        employment = 0.95 * Y

        unemployment = max(
            0.0,
            12 - (0.01 * Y)
        )

        inflation = 2.0 + (
            0.005 * max(0, Y - 300)
        )

        return EquilibriumResult(
            Y=Y,
            r=r,
            inflation=inflation,
            employment=employment,
            unemployment=unemployment,
            exchange_rate=1.0,
            nx=config.NX0,
            model="IS-LM",
            notes="Equilibrio calculado con IS-LM."
        )
