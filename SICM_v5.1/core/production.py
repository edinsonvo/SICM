import math


class CobbDouglas:

    @staticmethod
    def output(
        A,
        K,
        N,
        alpha
    ):

        return (
            A
            * (K ** alpha)
            * (N ** (1 - alpha))
        )
