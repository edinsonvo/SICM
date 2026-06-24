class SimulationMetrics:

    @staticmethod
    def output_gap(
        actual,
        potential
    ):

        return (
            (actual - potential)
            / potential
        ) * 100

    @staticmethod
    def inflation_gap(
        actual,
        target=2.0
    ):

        return (
            actual - target
        )

    @staticmethod
    def unemployment_gap(
        actual,
        natural_rate=5
    ):

        return (
            actual - natural_rate
        )
