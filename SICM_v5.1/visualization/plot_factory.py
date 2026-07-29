from visualization.islm_plot import ISLMPlot

from visualization.mf_plot import MFPlot

from visualization.labor_market_plot import LaborMarketPlot

from visualization.production_plot import ProductionPlot


class PlotFactory:

    @staticmethod
    def create(

        model,

        result,

        config

    ):

        if model == "islm":

            return ISLMPlot().create(

                result,

                config
            )

        elif model == "mundell_fleming":

            return MFPlot().create(

                result,

                config
            )

        elif model == "classical_closed":

            return LaborMarketPlot().create(

                result,

                config
            )

        elif model == "classical_open":

            return ProductionPlot().create(

                config
            )

        raise ValueError(
            "Modelo no soportado"
        )
