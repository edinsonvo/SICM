import numpy as np

import plotly.graph_objects as go

from visualization.base_plot import BasePlot


class ProductionPlot(BasePlot):

    def create(

        self,

        config

    ):

        labor = np.linspace(

            1,

            150,

            150
        )

        output = (

            config.A

            * config.K**config.alpha

            * labor**(1-config.alpha)
        )

        self.fig.add_trace(

            go.Scatter(

                x=labor,

                y=output,

                name="Producción"
            )
        )

        self.configure(

            "Función de Producción",

            "Trabajo",

            "Producto"
        )

        return self.fig
