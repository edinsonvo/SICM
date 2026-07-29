import numpy as np

import plotly.graph_objects as go

from visualization.base_plot import BasePlot


class ISLMPlot(BasePlot):

    def create(

        self,

        result,

        config

    ):

        Y = np.linspace(50,700,300)

        IS = (

            config.C0
            + config.I0
            + config.G
            - Y
        ) / config.b

        LM = (

            config.k*Y
            - config.M/config.P

        ) / config.h

        self.fig.add_trace(

            go.Scatter(

                x=Y,

                y=IS,

                name="IS"
            )
        )

        self.fig.add_trace(

            go.Scatter(

                x=Y,

                y=LM,

                name="LM"
            )
        )

        self.fig.add_trace(

            go.Scatter(

                x=[result.Y],

                y=[result.r],

                mode="markers",

                marker_size=12,

                name="Equilibrio"
            )
        )

        self.configure(

            "Modelo IS-LM",

            "Producción (Y)",

            "Tasa de interés"
        )

        return self.fig
