import numpy as np

import plotly.graph_objects as go

from visualization.base_plot import BasePlot


class MFPlot(BasePlot):

    def create(

        self,

        result,

        config

    ):

        Y = np.linspace(100,700,250)

        IS = (

            config.C0
            + config.G
            + config.NX0

            - Y

        ) / config.b

        LM = (

            config.k*Y
            - config.M/config.P

        ) / config.h

        BP = np.ones_like(Y)*result.r

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

                x=Y,

                y=BP,

                name="BP"
            )
        )

        self.fig.add_trace(

            go.Scatter(

                x=[result.Y],

                y=[result.r],

                mode="markers",

                name="Equilibrio"
            )
        )

        self.configure(

            "Mundell-Fleming",

            "Producción",

            "Interés"
        )

        return self.fig
