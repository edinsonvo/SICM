import numpy as np

import plotly.graph_objects as go

from visualization.base_plot import BasePlot


class LaborMarketPlot(BasePlot):

    def create(

        self,

        result,

        config

    ):

        wage = np.linspace(

            1,

            20,

            200
        )

        labor_supply = (

            40
            + 5*wage
        )

        labor_demand = (

            150
            - 4*wage
        )

        self.fig.add_trace(

            go.Scatter(

                x=labor_supply,

                y=wage,

                name="Oferta"
            )
        )

        self.fig.add_trace(

            go.Scatter(

                x=labor_demand,

                y=wage,

                name="Demanda"
            )
        )

        self.configure(

            "Mercado Laboral",

            "Trabajo",

            "Salario Real"
        )

        return self.fig
