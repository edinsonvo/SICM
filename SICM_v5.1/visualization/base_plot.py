from abc import ABC
import plotly.graph_objects as go

from visualization.styles import PlotStyle


class BasePlot(ABC):

    def __init__(self):

        self.fig = go.Figure()

        self.style = PlotStyle()

    def configure(self, title, x_label, y_label):

        self.fig.update_layout(

            title=title,

            width=self.style.width,

            height=self.style.height,

            template=self.style.template,

            xaxis_title=x_label,

            yaxis_title=y_label,

            font_size=self.style.font_size
        )

        return self.fig
