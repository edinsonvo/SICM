from dataclasses import dataclass


@dataclass
class PlotStyle:

    width = 800

    height = 600

    template = "plotly_white"

    equilibrium_color = "red"

    curve_width = 3

    font_size = 15

    title_size = 20
