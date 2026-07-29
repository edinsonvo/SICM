PLOTS = {

    "islm":ISLMPlot,

    "mf":MFPlot,

    "labor":LaborPlot,

    "production":ProductionPlot

}
plot = PLOTS[
    model.default_plot()
]
