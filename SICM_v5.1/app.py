import streamlit as st

from ui.sidebar import render_sidebar

from ui.tabs import create_tabs

from ui.layout import (
    render_metrics,
    render_summary
)

from core.api import SICMEngine

from core.footer import (
    render_footer
)


st.set_page_config(
    page_title="SICM v5.1 Research Lab",
    layout="wide"
)

st.title(
    "SICM v5.1 Research Lab"
)

(
    model,
    shock_type,
    shock_size,
    config
) = render_sidebar()

engine = SICMEngine()

tabs = create_tabs()

if st.button("Simular"):

    result = engine.run(
        model,
        config
    )

    with tabs[0]:

        render_metrics(result)

        render_summary(result)

    with tabs[1]:

        st.info(
            "Gráfica disponible en Entrega 2.2"
        )

    with tabs[2]:

        st.info(
            "Comparador disponible en Sprint 3"
        )

render_footer()

from visualization.plot_factory import PlotFactory

    ...
    
    with tabs[1]:
    
        figure = PlotFactory.create(
    
            model,
    
            result,
    
            config
        )
    
        st.plotly_chart(
    
            figure,
    
            use_container_width=True
        )
