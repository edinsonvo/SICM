import streamlit as st

from core.parameters import (
    EconomyConfig
)

from core.api import (
    SICMEngine
)

from core.footer import (
    render_footer
)

st.set_page_config(
    layout="wide"
)

st.title(
    "SICM v5.1 Research Lab"
)

model = st.selectbox(

    "Modelo",

    [
        "islm",
        "mundell_fleming",
        "classical_closed",
        "classical_open"
    ]
)

config = EconomyConfig()

engine = SICMEngine()

if st.button(
    "Simular"
):

    result = (
        engine.run(
            model,
            config
        )
    )

    st.write(result)

render_footer()
