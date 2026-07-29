import streamlit as st

from core.engine import Engine

from ui.sidebar import sidebar

from ui.dashboard import dashboard

from ui.footer import footer


st.set_page_config(

    layout="wide"

)

engine = Engine()

model, config = sidebar()

if st.button(

    "Simular"

):

    result = engine.run(

        model,

        config

    )

    dashboard(result)

footer()
