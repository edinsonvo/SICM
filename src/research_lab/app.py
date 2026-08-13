from __future__ import annotations

import streamlit as st

from sicm_core import __version__

st.set_page_config(

    page_title="SICM Research Lab",

    layout="wide"

)

st.title("SICM Research Lab")

st.info(

    f"SICM Core version {__version__}"

)

st.write(

    "Infrastructure initialized successfully."

)

st.divider()

st.caption(

    "© Edinson Patrocinio Valencia • edvalenciao@unal.edu.co"

)
