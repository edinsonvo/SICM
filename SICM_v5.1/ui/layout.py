import streamlit as st


def render_metrics(result):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PIB",
        round(result.Y, 2)
    )

    col2.metric(
        "Inflación",
        round(result.inflation, 2)
    )

    col3.metric(
        "Interés",
        round(result.r, 2)
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Empleo",
        round(result.employment, 2)
    )

    col5.metric(
        "Desempleo",
        round(result.unemployment, 2)
    )

    col6.metric(
        "Tipo de Cambio",
        round(result.exchange_rate, 2)
    )


def render_summary(result):

    st.subheader("Resultado")

    st.json(
        result.__dict__
    )
