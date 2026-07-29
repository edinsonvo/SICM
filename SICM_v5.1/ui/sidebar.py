import streamlit as st

from core.parameters import (
    EconomyConfig,
    EconomyType,
    ExchangeRateRegime,
    CapitalMobility
)


def render_sidebar():

    st.sidebar.title("Configuración")

    model = st.sidebar.selectbox(
        "Modelo",
        [
            "islm",
            "mundell_fleming",
            "classical_closed",
            "classical_open"
        ]
    )

    shock_type = st.sidebar.selectbox(
        "Choque",
        [
            "none",
            "fiscal",
            "monetary",
            "supply",
            "external"
        ]
    )

    shock_size = st.sidebar.slider(
        "Magnitud del choque",
        0,
        100,
        25
    )

    st.sidebar.subheader("Consumo")

    C0 = st.sidebar.number_input(
        "C0",
        value=50.0
    )

    c = st.sidebar.slider(
        "c",
        0.1,
        0.99,
        0.80
    )

    st.sidebar.subheader("Inversión")

    I0 = st.sidebar.number_input(
        "I0",
        value=100.0
    )

    b = st.sidebar.number_input(
        "b",
        value=5.0
    )

    st.sidebar.subheader("Gobierno")

    G = st.sidebar.number_input(
        "G",
        value=150.0
    )

    T = st.sidebar.number_input(
        "T",
        value=120.0
    )

    st.sidebar.subheader("Mercado Monetario")

    M = st.sidebar.number_input(
        "M",
        value=500.0
    )

    P = st.sidebar.number_input(
        "P",
        value=1.0
    )

    k = st.sidebar.number_input(
        "k",
        value=0.5
    )

    h = st.sidebar.number_input(
        "h",
        value=10.0
    )

    economy_type = (
        EconomyType.CLOSED
        if model in ["islm", "classical_closed"]
        else EconomyType.OPEN
    )

    config = EconomyConfig(

        economy_type=economy_type,

        exchange_rate_regime=
        ExchangeRateRegime.FLEXIBLE,

        capital_mobility=
        CapitalMobility.MEDIUM,

        C0=C0,
        c=c,

        I0=I0,
        b=b,

        G=G,
        T=T,

        M=M,
        P=P,

        k=k,
        h=h
    )

    return (
        model,
        shock_type,
        shock_size,
        config
    )
