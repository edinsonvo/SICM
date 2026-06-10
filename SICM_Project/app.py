
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="SICM", layout="wide")

st.title("SICM - Simulador Integral de Choques Macroeconómicos")

economia = st.sidebar.selectbox("Economía", ["Cerrada","Abierta"])
regimen = st.sidebar.selectbox("Tipo de cambio", ["Flexible","Fijo"])
modelo = st.sidebar.selectbox("Modelo", ["IS-LM","Mundell-Fleming","AD-AS"])
choque = st.sidebar.selectbox("Choque", ["Fiscal","Monetario","Oferta","Externo"])
magnitud = st.sidebar.slider("Magnitud",0.0,5.0,1.0)

x = np.linspace(0,100,200)

fig = go.Figure()

if modelo=="IS-LM":
    is0 = 80 - 0.5*x
    lm0 = 20 + 0.4*x
    is1 = is0 + (magnitud*5 if choque=="Fiscal" else 0)
    lm1 = lm0 + (magnitud*5 if choque=="Monetario" else 0)

    fig.add_scatter(x=x,y=is0,name="IS Inicial")
    fig.add_scatter(x=x,y=lm0,name="LM Inicial")
    fig.add_scatter(x=x,y=is1,name="IS Final")
    fig.add_scatter(x=x,y=lm1,name="LM Final")

elif modelo=="AD-AS":
    ad0 = 100 - 0.5*x
    as0 = 20 + 0.4*x
    ad1 = ad0 + magnitud*5
    as1 = as0 + (magnitud*5 if choque=="Oferta" else 0)

    fig.add_scatter(x=x,y=ad0,name="AD Inicial")
    fig.add_scatter(x=x,y=as0,name="AS Inicial")
    fig.add_scatter(x=x,y=ad1,name="AD Final")
    fig.add_scatter(x=x,y=as1,name="AS Final")

else:
    is0 = 80 - 0.5*x
    bp = np.repeat(40,len(x))
    is1 = is0 + magnitud*5

    fig.add_scatter(x=x,y=is0,name="IS Inicial")
    fig.add_scatter(x=x,y=is1,name="IS Final")
    fig.add_scatter(x=x,y=bp,name="BP")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Mecanismo de transmisión")
st.write(f"Modelo: {modelo} | Economía: {economia} | Régimen: {regimen} | Choque: {choque}")
