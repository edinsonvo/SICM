import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SICM v2")
st.title("SICM v2 - Simulador Interactivo de Choques Macroeconómicos")

modelo = st.selectbox("Modelo", ["IS-LM", "AD-AS", "Mundell-Fleming"])
choque = st.slider("Intensidad del choque", 0, 100, 25)

x = np.linspace(0,100,100)
y1 = 100 - 0.5*x + choque/5
y2 = 20 + 0.4*x

fig, ax = plt.subplots()
ax.plot(x,y1, label="Curva 1")
ax.plot(x,y2, label="Curva 2")
ax.legend()
st.pyplot(fig)
