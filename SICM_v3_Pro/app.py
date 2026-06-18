import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SICM v3 Pro", layout="wide")

st.title("📈 SICM v3 Pro - Simulador Interactivo de Choques Macroeconómicos")

st.sidebar.header("Configuración")
modelo = st.sidebar.selectbox("Modelo económico", ["IS-LM", "AD-AS", "Mundell-Fleming"])
choque = st.sidebar.selectbox("Tipo de choque", ["Fiscal", "Monetario", "Oferta", "Externo"])
intensidad = st.sidebar.slider("Intensidad", 0, 100, 30)

col1, col2 = st.columns(2)

x = np.linspace(0, 100, 200)

with col1:
    fig, ax = plt.subplots()
    curva_inicial = 80 - 0.5*x
    curva_final = curva_inicial + intensidad/4
    equilibrio = 50 + intensidad/10

    ax.plot(x, curva_inicial, label="Situación inicial")
    ax.plot(x, curva_final, label="Después del choque")
    ax.legend()
    ax.set_title(modelo)
    st.pyplot(fig)

with col2:
    st.subheader("Indicadores simulados")
    st.metric("PIB", f"{equilibrio:.1f}")
    st.metric("Inflación", f"{2 + intensidad/50:.2f}%")
    st.metric("Tasa de interés", f"{3 + intensidad/40:.2f}%")
    st.metric("Desempleo", f"{8 - intensidad/60:.2f}%")

st.subheader("Interpretación económica")
st.write(f"El choque {choque.lower()} modifica el equilibrio del modelo {modelo}. Esta simulación permite analizar la dirección y magnitud de los efectos macroeconómicos.")
