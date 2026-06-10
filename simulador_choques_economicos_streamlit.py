
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Choques Económicos", layout="wide")

st.title("Simulador de Choques Económicos")
st.markdown("""
Analiza el impacto de choques económicos en:
- Economía cerrada (IS-LM)
- Economía abierta (Mundell-Fleming)
- Tipo de cambio fijo o flexible
- Choques fiscales, monetarios, externos y de oferta
""")

modelo = st.sidebar.selectbox(
    "Modelo",
    ["IS-LM (Economía cerrada)", "Mundell-Fleming (Economía abierta)", "AD-AS"]
)

choque = st.sidebar.selectbox(
    "Choque",
    ["Fiscal expansivo", "Fiscal contractivo",
     "Monetario expansivo", "Monetario contractivo",
     "Choque externo", "Choque de oferta"]
)

magnitud = st.sidebar.slider("Magnitud", 0.1, 5.0, 1.0, 0.1)

fig, ax = plt.subplots(figsize=(7,5))

if modelo == "IS-LM (Economía cerrada)":
    y = np.linspace(0,10,100)
    is0 = 8 - 0.5*y
    lm0 = 2 + 0.4*y

    is1 = is0.copy()
    lm1 = lm0.copy()

    if "Fiscal expansivo" in choque:
        is1 += magnitud
        mecanismo = "Aumento del gasto público → desplazamiento IS a la derecha."
    elif "Fiscal contractivo" in choque:
        is1 -= magnitud
        mecanismo = "Reducción del gasto público → desplazamiento IS a la izquierda."
    elif "Monetario expansivo" in choque:
        lm1 -= magnitud
        mecanismo = "Mayor oferta monetaria → LM a la derecha."
    elif "Monetario contractivo" in choque:
        lm1 += magnitud
        mecanismo = "Menor oferta monetaria → LM a la izquierda."
    else:
        mecanismo = "Choque no estándar para IS-LM."

    ax.plot(y,is0,label="IS inicial")
    ax.plot(y,lm0,label="LM inicial")
    ax.plot(y,is1,'--',label="IS nueva")
    ax.plot(y,lm1,'--',label="LM nueva")

elif modelo == "Mundell-Fleming (Economía abierta)":
    y = np.linspace(0,10,100)
    is0 = 8 - 0.5*y
    bp0 = np.full_like(y,4)
    is1 = is0.copy()

    if "Fiscal expansivo" in choque:
        is1 += magnitud
        mecanismo = "Mayor demanda agregada; efecto depende del régimen cambiario."
    elif "Monetario expansivo" in choque:
        is1 += magnitud/2
        mecanismo = "Baja tasas de interés, salida de capitales y depreciación bajo TC flexible."
    elif "Choque externo" in choque:
        is1 += magnitud
        mecanismo = "Cambio en exportaciones netas."
    else:
        mecanismo = "Ajuste vía cuenta externa y movilidad de capitales."

    ax.plot(y,is0,label="IS inicial")
    ax.plot(y,is1,'--',label="IS nueva")
    ax.plot(y,bp0,label="BP")

else:
    x = np.linspace(0,10,100)
    ad0 = 8 - 0.5*x
    as0 = 2 + 0.4*x

    ad1 = ad0.copy()
    as1 = as0.copy()

    if "Fiscal" in choque or "Monetario" in choque:
        ad1 += magnitud
        mecanismo = "Desplazamiento de demanda agregada."
    elif "oferta" in choque.lower():
        as1 += magnitud
        mecanismo = "Choque de oferta desplaza AS."
    else:
        ad1 += magnitud/2
        mecanismo = "Impacto sobre demanda agregada."

    ax.plot(x,ad0,label="AD inicial")
    ax.plot(x,as0,label="AS inicial")
    ax.plot(x,ad1,'--',label="AD nueva")
    ax.plot(x,as1,'--',label="AS nueva")

ax.set_title("Curvas antes y después del choque")
ax.legend()
ax.grid(True)

col1, col2 = st.columns([2,1])

with col1:
    st.pyplot(fig)

with col2:
    st.subheader("Mecanismo de transmisión")
    st.write(mecanismo)

    st.subheader("Interpretación")
    st.write("""
    Observe cómo cambian las curvas antes y después del choque.
    El desplazamiento representa el mecanismo de transmisión del modelo.
    """)
