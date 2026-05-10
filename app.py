import os
# PARCHE CRÍTICO: Debe ir antes de cualquier otro import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import streamlit.components.v1 as components
import folium
import datetime
import pandas as pd
import joblib

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Santander Bike AI", layout="wide")
st.title("🚲 Santander E-Bike AI Predictor")

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    if os.path.exists('modelo_bicis_santander.joblib'):
        return joblib.load('modelo_bicis_santander.joblib')
    return None

modelo = cargar_modelo()

# --- BARRA LATERAL ---
st.sidebar.header("Parámetros de Predicción")

hoy = datetime.date.today()
manana = hoy + datetime.timedelta(days=1)
fecha_elegida = st.sidebar.date_input("Día:", value=manana, min_value=manana, max_value=hoy + datetime.timedelta(days=7))
hora_elegida = st.sidebar.slider("Hora:", 6, 23, 12)
clima = st.sidebar.selectbox("Clima:", ["Soleado (25°C)", "Nublado (15°C)", "Lluvia (10°C)"])

# Traducción de clima a valores numéricos
temp, precip = (25.0, 0.0) if "Soleado" in clima else (15.0, 0.0) if "Nublado" in clima else (10.0, 15.0)

# --- DATOS Y LÓGICA ---
col1, col2 = st.columns([3, 1])

estaciones = [
    {"name": "Universidad", "lat": 43.4757, "lon": -3.7980, "cap": 20},
    {"name": "Ayuntamiento", "lat": 43.4627, "lon": -3.7968, "cap": 15},
    {"name": "Sardinero", "lat": 43.4725, "lon": -3.7830, "cap": 25},
    {"name": "Puerto Chico", "lat": 43.4615, "lon": -3.7995, "cap": 18}
]

with col1:
    st.subheader(f"Predicción: {fecha_elegida} | {hora_elegida}:00h")
    m = folium.Map(location=[43.4647, -3.8044], zoom_start=14, tiles='CartoDB dark_matter')
    
    total_bicis = 0
    total_cap = 0

    for est in estaciones:
        if modelo:
            df_input = pd.DataFrame([{
                'hour': hora_elegida,
                'day_of_week': fecha_elegida.weekday(),
                'is_weekend': 1 if fecha_elegida.weekday() >= 5 else 0,
                'temperature_c': temp,
                'precipitation_mm': precip,
                'capacity': est["cap"]
            }])
            try:
                pred = max(0.0, min(1.0, modelo.predict(df_input)[0]))
                disponibles = int(round(pred * est["cap"]))
            except:
                disponibles, pred = (est["cap"] // 2, 0.5)
        else:
            disponibles, pred = (est["cap"] // 2, 0.5)

        total_bicis += disponibles
        total_cap += est["cap"]
        color = "red" if pred < 0.2 else "orange" if pred < 0.6 else "green"

        folium.CircleMarker(
            [est["lat"], est["lon"]], radius=12, color=color, fill=True, fill_opacity=0.7,
            popup=f"{est['name']}: {disponibles}/{est['cap']}"
        ).add_to(m)

    # RENDERIZADO SEGURO: HTML puro para evitar bloqueos de JS
    components.html(m._repr_html_(), height=500)

with col2:
    st.subheader("Estado de la Red")
    if modelo is None:
        st.warning("Usando modo simulación (falta .joblib)")
    
    st.metric("Ocupación Total", f"{total_bicis} / {total_cap}")
    st.metric("Confianza", f"{95 - (fecha_elegida - hoy).days * 3}%")
    st.write(f"**Condiciones:** {temp}°C, {precip}mm lluvia")