# app/dashboard.py
import mlflow
import streamlit as st
from pathlib import Path

# Configuración infalible para tu estructura
mlflow.set_tracking_uri("file:///" + str(Path("mlruns").absolute()))  # Ruta absoluta a mlruns
client = mlflow.tracking.MlflowClient()

# Verificación de conexión
st.set_page_config(layout="wide")
st.title("📊 Dashboard de Evaluación")

# 1. Obtener TODOS los experimentos (sin filtrar por nombre)
all_experiments = client.search_experiments()
st.write("### Experimentos detectados:", [f"ID: {exp.experiment_id} - Nombre: {exp.name}" for exp in all_experiments])

# 2. Usar el PRIMER experimento (ajusta el índice si es necesario)
if not all_experiments:
    st.error("❌ No hay experimentos en mlruns/")
else:
    selected_exp = all_experiments[0]  # Primer experimento (puedes cambiarlo)
    runs = client.search_runs(experiment_ids=[selected_exp.experiment_id])
    
    st.success(f"✅ Usando experimento: ID={selected_exp.experiment_id}, Nombre='{selected_exp.name}'")
    
    # 3. Mostrar métricas (ejemplo básico)
    if runs:
        st.write("### Métricas de las ejecuciones:")
        for i, run in enumerate(runs[:5]):  # Muestra las primeras 5 ejecuciones
            with st.expander(f"Ejecución {i+1} - Run ID: {run.info.run_id}"):
                st.write("**Parámetros:**", run.data.params)
                st.write("**Métricas:**", run.data.metrics)
    else:
        st.warning("⚠️ No hay ejecuciones en este experimento")
