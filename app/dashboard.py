# app/dashboard.py

import mlflow
import pandas as pd
import streamlit as st

# Configuración de página (original)
st.set_page_config(page_title="📊 Dashboard Unificado", layout="wide")
st.title("🚀 Evaluación Integral del Chatbot")

# --- Datos desde MLflow (código original sin cambios) ---
client = mlflow.tracking.MlflowClient()
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

if not experiments:
    st.warning("No se encontraron experimentos de evaluación.")
    st.stop()

exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento:", exp_names)

experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

if not runs:
    st.warning("No hay ejecuciones registradas en este experimento.")
    st.stop()

# Procesamiento de datos (original)
data = []
for run in runs:
    params = run.data.params
    metrics = run.data.metrics
    data.append({
        "pregunta": params.get("question"),
        "prompt_version": params.get("prompt_version"),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0)),
        "lc_is_correct": metrics.get("lc_is_correct", 0),
        "Coherence": metrics.get("coherence_score", 0),
        "Correctness": metrics.get("correctness_score", 0),
        "Harmfulness": metrics.get("harmfulness_score", 0),
        "Relevance": metrics.get("relevance_score", 0),
        "Toxicity": metrics.get("toxicity_score", 0)
    })

df = pd.DataFrame(data)

# --- Métricas tradicionales añadidas aquí ---
st.header("📊 Métricas Clave")
col1, col2 = st.columns(2)
with col1:
    precision_global = df["lc_is_correct"].mean() * 100
    st.metric("Precisión Global", f"{precision_global:.1f}%")
with col2:
    # Si tienes métrica de latencia, reemplaza esto:
    st.metric("Respuestas Evaluadas", len(df))

# --- Tabla y análisis original (sin cambios) ---
st.header("📋 Resultados por Pregunta")
st.dataframe(df)

# Agrupación original (sin cambios)
grouped = df.groupby(["prompt_version", "chunk_size"]).agg(
    promedio_correcto=("lc_is_correct", "mean"),
    promedio_Coherence=("Coherence", "mean"),
    promedio_Correctness=("Correctness", "mean"),
    promedio_Harmfulness=("Harmfulness", "mean"),
    promedio_Relevance=("Relevance", "mean"),
    promedio_Toxicity=("Toxicity", "mean"),
    preguntas=("pregunta", "count")
).reset_index()

st.header("📈 Resumen Agrupado")
st.dataframe(grouped)

# Gráfico original (sin cambios)
grouped["config"] = grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)
st.bar_chart(grouped.set_index("config")[[
    "promedio_correcto", 
    "promedio_Coherence",
    "promedio_Correctness"
]])

try:
    st.header("📌 Evidencias Experimentales")
    st.image("evidencias/comparativa_experimentos_chunk_size.png")
    st.image("evidencias/comparativa_tipo_prompt.png")
except FileNotFoundError:
    st.warning("No se encontraron imágenes de evidencias")
