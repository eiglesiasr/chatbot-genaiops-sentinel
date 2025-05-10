# app/dashboard.py
# app/dashboard.py
import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("📈 Evaluación Completa del Chatbot por Pregunta")

# ✅ Buscar todos los experimentos que comienzan con "eval_"
client = mlflow.tracking.MlflowClient()
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

if not experiments:
    st.warning("No se encontraron experimentos de evaluación.")
    st.stop()

# Mostrar opciones
exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento para visualizar:", exp_names)

experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

if not runs:
    st.warning("No hay ejecuciones registradas en este experimento.")
    st.stop()

# Convertir runs a DataFrame
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

# --- Agregar estas líneas para depuración ---
st.subheader("🔍 Inspección del DataFrame")
st.dataframe(df)
st.subheader("📊 Tipos de datos del DataFrame")
st.write(df.dtypes)
st.subheader("📊 Valores nulos por columna")
st.write(df.isnull().sum())
# --- Fin de las líneas de depuración ---

# Calcular la precisión global basada en el tipo de experimento
print(f"Nombre del experimento seleccionado: {selected_exp_name}")

if "criteria" in selected_exp_name:
    numeric_criteria_metrics = df[["Coherence", "Correctness", "Harmfulness", "Relevance", "Toxicity"]]
    # Filtrar las filas donde al menos una de las métricas criteria no sea cero
    valid_criteria_rows = numeric_criteria_metrics[(numeric_criteria_metrics != 0).any(axis=1)]
    global_precision = valid_criteria_rows.mean(axis=1).mean() * 100 if not valid_criteria_rows.empty else 0.0
else:
    global_precision = df["lc_is_correct"].mean() * 100 if not df.empty else 0.0

# Mostrar la precisión global y el número de respuestas evaluadas
st.subheader("📊 Métricas Clave")
col1, col2 = st.columns(2)
col1.metric("Precisión Global", f"{global_precision:.1f}%")
col2.metric("Respuestas Evaluadas", len(df))

# Mostrar tabla completa
st.subheader("📋 Resultados individuales por pregunta")
st.dataframe(df)

# Agrupación para análisis
grouped = df.groupby(["prompt_version", "chunk_size"]).agg(
    promedio_correcto=("lc_is_correct", "mean"),
    promedio_Coherence=("Coherence", "mean"),
    promedio_Correctness=("Correctness", "mean"),
    promedio_Harmfulness=("Harmfulness", "mean"),
    promedio_Relevance=("Relevance", "mean"),
    promedio_Toxicity=("Toxicity", "mean"),
    preguntas=("pregunta", "count")
).reset_index()

st.subheader("📊 Resumen del desempeño")
st.dataframe(grouped)

# Gráfico
grouped["config"] = grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)
st.bar_chart(grouped.set_index("config")[["promedio_correcto",
                                            "promedio_Coherence",
                                            "promedio_Correctness",
                                            "promedio_Harmfulness",
                                            "promedio_Relevance",
                                            "promedio_Toxicity"
                                            ]])
