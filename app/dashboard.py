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

# Calcular la precisión global basada en el tipo de experimento
if "criteria" in selected_exp_name:
    numeric_criteria_metrics = df[["Coherence", "Correctness", "Harmfulness", "Relevance", "Toxicity"]]
    st.subheader("📊 DataFrame de métricas criteria:")
    st.dataframe(numeric_criteria_metrics)
    valid_criteria_rows = numeric_criteria_metrics[(numeric_criteria_metrics != 0).any(axis=1)]
    st.subheader("📊 DataFrame filtrado (filas no cero):")
    st.dataframe(valid_criteria_rows)
    mean_by_row = valid_criteria_rows.mean(axis=1)
    st.subheader("📊 Promedio por fila:")
    st.write(mean_by_row)
    global_precision = mean_by_row.mean() * 100 if not mean_by_row.empty else 0.0
    st.subheader("✅ Precisión Global Calculada (experimentos criteria):")
    st.write(f"{global_precision:.1f}%")
else:
    global_precision = df["lc_is_correct"].mean() * 100 if not df.empty else 0.0
    st.subheader("✅ Precisión Global Calculada (otros experimentos):")
    st.write(f"{global_precision:.1f}%")

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
