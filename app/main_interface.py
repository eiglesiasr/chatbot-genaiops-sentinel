# app/main_interface.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="📚 Chatbot GenAI + Métricas", layout="wide")

import pandas as pd
import mlflow
import json
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

import matplotlib.pyplot as plt
import numpy as np

modo = st.sidebar.radio("Selecciona una vista:", ["🤖🚀 Chatbot", "📊 Metrics","📊 Metrics Criteria","📊 Metrics by Experiment"])

vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb)

if modo == "🤖🚀 Chatbot":
    st.title("🤖🚀 Satellite Assistant")
    pregunta = st.text_input("What do you want to know? / ¿Qué deseas consultar? / 何をお知りになりたいですか？ ")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if pregunta:
        with st.spinner("Reading docs..."):
            result = chain.invoke({"question": pregunta, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((pregunta, result["answer"]))

    if st.session_state.chat_history:
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**👤 User:** {q}")
            st.markdown(f"**🤖 Bot:** {a}")
            st.markdown("---")

elif modo == "📊 Metrics":
    st.title("📈 Evaluation Results")

    client = mlflow.tracking.MlflowClient()
    experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

    if not experiments:
        st.warning("Not experiments found.")
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("Select an experiment:", exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.warning("No executions found.")
        st.stop()

    # Armar dataframe
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        data.append({
            "Pregunta": params.get("question"),
            "Prompt": params.get("prompt_version"),
            "Chunk Size": int(params.get("chunk_size", 0)),
            "Correcto (LC)": metrics.get("lc_is_correct", 0)
        })

    df = pd.DataFrame(data)
    st.dataframe(df)

    # Agrupado
    st.subheader("📊 AVG per config")
    grouped = df.groupby(["Prompt", "Chunk Size"]).agg({"Correcto (LC)": "mean"}).reset_index()
    grouped.rename(columns={"Correcto (LC)": "Precisión"}, inplace=True)
    grouped["config"] = grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)
    st.bar_chart(grouped.set_index("config")["Precisión"])



elif modo == "📊 Metrics Criteria":
    st.title("📈 Evaluation Results")

    client = mlflow.tracking.MlflowClient()
    experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

    if not experiments:
        st.warning("Not experiments found.")
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("Select an experiment:", exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.warning("No executions found.")
        st.stop()

    # Armar dataframe
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        data.append({
            "Pregunta": params.get("question"),
            "Prompt": params.get("prompt_version"),
            "Chunk Size": int(params.get("chunk_size", 0)),
            "Coherence": metrics.get("coherence_score", 0),
            "Correctness": metrics.get("correctness_score", 0),
            "Harmfulness": metrics.get("harmfulness_score", 0),
            "Relevance": metrics.get("relevance_score", 0),
            "Toxicity": metrics.get("toxicity_score", 0)})

    df = pd.DataFrame(data)
    st.dataframe(df)

#    # Agrupado
#    st.subheader("📊 AVG per config")
#    grouped = df.groupby(["Prompt", "Chunk Size"]).agg({"Correcto (LC)": "mean"}).reset_index()
#    grouped.rename(columns={"Correcto (LC)": "Precisión"}, inplace=True)
#    grouped["config"] = grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)
#    st.bar_chart(grouped.set_index("config")["Precisión"])


    st.subheader("📊 Stacked Bar by Criterion")
    # Agrupar los datos por Pregunta, calculando el promedio por criterio
    grouped = df.groupby("Pregunta")[["Coherence", "Correctness", "Harmfulness", "Relevance", "Toxicity"]].mean().reset_index()

    # Asignar un índice numérico para cada pregunta
    grouped["Pregunta_idx"] = range(0, len(grouped))

    # Definir criterios y colores para dark mode
    criterios = ["Coherence", "Correctness", "Harmfulness", "Relevance", "Toxicity"]
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # colores oscuros con buen contraste

    # Crear el gráfico apilado con estilo dark
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    fig.patch.set_facecolor('black')
    bottom = np.zeros(len(grouped))

    # Dibujar barras apiladas
    for i, criterio in enumerate(criterios):
        ax.bar(grouped["Pregunta_idx"], grouped[criterio], bottom=bottom, label=criterio, color=colores[i])
        bottom += grouped[criterio]

    # Añadir etiquetas de promedio total arriba de cada barra
    promedios_totales = grouped[criterios].mean(axis=1)
    for i, total in enumerate(promedios_totales):
        ax.text(grouped["Pregunta_idx"][i], bottom[i] + 0.02, f"{total:.2f}", ha="center", va="bottom", fontsize=9, color="white")

    # Estilo del gráfico
    ax.set_facecolor("black")
    ax.set_title("Comparación de criterios por Pregunta (Apilado)", color="white")
    ax.set_ylabel("Puntaje", color="white")
    ax.set_xlabel("Índice de Pregunta", color="white")
    ax.set_xticks(grouped["Pregunta_idx"])
    ax.tick_params(colors="white")
    ax.legend(title="Criterio", facecolor="black", edgecolor="gray", labelcolor="white", title_fontsize="10", fontsize="9")

    plt.tight_layout()
    st.pyplot(fig)


elif modo == "📊 Metrics by Experiment":
    st.title("📈 Evaluation Summary by Experiment")

    client = mlflow.tracking.MlflowClient()
    experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

    if not experiments:
        st.warning("No experiments found.")
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exps = st.multiselect("Select one or more experiments:", exp_names)

    if not selected_exps:
        st.info("Please select at least one experiment.")
        st.stop()

    # Leer y acumular runs de todos los experimentos seleccionados
    all_data = []
    for exp_name in selected_exps:
        experiment = client.get_experiment_by_name(exp_name)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
        for run in runs:
            params = run.data.params
            metrics = run.data.metrics
            all_data.append({
                "Experiment": exp_name,
                "Pregunta": params.get("question"),
                "Prompt": params.get("prompt_version"),
                "Chunk Size": int(params.get("chunk_size", 0)),
                "Coherence": metrics.get("coherence_score", 0),
                "Correctness": metrics.get("correctness_score", 0),
                "Harmfulness": metrics.get("harmfulness_score", 0),
                "Relevance": metrics.get("relevance_score", 0),
                "Toxicity": metrics.get("toxicity_score", 0)
            })

    df = pd.DataFrame(all_data)

    # Agrupar por experimento y calcular promedio
    criterios = ["Coherence", "Correctness", "Harmfulness", "Relevance", "Toxicity"]
    grouped = df.groupby("Experiment")[criterios].mean().reset_index()

    st.subheader("📋 Average Metrics per Experiment")
    st.dataframe(grouped)

    st.subheader("📊 Stacked Bar Chart by Experiment")

    # Dark mode colors
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Crear gráfico apilado
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    fig.patch.set_facecolor('black')
    bottom = np.zeros(len(grouped))

    for i, criterio in enumerate(criterios):
        ax.bar(grouped["Experiment"], grouped[criterio], bottom=bottom, label=criterio, color=colores[i])
        bottom += grouped[criterio]

    # Promedios totales para cada experimento
    promedios_totales = grouped[criterios].mean(axis=1)
    for i, total in enumerate(promedios_totales):
        ax.text(i, bottom[i] + 0.02, f"{total:.2f}", ha="center", va="bottom", fontsize=9, color="white")

    # Estilo del gráfico
    ax.set_facecolor("black")
    ax.set_title("Average Scores by Experiment (Stacked)", color="white")
    ax.set_ylabel("Score", color="white")
    ax.set_xlabel("Experiment", color="white")
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(grouped["Experiment"], rotation=45, ha="right")
    ax.tick_params(colors="white")
    ax.legend(title="Criterion", facecolor="black", edgecolor="gray", labelcolor="white", title_fontsize=10, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)