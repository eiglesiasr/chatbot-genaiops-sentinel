# app/dashboard.py
import mlflow
import streamlit as st

st.set_page_config(layout="wide")
st.title("Debug: MLflow Experiments")

# Configura MLflow (ajusta la URI si es necesario)
mlflow.set_tracking_uri("http://localhost:5000")  # O la URI de tu servidor
client = mlflow.tracking.MlflowClient()

# Listar todos los experimentos
all_experiments = client.search_experiments()
st.write("### Todos los experimentos:")
st.write([exp.name for exp in all_experiments])

# Filtrar experimentos (ajusta el prefijo)
target_experiments = [exp for exp in all_experiments if exp.name.startswith("eval_")]
st.write("### Experimentos filtrados (eval_):")
st.write([exp.name for exp in target_experiments])

if not target_experiments:
    st.error("No hay experimentos con prefijo 'eval_'. Verifica los nombres o el servidor de MLflow.")

