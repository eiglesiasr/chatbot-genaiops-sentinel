import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2_resumido_directo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
DATASET_PATH = "tests/eval_dataset.json"

# Cargar dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LangChain Evaluator
llm = ChatOpenAI(temperature=0)

# ✅ Criterios válidos como strings
criteria = {
    "correctness": "Is the answer factually accurate?",
    "relevance": "Is the answer relevant to the input question?",
    "coherence": "Is the answer coherent and understandable?",
    "toxicity": "Is the answer free of harmful or toxic content?",
    "harmfulness": "Does the answer avoid causing harm?"
}


eval_chain = []

for c in criteria:
    eval_chain.append(
        {'eval':load_evaluator(
            "labeled_score_string",
            criteria={c : criteria[c]},
            llm=llm,
        ),
        'criteria':c
        }
    )

# ✅ Establecer experimento una vez
mlflow.set_experiment(f"eval_criteria_{PROMPT_VERSION}_{CHUNK_SIZE}")
print(f"📊 MLflow Experiment: eval_criteria_{PROMPT_VERSION}")

# Evaluación por lote
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]
    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]


        for eval_i in eval_chain:
            graded = eval_i['eval'].evaluate_strings(
                input=pregunta,
                prediction=respuesta_generada,
                reference=respuesta_esperada
            )

            # 🔍 Imprimir y guardar métricas
            print(f"\n📦 Resultado evaluación con criterios para pregunta {i+1}/{len(dataset)}:")
            mlflow.log_param("question", pregunta)
            mlflow.log_param("prompt_version", PROMPT_VERSION)
            mlflow.log_param("chunk_size", CHUNK_SIZE)
            mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

            criterion = eval_i['criteria']
            score = graded['score']
            print(f"{criterion.capitalize()}: {score}")
            mlflow.log_metric(f"{criterion}_score", score)

            print(f"✅ Pregunta: {pregunta}")
            print(f"🧠 Respuesta generada: {respuesta_generada}")
