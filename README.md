# 🛰️ Sentinel-1 Chatbot (basado en GenAIOps)

Este proyecto es una adaptación del repositorio original [GenAIOps_Pycon2025](https://github.com/darkanita/GenAIOps_Pycon2025), redirigido al dominio de las misiones espaciales. Implementa un chatbot tipo RAG (Retrieval-Augmented Generation) para responder preguntas relacionadas con la misión Sentinel-1, un satélite equipado con un radar de apertura sintética (SAR).

---

## 📚 Nuevas fuentes de conocimiento

En lugar de documentos internos empresariales, este sistema se alimenta de:

- **Documento de definición del producto Sentinel-1**: Describe el sistema, especificaciones técnicas y capacidades del sensor SAR. Disponible en [Sentiwiki](https://sentiwiki.copernicus.eu/__attachments/1673968/S1-RS-MDA-52-7440%20-%20Sentinel-1%20Product%20Definition%202016%20-%202.7.pdf)
- **Reporte del estado actual de la misión Sentinel-1**: Incluye información operativa, actualizaciones y estado de la mision para febrero de 2025. Disponible en [Sentiwiki](https://sentiwiki.copernicus.eu/__attachments/1681272/Sentinel-1-Mission_Status_Report_440.pdf?inst-v=098dc289-59b8-4d70-ad21-3a78e6b0a4b0)

> Estos documentos han sido indexados con LlamaIndex para permitir la recuperación semántica contextual.

## 💬 Modos de interacción: Prompts personalizados
Se han definido cinco estilos de asistente (prompt templates) para evaluar cómo varía el comportamiento del modelo según el rol asignado:
| **Versión**               | **Descripción**                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| `v1_asistente_cientifico` | Asistente profesional, tono científico-formal.                       |
| `v2_resumido_directo`     | Responde de forma breve y directa, sin rodeos.                       |
| `v3_alucinogeno`          | No usa prompt, respuestas sin filtro.                                |
| `v4_asistente_experto`    | Especialista en satélites y misiones espaciales, tono claro.         |
| `v5_asistente_orgulloso`  | También experto, pero propenso a inventar si no conoce la respuesta. |


## ✅ Evaluación automática

Se ha ampliado el sistema de evaluación para incluir cinco criterios de calidad, implementados en `run_eval_criteria.py`:
| **Criterio**  | **Descripción**                                     |
| ------------- | ----------------------------------------------- |
| `correctness` | ¿Es la respuesta fácticamente precisa?          |
| `relevance`   | ¿Está relacionada directamente con la pregunta? |
| `coherence`   | ¿Es clara y coherente?                          |
| `toxicity`    | ¿Evita lenguaje ofensivo o problemático?        |
| `harmfulness` | ¿Evita causar daño o inducir a errores?         |

Las respuestas generadas por cada asistente son evaluadas automáticamente por un LLM evaluador siguiendo estos criterios.

## 📊 Visualización de resultados
Los resultados de la evaluación pueden consultarse de dos formas:

- main_interface.py: Permite consultar la evaluación de una respuesta específica.

- dashboard.py: Muestra visualizaciones comparativas entre diferentes versiones de asistente.

Esto permite hacer análisis A/B de calidad y comportamiento del modelo bajo distintas condiciones.

---

## 🚀 Demo

### 1. 🧱 Preparación del entorno

Puedes clonar el repositorio y crear un entorno virtual con Conda:
```bash
git clone https://github.com/darkanita/GenAIOps_Pycon2025 chatbot-genaiops
cd chatbot-genaiops
conda create -n chatbot-genaiops python=3.10 -y
conda activate chatbot-genaiops
pip install -r requirements.txt
```
O bien, puedes usar CodeSpaces por medio del DevContainer de GitHub. Esto te permitirá ejecutar el proyecto sin necesidad de instalar nada en tu máquina local. De cualquier modo, copia el archivo por medio de 
```bash
cp .env.example .env
```
para cargar las variables del sistema. 

> 💡
> Recuerda agregar tu API KEY. Acá también puedes elegir la versión de Prompt para que el modelo use la que prefieras. Por defecto, se usa `v1_asistente_cientifico`. Además del `CHUNK_SIZE` y `CHUNK_OVERLAP` de tu preferencia

### 2. 💻 Ejecuta la app principal

Primero procesa los PDFs y genera el índice vectorial por medio del siguiente comando:

```bash
python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"
```
Después, ejecuta la app principal, donde podrás hacer preguntas al chatbot y ver las métricas de evaluación (tradicionales y semánticas):
```bash
streamlit run app/main_interface.py
```

### 3. 🧪 Evaluación automática de calidad

Usando `tests/eval_dataset.json` como ground truth, ejecuta la evaluación automática de calidad. Este script evalúa el rendimiento del modelo en función de los criterios definidos.

```bash
python app/run_eval.py # Evaluación tradicional de la calidad
python app/run_eval_criteria.py # Evaluación semántica de la calidad
```

#### 📈 Visualización de resultados

Puedes observar los resultados de la evaluación en el dashboard. Este script genera gráficos y tablas para comparar el rendimiento de diferentes versiones del asistente para las preguntas del dataset de evaluación.

```bash
streamlit run app/dashboard.py
```

#### 🛠 Validación automatizada

Puedes evaluar el sistema de una forma alternativa y asegurarte que éste tenga al menos 80% de precisión con el dataset base.

```bash
pytest tests/test_run_eval.py
```
# Chatbot GenAI + Ops Sentinel  

## 📌 Evidencias Experimentales  

### 🔍 Hallazgos Clave  

#### 1. Optimización de Tamaño de Chunks  
- **Chunk size óptimo**: **1024 tokens** (balance ideal entre contexto y eficiencia procesamiento).  
- **Gráfica de soporte**:  
  ![Comparativa Chunk Size](/evidencias/comparativa_experimentos_chunk_size.png)  

#### 2. Efectividad de Tipos de Prompt  
- **Prompts estructurados** (ej: con pasos o ejemplos) incrementan la precisión en **~22%** frente a prompts libres.  
- **Gráfica de soporte**:  
  ![Comparativa de Prompts](/evidencias/comparativa_tipo_prompt.png)  

#### 3. Pipeline de Experimentos  
- **Flujo validado**:  
  1. **Chunking** (1024 tokens).  
  2. **Procesamiento** con LLMs (GPT-3.5/Mistral).  
  3. **Post-procesamiento** y evaluación.  
- **Diagrama**:  
  ![Pipeline Experimental](/evidencias/mlflow_experiments.png)  

#### 4. Resúmenes Automáticos  
- **Chunks de 1024 tokens** permiten generar resúmenes coherentes en respuestas largas.  
- **Ejemplo visual**:  
  ![Resúmenes](/evidencias/v2_resumido_criterio_1024.png)  

---

## 🛠️ Configuración Técnica  
```bash
git clone https://github.com/eiglesiasr/chatbot-genaiops-sentinel.git  
pip install -r requirements.txt  
