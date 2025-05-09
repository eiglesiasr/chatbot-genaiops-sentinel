# app/ui_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="🤖🚀 Satellite Assistant", layout="centered")

from app.rag_pipeline import load_vectorstore_from_disk, build_chain


st.title("🤖🚀 Satellite Assistant")

question = st.text_input("What do you want to know? / ¿Qué deseas consultar? / 何をお知りになりたいですか？ ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cargar vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb)

if question:
    with st.spinner("Thinking..."):
        result = chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((question, result["answer"]))

if st.session_state.chat_history:
    st.markdown("---")    
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 User:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
