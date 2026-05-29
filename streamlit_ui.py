import streamlit as st
from rag_pipeline_file import loader, textsplliter, create_vectorstore, similar_documents, model_response
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
import os

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
    
    st.markdown("---")
    upload_file = st.file_uploader("Upload PDF")

    if upload_file and st.session_state.vector_store is None:
        with st.spinner("Processing PDF..."):
            tmp_path = process_pdf(upload_file)
            docs = loader(tmp_path)
            chunks = textsplliter(docs)
            st.session_state.vector_store = create_vectorstore(chunks)
            os.unlink(tmp_path)
        st.success("PDF ready!")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask something about your PDF...")

if user_input:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = model_response(user_input, st.session_state.chat_history, st.session_state.vector_store)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)