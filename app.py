import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- Page Config ---
st.set_page_config(page_title="Fin-Doc RAG Intelligence", page_icon="🏦", layout="wide")

# --- 1. API Logic ---
# This looks for the secret you put in the Streamlit Dashboard
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)

    # --- 2. Document Ingestion ---
    uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type="pdf")
    
    if uploaded_file:
        if "vector_db" not in st.session_state:
            with st.spinner("Processing Document..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                texts = splitter.split_documents(docs)
                st.session_state.vector_db = FAISS.from_documents(texts, embeddings)
                st.success("✅ Knowledge Base Ready!")

    # --- 3. Chat Interface ---
    st.title("🤖 Financial Signal Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "vector_db" in st.session_state:
            with st.chat_message("assistant"):
                # Use invoke instead of the old __call__ for LangChain v0.1+
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", 
                    retriever=st.session_state.vector_db.as_retriever()
                )
                response = qa_chain.invoke(prompt)
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})
        else:
            st.error("Please upload a PDF first!")
else:
    st.warning("Please provide an OpenAI API Key to start.")
