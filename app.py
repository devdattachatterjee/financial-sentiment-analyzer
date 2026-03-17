import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. Page Configuration ---
st.set_page_config(page_title="Fin-Doc RAG Intelligence", page_icon="🏦", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. API Key & Model Setup ---
# Checks Streamlit Secrets first, then sidebar input
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key, temperature=0)
        
        # --- 3. Sidebar: Document Upload ---
        st.sidebar.header("📁 Document Ingestion")
        uploaded_file = st.sidebar.file_uploader("Upload Financial PDF (RBI/Annual Report)", type="pdf")
        
        if uploaded_file:
            if "vector_db" not in st.session_state:
                with st.status("🧠 Building Knowledge Base...", expanded=True) as status:
                    # Save temporary file
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.write("Parsing document structure...")
                    loader = PyPDFLoader("temp.pdf")
                    docs = loader.load()
                    
                    st.write("Generating semantic chunks...")
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    texts = splitter.split_documents(docs)
                    
                    st.write("Indexing vectors in FAISS...")
                    st.session_state.vector_db = FAISS.from_documents(texts, embeddings)
                    status.update(label="✅ Analysis Ready!", state="complete", expanded=False)

        # --- 4. Main Chat Interface ---
        st.title("🤖 Financial Signal Assistant")
        st.caption("Context-Aware Analysis of Regulatory and Corporate Filings")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Query the document (e.g., 'What is the GNPA percentage?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if "vector_db" in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving data..."):
                        # RAG Chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}),
                            return_source_documents=True
                        )
                        # Updated LangChain 0.1+ syntax
                        response = qa_chain.invoke({"query": prompt})
                        answer = response["result"]
                        
                        # Display the answer to the user and add to history
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Catching any API or Langchain initialization errors
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    # Prompt user to enter key if missing
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")
