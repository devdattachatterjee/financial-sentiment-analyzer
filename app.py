import streamlit as st
import PyPDF2
import numpy as np
import faiss
from openai import OpenAI

# --- 1. Page Configuration ---
st.set_page_config(page_title="Fin-Doc RAG Intelligence", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# Helper function to chunk text manually (replacing LangChain's RecursiveTextSplitter)
def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# --- 2. API Key & Setup ---
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    try:
        # Initialize the official OpenAI client
        client = OpenAI(api_key=api_key)
        
        # --- 3. Sidebar: Document Upload ---
        st.sidebar.header("📁 Document Ingestion")
        uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type="pdf")
        
        if uploaded_file:
            if "faiss_index" not in st.session_state:
                with st.status("🧠 Building Knowledge Base...", expanded=True) as status:
                    
                    # 1. Read PDF directly
                    st.write("Parsing document structure...")
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    full_text = ""
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() or ""
                    
                    # 2. Chunk the text
                    st.write("Generating semantic chunks...")
                    chunks = chunk_text(full_text)
                    st.session_state.chunks = chunks # Save chunks to map back to later
                    
                    # 3. Get Embeddings directly from OpenAI
                    st.write("Creating vector embeddings...")
                    response = client.embeddings.create(input=chunks, model="text-embedding-3-small")
                    embeddings = np.array([res.embedding for res in response.data]).astype("float32")
                    
                    # 4. Index in FAISS manually
                    st.write("Indexing vectors in FAISS...")
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(embeddings)
                    st.session_state.faiss_index = index
                    
                    status.update(label="✅ Analysis Ready!", state="complete", expanded=False)

        # --- 4. Main Chat Interface ---
        st.title("🤖 Financial Signal Assistant")
        st.caption("Context-Aware Analysis using Pure OpenAI API (No LangChain)")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Query the document (e.g., 'What is the GNPA percentage?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if "faiss_index" in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving data..."):
                        
                        # 1. Embed the user's question
                        query_res = client.embeddings.create(input=[prompt], model="text-embedding-3-small")
                        query_embedding = np.array([query_res.data[0].embedding]).astype("float32")
                        
                        # 2. Search FAISS for the top 3 closest chunks
                        distances, indices = st.session_state.faiss_index.search(query_embedding, k=3)
                        retrieved_chunks = [st.session_state.chunks[i] for i in indices[0]]
                        
                        # 3. Combine chunks into context
                        context = "\n\n---\n\n".join(retrieved_chunks)
                        
                        # 4. Construct the prompt for GPT
                        system_prompt = f"""You are a financial assistant. Use the following context extracted from a document to answer the user's question. If the answer is not in the context, say you don't know.
                        
                        CONTEXT:
                        {context}
                        """
                        
                        # 5. Call GPT directly
                        chat_completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            temperature=0,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        answer = chat_completion.choices[0].message.content
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")
