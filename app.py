import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Fin-Intelligence NLP", page_icon="📈", layout="wide")

# --- 1. CSS Visibility & Styling Fix ---
# This forces the metrics to be visible (black text) even if the browser defaults to dark mode
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    
    /* Force Metric Labels to be Dark Grey */
    [data-testid="stMetricLabel"] {
        color: #31333F !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    /* Force Metric Values (Sentiment & %) to be Pitch Black */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 24px !important;
    }
    
    /* Card-style container for Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Model Loading ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("Devda1421/financial-sentiment-distilbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- 3. Label Mapping ---
# Based on standard Financial PhraseBank orders: 0=Neg, 1=Neu, 2=Pos
label_map = {0: "Negative 🔴", 1: "Neutral ⚪", 2: "Positive 🟢"}
label_names = ["Negative", "Neutral", "Positive"]

# --- 4. Sidebar & Header ---
st.title("📊 Financial Sentiment & Signal Intelligence")
st.markdown("Fine-tuned **DistilBERT** Transformers for High-Precision Financial Analysis")

with st.sidebar:
    st.header("🛠️ Model Architecture")
    st.info("**Base Model:** DistilBERT")
    st.info("**Specialization:** Financial NLP")
    st.divider()
    st.markdown("### Contextual Awareness")
    st.caption("Captures complex financial semantics like 'yield compression' or 'margin expansion' that generic models miss.")

# --- 5. Session State Initialization ---
if "results" not in st.session_state:
    st.session_state.results = None

# --- 6. Main Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 News Headline Input")
    
    # Example prompts
    st.caption("Quick Load Examples:")
    c1, c2, c3 = st.columns(3)
    if c1.button("📈 Bullish", use_container_width=True):
        st.session_state["headline"] = "Adani Ports reports 20% jump in cargo volumes; raises full-year guidance."
    if c2.button("📉 Bearish", use_container_width=True):
        st.session_state["headline"] = "Inflation data exceeds estimates, sparking fears of an aggressive rate hike by RBI."
    if c3.button("📰 Neutral", use_container_width=True):
        st.session_state["headline"] = "Quarterly results for mid-cap IT firms expected to remain in-line with street expectations."

    headline_input = st.text_area(
        "Enter financial context:",
        value=st.session_state.get("headline", ""),
        height=180,
        placeholder="Type or paste financial news here..."
    )
    
    analyze_btn = st.button("🔍 Run Signal Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("🎯 Analysis Output")
    
    if analyze_btn and headline_input:
        with st.spinner("Analyzing linguistic features..."):
            # Inference
            inputs = tokenizer(headline_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Probability calculation
            probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
            pred_idx = probs.index(max(probs))
            
            # Store in Session State
            st.
