import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Fin-Intelligence NLP", page_icon="📈", layout="wide")

# --- 1. CSS Visibility & Styling Fix ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    
    /* Force Metric Labels and Values to be Dark/Visible */
    [data-testid="stMetricLabel"] {
        color: #31333F !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 24px !important;
    }
    
    /* Metric Card Styling */
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
    # Using your fine-tuned model
    model = DistilBertForSequenceClassification.from_pretrained("Devda1421/financial-sentiment-distilbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- 3. Label Mapping ---
# 0=Neg, 1=Neu, 2=Pos based on typical Financial PhraseBank tuning
label_map = {0: "Negative 🔴", 1: "Neutral ⚪", 2: "Positive 🟢"}
label_names = ["Negative", "Neutral", "Positive"]

# --- 4. Sidebar & Header ---
st.title("📊 Financial Sentiment & Signal Intelligence")
st.markdown("Fine-tuned **DistilBERT** Transformers for High-Precision Financial Analysis")

with st.sidebar:
    st.header("🛠️ Model Architecture")
    st.info("**Base Model:** DistilBERT")
    st.info("**Domain:** Financial NLP")
    st.divider()
    st.caption("Developed by Devdatta Chatterjee")

# --- 5. Session State Initialization ---
if "results" not in st.session_state:
    st.session_state.results = None

# --- 6. Main Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 News Headline Input")
    
    st.caption("Quick Test Scenarios:")
    c1, c2, c3 = st.columns(3)
    if c1.button("📈 Bullish", use_container_width=True):
        st.session_state["headline"] = "HDFC Bank reports record profits as loan demand surges across sectors."
    if c2.button("📉 Bearish", use_container_width=True):
        st.session_state["headline"] = "Market indices crash as global inflation fears spark a massive sell-off."
    if c3.button("📰 Neutral", use_container_width=True):
        st.session_state["headline"] = "The central bank maintains current interest rates in its latest policy review."

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
            inputs = tokenizer(headline_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
            pred_idx = probs.index(max(probs))
            
            st.session_state.results = {
                "label": label_map.get(pred_idx, "Unknown"),
                "confidence": probs[pred_idx],
                "all_probs": probs
            }

    if st.session_state.results:
        res = st.session_state.results
        
        # Display Metrics
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Market Sentiment", res["label"])
        m_col2.metric("Signal Confidence", f"{res['confidence']*100:.1f}%")
        
        # Display Chart
        prob_df = pd.DataFrame({'Sentiment': label_names, 'Probability': res["all_probs"]})
        fig = px.bar(
            prob_df, x='Sentiment', y='Probability', 
            color='Sentiment',
            color_discrete_map={'Positive':'#00CC96','Negative':'#EF553B','Neutral':'#636EFA'},
            text_auto='.2%'
        )
        fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 **Awaiting Input:** Click 'Run Signal Analysis' to visualize results.")

st.divider()
st.caption("Stack: Python · PyTorch · HuggingFace · Streamlit Cloud")
