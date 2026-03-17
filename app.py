import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd

# --- UI Setup ---
st.set_page_config(page_title="Fin-Intelligence NLP", page_icon="📈", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Financial Sentiment & Signal Intelligence")
st.markdown("Fine-tuned **DistilBERT** Transformers for High-Precision Financial Analysis · Built by **Devdatta Chatterjee**")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("Devda1421/financial-sentiment-distilbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_names = ["Negative", "Positive", "Neutral"]
label_map = {0: "Negative 🔴", 1: "Positive 🟢", 2: "Neutral ⚪"}

# --- Sidebar: Technical Stats ---
with st.sidebar:
    st.header("🛠️ Model Architecture")
    st.info("**Base:** DistilBERT-uncased")
    st.info("**Fine-tuned on:** Financial PhraseBank Dataset")
    st.info("**Framework:** PyTorch & HuggingFace")
    st.divider()
    st.markdown("### How it works")
    st.caption("Unlike generic NLP, this model understands financial context (e.g., 'crude oil spike' vs 'interest rate cut').")

# --- Main Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Headline Input")
    
    # Quick Examples
    st.caption("Quick Load Examples:")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    if ex_col1.button("📈 Bullish", use_container_width=True):
        st.session_state["headline"] = "Reliance Industries beats quarterly estimates, EBITDA margins expand by 200bps"
    if ex_col2.button("📉 Bearish", use_container_width=True):
        st.session_state["headline"] = "Regulatory crackdown on fintech giants leads to significant market sell-off"
    if ex_col3.button("📰 Neutral", use_container_width=True):
        st.session_state["headline"] = "RBI maintains repo rate at 6.5%, focus remains on withdrawal of accommodation"

    headline = st.text_area(
        "Financial Context:",
        value=st.session_state.get("headline", ""),
        height=150,
        placeholder="Paste a news headline or corporate statement here..."
    )
    
    analyze = st.button("🚀 Analyze Signal Strength", type="primary", use_container_width=True)

with col_right:
    st.subheader("🎯 Analysis Output")
    
    if analyze and headline:
        with st.spinner("Processing deep-learning layers..."):
            inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            # KPI Cards
            res_1, res_2 = st.columns(2)
            res_1.metric("Market Sentiment", label_map[pred_idx])
            res_2.metric("Signal Confidence", f"{confidence*100:.1f}%")
            
            # Probability Chart
            prob_df = pd.DataFrame({
                'Label': ["Negative", "Positive", "Neutral"],
                'Probability': [float(p) for p in probs]
            })
            
            fig = px.bar(prob_df, x='Label', y='Probability', 
                         color='Label', 
                         color_discrete_map={'Positive':'#00CC96','Negative':'#EF553B','Neutral':'#636EFA'},
                         text_auto='.2%')
            fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
    elif analyze and not headline:
        st.warning("Please enter a headline to begin analysis.")
    else:
        st.info("Awaiting input for real-time inference.")

# --- Bottom Section: Why this model? ---
st.divider()
st.subheader("📈 Signal Explained")
exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    st.markdown("**Context Awareness**")
    st.caption("DistilBERT captures long-range dependencies in text, understanding that the subject and object of a sentence determine the financial impact.")

with exp_col2:
    st.markdown("**Stochastic Optimization**")
    st.caption("Softmax normalization provides a probabilistic confidence score, allowing for threshold-based trading signal generation.")

with exp_col3:
    st.markdown("**Biotech-Ready**")
    st.caption("Can be extended to analyze clinical trial results or FDA approval news—bridging the gap between Biology and Finance.")
