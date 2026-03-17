import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Fin-Intelligence NLP", page_icon="📈", layout="wide")

# Custom CSS for a clean, professional dashboard look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. Model Loading ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("Devda1421/financial-sentiment-distilbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_map = {0: "Negative 🔴", 1: "Positive 🟢", 2: "Neutral ⚪"}
label_names = ["Negative", "Positive", "Neutral"]

# --- 2. Title & Intro ---
st.title("📊 Financial Sentiment & Signal Intelligence")
st.markdown("Fine-tuned **DistilBERT** Transformers for High-Precision Financial Analysis · Built by **Devdatta Chatterjee**")
st.divider()

# --- 3. Sidebar Configuration ---
with st.sidebar:
    st.header("🛠️ Model Architecture")
    st.info("**Base:** DistilBERT-uncased")
    st.info("**Fine-tuned on:** Financial PhraseBank")
    st.info("**Optimization:** Softmax Normalization")
    st.divider()
    st.markdown("### Decision Logic")
    st.caption("This engine identifies market sentiment shifts by analyzing linguistic nuances in corporate reporting and financial news.")

# --- 4. Main UI Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 News Headline Input")
    
    # Example buttons to trigger session state
    st.caption("Quick Test Scenarios:")
    ex_c1, ex_c2, ex_c3 = st.columns(3)
    if ex_c1.button("📈 Bullish", use_container_width=True):
        st.session_state["headline"] = "HDFC Bank net profit jumps 30% YoY, asset quality remains robust"
    if ex_c2.button("📉 Bearish", use_container_width=True):
        st.session_state["headline"] = "Tech stocks plummet as global central banks hint at prolonged high interest rates"
    if ex_c3.button("📰 Neutral", use_container_width=True):
        st.session_state["headline"] = "Standard & Poor's affirms India's sovereign rating with stable outlook"

    # Input Area
    input_text = st.text_area(
        "Enter financial text for analysis:",
        value=st.session_state.get("headline", ""),
        height=180,
        placeholder="Paste a news headline, quarterly result summary, or market update..."
    )
    
    analyze_btn = st.button("🔍 Run Signal Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("🎯 Analysis Output")
    
    # Logic to trigger only when button is pressed
    if analyze_btn and input_text:
        with st.spinner("Executing Transformer layers..."):
            # Tokenization
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Probability Calculation
            probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
            pred_idx = probs.index(max(probs))
            confidence = probs[pred_idx]
            
            # Result Display - Metric Cards
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Market Sentiment", label_map[pred_idx])
            m_col2.metric("Signal Confidence", f"{confidence*100:.1f}%")
            
            # Result Display - Visualization
            prob_df = pd.DataFrame({
                'Sentiment': label_names,
                'Probability': probs
            })
            
            fig = px.bar(
                prob_df, 
                x='Sentiment', 
                y='Probability', 
                color='Sentiment',
                color_discrete_map={'Positive':'#00CC96','Negative':'#EF553B','Neutral':'#636EFA'},
                text_auto='.2%'
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(t=10, b=10, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
    elif analyze_btn and not input_text:
        st.warning("⚠️ Please provide input text to analyze.")
    else:
        st.info("💡 **Awaiting Signal:** Enter a headline in the left panel and click 'Run Signal Analysis' to see deep-learning insights.")

# --- 5. Footer / Technical Context ---
st.divider()
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Transfer Learning**")
    st.caption("Leverages DistilBERT's pre-trained linguistic knowledge, specifically fine-tuned for the unique vocabulary of global financial markets.")
with f2:
    st.markdown("**Real-time Inference**")
    st.caption("Optimized PyTorch backend allows for sub-second sentiment scoring, suitable for high-frequency news monitoring pipelines.")
with f3:
    st.markdown("**Biotech Applicability**")
    st.caption("Architecture is modular—capable of being re-tuned for Pharma-specific news, clinical trial outcomes, and healthcare regulatory filings.")
