import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# --- Custom CSS for "Production-grade" Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0e1117;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading (Cached) ---
# Note: Using a standard financial-tuned DistilBERT from HuggingFace Hub
MODEL_NAME = "mrm8488/distilbert-base-uncased-finetuned-financial-phrasebank"

@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_assets()

# --- App Header ---
st.title("📈 Financial Sentiment Analyzer")
st.caption("Fine-tuned DistilBERT Transformer | PyTorch | Streamlit Cloud")
st.info("Classifies financial news into 3-class sentiment (Positive / Negative / Neutral) with confidence scoring.")

# --- User Input Section ---
with st.container():
    user_input = st.text_area(
        "Enter Financial Headline:", 
        placeholder="e.g., Q3 profits exceeded expectations despite global supply chain pressures...",
        height=100
    )
    
    analyze_btn = st.button("Run Inference")

# --- Inference & Visualization Logic ---
if analyze_btn:
    if not user_input.strip():
        st.error("Please enter a headline to analyze.")
    else:
        with st.spinner("Processing through transformer layers..."):
            # 1. Tokenization
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

            # 2. Forward Pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # 3. Softmax for Confidence Scores
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            labels = ["Negative", "Neutral", "Positive"]
            
            # Create Results DataFrame
            df_results = pd.DataFrame({
                "Sentiment": labels,
                "Confidence": probs
            })
            
            # Get top prediction
            prediction = labels[torch.argmax(logits).item()]
            confidence = max(probs)

        # --- Display Results ---
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Predicted Sentiment", prediction)
            st.metric("Confidence Score", f"{confidence:.2%}")
            
        with col2:
            # Sentiment Distribution Bar Graph
            fig = px.bar(
                df_results, 
                x='Sentiment', 
                y='Confidence', 
                color='Sentiment',
                text_auto='.2%',
                title="Confidence Distribution",
                color_discrete_map={
                    'Negative': '#d32f2f', 
                    'Neutral': '#546e7a', 
                    'Positive': '#388e3c'
                }
            )
            fig.update_layout(
                yaxis_range=[0, 1],
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Sidebar Documentation ---
st.sidebar.header("Project Technical Specs")
st.sidebar.markdown(f"""
- **Model:** `DistilBERT-base-uncased`
- **Params:** 67 Million
- **Accuracy:** 85%+ (on Financial Phrasebank)
- **Pipeline:** Tokenization → 6-layer attention → Softmax
""")
          
