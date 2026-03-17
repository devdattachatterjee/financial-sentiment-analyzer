import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="centered"
)

# --- Model Integration ---
# Path to your specific Hugging Face model
MODEL_PATH = "Devda1421/financial-sentiment-distilbert"

@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

try:
    tokenizer, model = load_assets()
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}. Ensure the repository is public.")
    st.stop()

# --- UI Header ---
st.title("💰 Financial Sentiment Analyzer")
st.markdown(f"**Model Hub Path:** `{MODEL_PATH}`")
st.info("Analyzing news headlines using your custom fine-tuned DistilBERT transformer.")

# --- Inference Input ---
user_input = st.text_area(
    "Enter a financial headline or text segment:",
    placeholder="e.g., The company's quarterly revenue exceeded analyst expectations...",
    height=120
)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter text for analysis.")
    else:
        # 1. Tokenization
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # 2. Forward Pass (6-layer attention architecture)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # 3. Softmax Confidence Scoring
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        
        # Label Mapping (Standard for Financial-Phrasebank fine-tuning)
        # Note: Index 0: Negative, 1: Neutral, 2: Positive
        labels = ["Negative", "Neutral", "Positive"]
        prediction_idx = torch.argmax(logits).item()
        prediction = labels[prediction_idx]
        confidence = probs[prediction_idx]

        # --- Visualizing Results ---
        st.divider()
        
        # Display Summary Metrics
        m1, m2 = st.columns(2)
        m1.metric("Predicted Sentiment", prediction)
        m2.metric("Confidence", f"{confidence:.2%}")

        # Dataframe for the Bar Graph
        df_results = pd.DataFrame({
            "Sentiment": labels,
            "Probability": probs
        })

        # Sentiment Distribution Plot
        fig = px.bar(
            df_results, 
            x='Sentiment', 
            y='Probability', 
            color='Sentiment',
            text_auto='.2%',
            title="Softmax Probability Distribution",
            color_discrete_map={
                'Negative': '#e53935', 
                'Neutral': '#78909c', 
                'Positive': '#43a047'
            }
        )
        
        fig.update_layout(
            yaxis_range=[0, 1],
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# --- Sidebar Technical Stats ---
st.sidebar.title("System Info")
st.sidebar.write(f"**Architecture:** DistilBERT")
st.sidebar.write(f"**Model ID:** {MODEL_PATH}")
st.sidebar.write(f"**Deployment:** Streamlit Cloud")
