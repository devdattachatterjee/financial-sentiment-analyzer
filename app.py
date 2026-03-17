import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="💰",
    layout="centered"
)

# --- Session State Initialization ---
# This is required to make the preset buttons talk to the text area
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

def set_example(text):
    st.session_state.text_input = text

# --- Model Integration ---
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

# --- Preset Example Buttons ---
st.write("**Try a pre-made example or enter your own:**")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("🟢 Positive Example", on_click=set_example, 
              args=("The company reported a record 45% increase in Q3 profit margins, crushing Wall Street estimates.",), 
              use_container_width=True)
with col2:
    st.button("⚪ Neutral Example", on_click=set_example, 
              args=("The board of directors announced a regular quarterly dividend of $0.20 per share, unchanged from last quarter.",), 
              use_container_width=True)
with col3:
    st.button("🔴 Negative Example", on_click=set_example, 
              args=("Due to ongoing supply chain disruptions, operating losses widened by $50 million this fiscal year.",), 
              use_container_width=True)

# --- Inference Input ---
# Linking the text area to the session state so the buttons update it
user_input = st.text_area(
    "Enter a financial headline or text segment:",
    key="text_input",
    height=120
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter text for analysis.")
    else:
        # 1. Tokenization
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # 2. Forward Pass 
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # 3. Softmax Confidence Scoring
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        
        # Label Mapping
        labels = ["Negative", "Neutral", "Positive"]
        prediction_idx = torch.argmax(logits).item()
        prediction = labels[prediction_idx]
        confidence = probs[prediction_idx]

        # --- Visualizing Results ---
        st.divider()
        
        # Display Summary Metrics
        m1, m2 = st.columns(2)
        m1.metric("Predicted Sentiment", prediction)
        m2.metric("Confidence Score", f"{confidence:.2%}")

        # Dataframe for the Bar Chart
        df_results = pd.DataFrame({
            "Sentiment": labels,
            "Probability": probs
        })

        # Sentiment Distribution Bar Chart
        fig = px.bar(
            df_results, 
            x='Sentiment', 
            y='Probability', 
            color='Sentiment',
            text_auto='.2%',
            title="Model Confidence Distribution",
            color_discrete_map={
                'Negative': '#e53935', # Red
                'Neutral': '#78909c',  # Gray
                'Positive': '#43a047'  # Green
            }
        )
        
        fig.update_layout(
            yaxis_range=[0, 1],
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
