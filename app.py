import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📊", layout="centered")

st.title("📊 Financial Sentiment Analyzer")
st.markdown("Fine-tuned DistilBERT on financial news · Built by **Devdatta Chatterjee**")
st.divider()

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("Devda1421/financial-sentiment-distilbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_names = ["Negative 🔴", "Positive 🟢", "Neutral ⚪"]
label_colors = {"Negative 🔴": "🔴", "Positive 🟢": "🟢", "Neutral ⚪": "⚪"}

st.markdown("### Try an example or type your own")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📈 Positive", use_container_width=True):
        st.session_state["headline"] = "Reliance Industries posts record revenue, shares surge 8%"
with col2:
    if st.button("📉 Negative", use_container_width=True):
        st.session_state["headline"] = "Yes Bank shares crash 34% after RBI imposes withdrawal limits"
with col3:
    if st.button("📰 Neutral", use_container_width=True):
        st.session_state["headline"] = "SEBI announces new disclosure norms for listed companies"

headline = st.text_area(
    "Enter a financial headline:",
    value=st.session_state.get("headline", ""),
    placeholder="e.g. Infosys reports 18% jump in quarterly profits driven by cloud demand",
    height=100
)

st.markdown("")
analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

if analyze and headline:
    with st.spinner("Analyzing..."):
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

    st.divider()
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric(label="Prediction", value=label_names[pred_idx])
    with col_b:
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

    st.markdown("**Confidence breakdown:**")
    for name, prob in zip(label_names, probs):
        st.progress(float(prob), text=f"{name}  —  {prob*100:.1f}%")

elif analyze and not headline:
    st.warning("Please enter a headline first.")

st.divider()
st.caption("Stack: DistilBERT · HuggingFace Transformers · PyTorch · Streamlit")
