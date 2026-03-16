import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📊", layout="centered")

st.title("📊 Financial Sentiment Analyzer")
st.markdown("Fine-tuned DistilBERT on financial news · Built by Devdatta Chatterjee")
st.divider()

@st.cache_resource
def load_model():
    from transformers import DistilBertForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # load tokenizer directly from HuggingFace
    
    model = DistilBertForSequenceClassification.from_pretrained("./financial_sentiment_model")
    # load model using specific class instead of Auto
    # Auto was failing because of version mismatch in config
    # DistilBertForSequenceClassification loads it directly, no version check
    
    model.eval()
    return tokenizer, model
tokenizer, model = load_model()
label_names = ["Negative 🔴", "Positive 🟢", "Neutral ⚪"]

headline = st.text_area("Enter a financial headline:", placeholder="e.g. Infosys reports 18% jump in quarterly profits", height=100)

st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Positive"):
        headline = "Reliance Industries posts record revenue, shares surge 8%"
with col2:
    if st.button("Negative"):
        headline = "Yes Bank shares crash 34% after RBI imposes withdrawal limits"
with col3:
    if st.button("Neutral"):
        headline = "SEBI announces new disclosure norms for listed companies"

if st.button("Analyze", type="primary") and headline:
    with st.spinner("Analyzing..."):
        inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

    st.divider()
    st.markdown(f"### {label_names[pred_idx]}")
    st.markdown(f"**Confidence: {confidence:.1f}%**")
    for i, (name, prob) in enumerate(zip(label_names, probs)):
        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

st.divider()
st.caption("DistilBERT fine-tuned on Twitter Financial News Sentiment dataset")