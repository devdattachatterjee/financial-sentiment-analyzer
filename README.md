# 📊 Financial Sentiment Analyzer

Fine-tuned DistilBERT transformer for classifying financial news sentiment — Positive, Negative, or Neutral.

## 🔗 Live Demo
👉 [Try the live app here](https://financial-sentiment-analyzer-ugdoqptuj4oekzrqxbb8jg.streamlit.app)

## 🧠 Model Details
- **Architecture**: DistilBERT base uncased (67M parameters)
- **Task**: 3-class sentiment classification
- **Classes**: Positive · Negative · Neutral
- **Training data**: Financial News Sentiment dataset
- **Accuracy**: 85%+
- **Model hosted**: [HuggingFace Hub](https://huggingface.co/Devda1421/financial-sentiment-distilbert)

## 🛠️ Tech Stack
- HuggingFace Transformers
- PyTorch
- Streamlit

## 🚀 Run Locally
```bash
git clone https://github.com/devdattachatterjee/financial-sentiment-analyzer
cd financial-sentiment-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 💡 How It Works
1. User enters any financial headline
2. DistilBERT tokenizer converts text to token IDs
3. Fine-tuned model runs forward pass through 6 transformer layers
4. Softmax converts raw logits to confidence probabilities
5. App displays prediction with full confidence breakdown

## 👤 Author
**Devdatta Chatterjee**  
PGP Data Science & AI/ML — Praxis Tech School, Kolkata  
[GitHub](https://github.com/devdattachatterjee)
