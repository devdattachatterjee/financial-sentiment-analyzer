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

## 💡 How It Works
1. User enters any financial headline
2. DistilBERT tokenizer converts text to token IDs
3. Fine-tuned model runs forward pass through 6 transformer layers
4. Softmax converts raw logits to confidence probabilities
5. App displays prediction with full confidence breakdown

## 🎯 Why This Project
Financial markets move on news. Being able to automatically classify whether a headline is bullish, bearish, or neutral at scale has direct applications in algorithmic trading, risk monitoring, and BFSI analytics — which is the domain I specialise in.

## 👤 Author
Devdatta Chatterjee 
