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
Financial markets move on news. When RBI announces a rate hike, when a company posts earnings, when a CEO resigns — stock prices react within seconds. Human analysts can't read thousands of headlines simultaneously. This model can process thousands of headlines per minute and flag negative signals automatically.

**Algorithmic trading** — Sentiment signals used as input features for trading strategies. A spike in negative sentiment around a stock can trigger a short position automatically.

**Credit risk monitoring** — Banks and NBFCs monitor news about borrowers continuously. Negative sentiment around a corporate borrower triggers early warnings in credit risk systems — directly complementing NPA forecasting models.

**Portfolio management** — Asset managers track sentiment across sectors to rebalance before price corrections happen.

This model is the foundation layer for all three use cases.

## 👤 Author
Devdatta Chatterjee 
