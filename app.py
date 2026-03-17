import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px # Added for a cleaner bar chart
import warnings

warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide")

@st.cache_resource
def load_local_model():
    # Specific financial model - no API key needed
    model_name = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

analyzer = load_local_model()

st.title("📈 Financial Sentiment Analysis Dashboard")
st.markdown("Upload a CSV/Excel file or paste text to analyze sentiment using a local **FinancialBERT** model.")

# --- Sidebar for Uploads ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Financial Data (CSV or XLSX)", type=["csv", "xlsx"])
column_to_analyze = None

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.write("Preview of Data:")
    st.sidebar.dataframe(df.head(3))
    
    column_to_analyze = st.sidebar.selectbox("Select the column containing text:", df.columns)

# --- Main Logic ---
user_text = st.text_area("Or paste text manually (one statement per line):", height=150)

if st.button("Run Full Analysis"):
    statements = []
    
    # Decide source of data
    if uploaded_file is not None and column_to_analyze:
        statements = df[column_to_analyze].fillna("").astype(str).tolist()
    elif user_text.strip():
        statements = [line.strip() for line in user_text.split('\n') if line.strip()]
    
    if statements:
        with st.spinner(f"Analyzing {len(statements)} statements on GPU..."):
            # Local Inference
            results = analyzer(statements)
            
            # Process Results
            processed_data = []
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            
            for text, res in zip(statements, results):
                label = res['label'].lower()
                counts[label] += 1
                processed_data.append({
                    "Text": text,
                    "Sentiment": label.capitalize(),
                    "Score": res['score']
                })
            
            res_df = pd.DataFrame(processed_data)

            # --- Visuals ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Statements", len(statements))
            col2.metric("Positive %", f"{(counts['positive']/len(statements))*100:.1f}%")
            col3.metric("Negative %", f"{(counts['negative']/len(statements))*100:.1f}%")

            # The Bar Graph Comparison
            st.subheader("Sentiment Distribution")
            fig = px.bar(
                x=list(counts.keys()), 
                y=list(counts.values()),
                labels={'x': 'Sentiment Category', 'y': 'Number of Statements'},
                color=list(counts.keys()),
                color_discrete_map={'positive': '#2ecc71', 'neutral': '#f1c40f', 'negative': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table Output
            st.subheader("Detailed Results")
            st.dataframe(res_df, use_container_width=True)
            
            # Download Results
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "sentiment_results.csv", "text/csv")
            
    else:
        st.error("Please provide data via upload or text box.")
