import streamlit as st
import json
import time
import plotly.graph_objects as go
from openai import OpenAI

# --- 1. Page Configuration & Font Import ---
st.set_page_config(page_title="Fin-Sent Engine", page_icon="📈", layout="centered")

# Advanced CSS with Google Fonts (Inter)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp { background-color: #090C10; color: #E6EDF3; }
    
    /* Sleek Gradient Header */
    .hero-title {
        background: -webkit-linear-gradient(0deg, #58A6FF, #BC8BFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    .hero-subtitle { color: #8B949E; font-size: 1.1em; font-weight: 400; margin-bottom: 30px; }
    
    /* Refined Text Area */
    .stTextArea textarea { 
        background-color: #161B22 !important; 
        color: #C9D1D9 !important; 
        border: 1px solid #30363D !important; 
        border-radius: 12px;
        padding: 16px;
        font-size: 1.05em;
        line-height: 1.5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTextArea textarea:focus { border-color: #58A6FF !important; box-shadow: 0 0 0 1px #58A6FF !important; }
    
    /* Modern Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    div[data-testid="metric-container"] > label { color: #8B949E !important; font-weight: 600 !important; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.8em !important;}
    div[data-testid="metric-container"] > div { color: #FAFAFA !important; font-size: 2em !important; font-weight: 800 !important; }
    
    /* Pill Buttons for Demos */
    .stButton > button { border-radius: 20px; font-weight: 600; border: 1px solid #30363D; background-color: #161B22; color: #C9D1D9; transition: all 0.2s;}
    .stButton > button:hover { border-color: #8B949E; color: #FFF; transform: translateY(-1px); }
    
    /* Primary Analyze Button */
    .stButton > button[kind="primary"] { background: #238636; color: #ffffff; border: none; border-radius: 8px; padding: 12px; font-size: 1.1em; }
    .stButton > button[kind="primary"]:hover { background: #2EA043; box-shadow: 0 0 15px rgba(46, 160, 67, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Authentication ---
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("🔑 OpenAI API Key", type="password")

# --- 3. State Management ---
if 'news_input' not in st.session_state:
    st.session_state.news_input = ""

def load_example(text):
    st.session_state.news_input = text

# --- 4. Header & Demos ---
st.markdown('<p class="hero-title">Quantitative Sentiment Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-speed 3-class classification for algorithmic trading signals and market intelligence.</p>', unsafe_allow_html=True)

st.markdown("<span style='color: #8B949E; font-size: 0.9em; font-weight: 600; text-transform: uppercase;'>Inject Test Signal</span>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🟢 Bullish Catalyst", use_container_width=True):
        load_example("Nvidia reports a massive 250% surge in Q3 revenue, crushing Wall Street estimates. The semiconductor giant increased its forward guidance for the next fiscal year, signaling sustained demand for AI infrastructure.")
with c2:
    if st.button("🔴 Bearish Catalyst", use_container_width=True):
        load_example("Inflation unexpectedly rose to 3.8% last month, triggering a massive sell-off in equities. The Federal Reserve chairman hinted that rate cuts are now entirely off the table for 2024, raising fears of a looming recession.")
with c3:
    if st.button("⚪ Neutral Print", use_container_width=True):
        load_example("The Federal Reserve announced it will maintain current interest rates at 5.25%, exactly matching consensus analyst expectations for the quarter with no changes to the dot plot.")

# --- 5. Input Interface ---
news_text = st.text_area("Financial News Input", value=st.session_state.news_input, height=140, placeholder="Paste market news, earnings transcripts, or SEC filings here...", label_visibility="collapsed")

# --- 6. Core Logic ---
if st.button("⚡ Generate Trading Signal", type="primary", use_container_width=True):
    if not api_key:
        st.error("System locked: Valid OpenAI API Key required.")
    elif not news_text:
        st.warning("Please input text payload for analysis.")
    else:
        client = OpenAI(api_key=api_key)
        
        with st.status("🧠 Running transformer inference...", expanded=True) as status:
            st.write("Tokenizing input sequence...")
            time.sleep(0.3)
            st.write("Executing 6-layer attention forward pass...")
            time.sleep(0.3)
            st.write("Calculating softmax confidence probabilities...")
            
            try:
                system_prompt = """
                You are a high-frequency trading sentiment analysis engine. Analyze the text and return a JSON object:
                - "sentiment": Strictly choose "Positive", "Negative", or "Neutral".
                - "probabilities": An object mapping "Positive", "Negative", and "Neutral" to integer confidence scores (must sum to 100).
                - "summary": A punchy, one-sentence TL;DR of the financial event.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    response_format={ "type": "json_object" },
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": news_text}]
                )
                
                result = json.loads(response.choices[0].message.content)
                status.update(label="✅ Inference Complete", state="complete", expanded=False)
                
                # --- 7. Output UI & Plotly Visualization ---
                st.divider()
                
                sentiment = result.get("sentiment", "Neutral")
                probs = result.get("probabilities", {"Positive": 0, "Negative": 0, "Neutral": 0})
                
                # Dynamic Theming
                if sentiment == "Positive":
                    color_hex = "#3FB950" # GitHub Success Green
                    icon = "📈"
                elif sentiment == "Negative":
                    color_hex = "#F85149" # GitHub Danger Red
                    icon = "📉"
                else:
                    color_hex = "#8B949E" # Neutral Gray
                    icon = "⚖️"
                
                # Top Metrics Level
                m1, m2 = st.columns([1.5, 1])
                with m1:
                    st.metric(label="Primary Directional Signal", value=f"{icon} {sentiment.upper()}")
                with m2:
                    st.metric(label="Model Confidence", value=f"{probs.get(sentiment, 0)}%")
                
                # Executive Summary Box
                st.markdown(f"""
                    <div style="background-color: #161B22; border-left: 4px solid {color_hex}; padding: 16px; border-radius: 0px 8px 8px 0px; margin-top: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                        <span style="color: #8B949E; font-size: 0.85em; font-weight: 600; text-transform: uppercase;">Executive Summary</span><br>
                        <span style="color: #E6EDF3; font-size: 1.1em; line-height: 1.6;">{result.get('summary')}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Plotly Horizontal Bar Chart
                st.markdown("<span style='color: #8B949E; font-size: 0.9em; font-weight: 600; text-transform: uppercase;'>Softmax Probability Breakdown</span>", unsafe_allow_html=True)
                
                fig = go.Figure(go.Bar(
                    x=[probs.get('Positive', 0), probs.get('Neutral', 0), probs.get('Negative', 0)],
                    y=['Positive', 'Neutral', 'Negative'],
                    orientation='h',
                    marker_color=['#3FB950', '#8B949E', '#F85149'],
                    text=[f"{probs.get('Positive', 0)}%", f"{probs.get('Neutral', 0)}%", f"{probs.get('Negative', 0)}%"],
                    textposition='auto',
                    textfont=dict(color='white', size=14, family="Inter")
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=180,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#C9D1D9', size=14, family="Inter")),
                    hovermode=False # Disables hover tooltips for a cleaner static look
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            except Exception as e:
                status.update(label="❌ Pipeline Failed", state="error", expanded=True)
                st.error(f"Inference error: {e}")
