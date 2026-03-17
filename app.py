import streamlit as st
import json
import time
from openai import OpenAI

# --- 1. Page Configuration ---
st.set_page_config(page_title="Fin-Sent AI | Market Intelligence", page_icon="📈", layout="wide")

# --- 2. Enterprise CSS Injection ---
st.markdown("""
    <style>
    /* Global Theme */
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1e293b; }
    
    /* Gradient Header */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-text { color: #94a3b8; font-size: 1.1em; margin-bottom: 25px; }
    
    /* Styled Text Area */
    .stTextArea textarea { 
        background-color: #1e293b !important; 
        color: #f8fafc !important; 
        border: 1px solid #334155 !important; 
        border-radius: 10px;
        padding: 15px;
        font-size: 1.05em;
    }
    .stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important; }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-2px); border-color: #475569; }
    div[data-testid="metric-container"] > label { color: #94a3b8 !important; font-size: 0.9em !important; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="metric-container"] > div { color: #f8fafc !important; font-size: 1.8em !important; font-weight: 700 !important; }
    
    /* Custom Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        border: none;
        padding: 10px 24px;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        transform: scale(1.02);
    }
    
    /* Dividers and Expanders */
    hr { border-color: #1e293b; }
    .stExpander { background-color: #111827; border: 1px solid #1e293b; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Authentication ---
api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")

# --- 4. Sidebar Elements ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=60) # Sleek icon
    st.markdown("### Fin-Sent Engine")
    st.caption("v2.0 | Advanced NLP Interface")
    
    st.divider()
    
    st.markdown("#### ⚡ Quick Test Samples")
    if st.button("📈 Load Bullish News", use_container_width=True):
        st.session_state.news_input = "Nvidia reports a 250% surge in Q3 revenue, crushing Wall Street estimates. The semiconductor giant increased its forward guidance for the next fiscal year, signaling sustained demand for AI chips across global data centers."
    if st.button("📉 Load Bearish News", use_container_width=True):
        st.session_state.news_input = "Inflation unexpectedly rose to 3.8% last month, triggering a massive sell-off in equities. The Federal Reserve chairman hinted that rate cuts are now entirely off the table for 2024, raising fears of a looming recession."
    
    st.divider()
    st.markdown("#### ⚙️ System Specs")
    st.caption("• **Core:** GPT-4o-Mini\n• **Task:** NER & Sentiment\n• **Latency:** < 2.5s")

# --- 5. Main UI Header ---
st.markdown('<p class="gradient-text">Market Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Extract actionable intelligence, directional sentiment, and fundamental drivers from raw financial text in seconds.</p>', unsafe_allow_html=True)

# --- 6. Main Application Logic ---
if api_key:
    client = OpenAI(api_key=api_key)
    
    # Pre-fill text area if sample is clicked
    default_text = st.session_state.get('news_input', '')
    news_text = st.text_area("Input Financial Text", height=180, value=default_text, placeholder="Paste a press release, earnings call excerpt, or market news here...", label_visibility="collapsed")
    
    # Layout for the analyze button to make it look cleaner
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Signal", type="primary", use_container_width=True)
    
    if analyze_clicked and news_text:
        # Animated loading state to simulate deep processing
        with st.status("🧠 Processing NLP Pipeline...", expanded=True) as status:
            st.write("Tokenizing input text...")
            time.sleep(0.5)
            st.write("Extracting named entities and market drivers...")
            time.sleep(0.5)
            st.write("Scoring directional sentiment...")
            
            try:
                system_prompt = """
                You are a senior quantitative analyst. Analyze the following text and output a strictly formatted JSON object with these keys:
                - "sentiment": Exactly "Bullish", "Bearish", or "Neutral".
                - "confidence": An integer between 0 and 100.
                - "summary": A concise, one-sentence executive summary.
                - "drivers": A JSON array of 3 brief strings explaining the specific fundamental/macro drivers of this sentiment.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0, # Zero temp for maximum consistency
                    response_format={ "type": "json_object" },
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": news_text}
                    ]
                )
                
                result = json.loads(response.choices[0].message.content)
                status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                
                st.divider()
                
                # --- 7. Results Visualization ---
                sentiment = result.get("sentiment", "Neutral")
                confidence = result.get("confidence", 50)
                
                # Dynamic UI based on sentiment
                if sentiment == "Bullish":
                    color_hex = "#10b981" # Emerald Green
                    icon = "🟢"
                elif sentiment == "Bearish":
                    color_hex = "#ef4444" # Red
                    icon = "🔴"
                else:
                    color_hex = "#64748b" # Slate Gray
                    icon = "⚪"
                
                # Top Metrics Level
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(label="Directional Signal", value=f"{icon} {sentiment}")
                with m2:
                    st.metric(label="Confidence Interval", value=f"{confidence}%")
                with m3:
                    st.metric(label="Processing Engine", value="GPT-4o-Mini")
                
                # Visual Confidence Bar
                st.caption("Confidence Meter")
                st.markdown(
                    f"""
                    <div style="width: 100%; background-color: #1e293b; border-radius: 5px; height: 8px; margin-top: -10px; margin-bottom: 20px;">
                        <div style="width: {confidence}%; background-color: {color_hex}; height: 8px; border-radius: 5px; transition: width 1s ease-in-out;"></div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Detailed Analysis Level
                col_left, col_right = st.columns([1.5, 1])
                
                with col_left:
                    st.markdown("### 📝 Executive Summary")
                    # Custom styled info box
                    st.markdown(f"""
                        <div style="background-color: #111827; border-left: 4px solid {color_hex}; padding: 15px; border-radius: 0px 8px 8px 0px; color: #e2e8f0; font-size: 1.05em; line-height: 1.5;">
                            {result.get('summary')}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_right:
                    st.markdown("### 🔑 Key Drivers")
                    for driver in result.get("drivers", []):
                        st.markdown(f"""
                            <div style="background-color: #1e293b; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; font-size: 0.95em; border: 1px solid #334155;">
                                • {driver}
                            </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                status.update(label="❌ Pipeline Failed", state="error", expanded=True)
                st.error(f"Error executing model: {str(e)}")

else:
    # Beautiful empty state if no API key is provided
    st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #111827; border: 1px dashed #334155; border-radius: 12px; margin-top: 20px;">
            <h3 style="color: #94a3b8;">System Locked 🔒</h3>
            <p style="color: #64748b;">Please initialize the system by providing an OpenAI API Key in the sidebar or via Streamlit Secrets.</p>
        </div>
    """, unsafe_allow_html=True)
