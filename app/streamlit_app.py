import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pickle
from pathlib import Path
import sys
import os

# Add src to Python path so we can import from it
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.explain import generate_narrative

st.set_page_config(page_title="WC2026 AI Scout", layout="wide", page_icon="⚽")

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: 600;
        color: #31333F;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚽ WC2026 Prophet: The AI Football Scout")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

@st.cache_data
def load_cache():
    cache_path = MODEL_DIR / "inference_cache.pkl"
    if not cache_path.exists():
        return None, None
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data["scores"], data["shap"]

scores_dict, shap_dict = load_cache()

if not scores_dict:
    st.error("Model cache not found! Please run the data gathering and modeling pipelines first.")
    st.stop()

# --- Sidebar (no API config section) ---
st.sidebar.title("Navigation")
teams = sorted(list(scores_dict.keys()))
selected_team = st.sidebar.selectbox("Select a Team to Analyze", teams)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 System Reasoning")
st.sidebar.info(
    "**How does it work?**\n\n"
    "This AI system is an ensemble of 3 specialized ML models trained on historical World Cup and international match data:\n\n"
    "📊 **The Statistician** (XGBoost): Analyzes strict win rates, goals, and FIFA rank trajectories.\n\n"
    "🕵️‍♂️ **The Scout** (Random Forest): Evaluates squad market value, total age, and player depth.\n\n"
    "🧠 **The Psychologist** (Logistic Regression): Looks at soft signals like coach tenure length and host advantages."
)

st.write(f"### Intelligent Profile: **{selected_team}**")

# Context
with st.expander("📖 Click here to understand what these metrics mean", expanded=True):
    st.markdown("""
    - **Underperformer Risk**: The likelihood the team will exit the tournament earlier than their pure FIFA ranking suggests. Lower is better.
    - **Underdog Score**: The probability this team will significantly outperform their historical seeding/expectations. Higher is better.
    - **Dark Horse Probability**: The chance of this team making a deep, unexpected championship run against elite opponents. Higher is better.
    """)

scores = scores_dict[selected_team]
shap_top5 = shap_dict[selected_team]

col1, col2, col3 = st.columns(3)

def create_gauge(val, title, invert_colors=False):
    # Soothing colors: Blue -> Yellow -> Coral
    if invert_colors:
        colors = ["#7FB3FF", "#FFE066", "#FF7F7F"]
    else:
        colors = ["#FF7F7F", "#FFE066", "#7FB3FF"]
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 33], 'color': colors[0]},
                {'range': [33, 66], 'color': colors[1]},
                {'range': [66, 100], 'color': colors[2]}
            ],
            'threshold': {
                'line': {'color': "#2C3E50", 'width': 4},
                'thickness': 0.75,
                'value': val
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

with col1:
    st.plotly_chart(create_gauge(scores['Underperformer Risk'], "Underperformer Risk", invert_colors=True), use_container_width=True)

with col2:
    st.plotly_chart(create_gauge(scores['Underdog Score'], "Underdog Score", invert_colors=False), use_container_width=True)

with col3:
    st.plotly_chart(create_gauge(scores['Dark Horse Probability'], "Dark Horse Probability", invert_colors=False), use_container_width=True)

st.markdown("---")
st.write("### 🔍 Model Explainability: Feature Impact")
st.write("The chart below illustrates the top 5 statistical features influencing the **Underdog Score** for this team. "
         "Features extending **right (teal)** push the team toward being a surprise underdog. "
         "Features extending **left (coral)** indicate the team is too established to surprise.")

shap_df = pd.DataFrame(shap_top5)
# Eye soothing colors (Teal for positive, Soft Coral for negative)
shap_df['color'] = shap_df['direction'].map({'positive': '#20B2AA', 'negative': '#F08080'})

fig_shap = px.bar(shap_df, x='shap_value', y='feature', orientation='h', 
                  color='direction', color_discrete_map={'positive': '#20B2AA', 'negative': '#F08080'},
                  labels={'shap_value': 'Model Impact Magnitude', 'feature': 'Dataset Feature'},
                  title="SHAP Values Interpretation")
fig_shap.update_layout(
    yaxis={'categoryorder':'total ascending'},
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("---")
st.write("### ✨ AI Scout's Final Verdict")
st.write("Merge the hard statistics with conversational AI to get a synthesized scouting report.")

# Detect whether a key is available from st.secrets or environment
def _get_configured_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY", None)
        if k and k != "your_gemini_api_key_here":
            return k
    except Exception:
        pass
    k = os.environ.get("GEMINI_API_KEY", None)
    if k and k != "your_gemini_api_key_here":
        return k
    return None

configured_key = _get_configured_key()

if not configured_key:
    # No key found — show a clean inline key entry (only appears on Cloud without secrets)
    st.info("🔑 To generate a scouting report, enter your [Google AI Studio](https://aistudio.google.com/apikey) key below (it's free & takes 10s).")
    inline_key = st.text_input("Gemini API Key", type="password", label_visibility="collapsed", placeholder="Paste your Gemini API key here...")
    if st.button("💬 Generate Gemini AI Scouting Report", disabled=not inline_key):
        with st.spinner("Consulting Lastor - The Last Dance (Gemini AI)..."):
            report = generate_narrative(selected_team, scores, shap_top5, api_key_override=inline_key)
        st.success("Report Generated!")
        st.markdown(f"**Verdict from Lastor - The Last Dance on {selected_team}:**")
        st.info(report)
else:
    # Key is available from server secrets/env — works seamlessly with no user input
    if st.button("💬 Generate Gemini AI Scouting Report"):
        with st.spinner("Consulting Lastor - The Last Dance (Gemini AI)..."):
            report = generate_narrative(selected_team, scores, shap_top5)
        st.success("Report Generated!")
        st.markdown(f"**Verdict from Lastor - The Last Dance on {selected_team}:**")
        st.info(report)
