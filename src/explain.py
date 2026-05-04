import pandas as pd
import numpy as np
import shap
import pickle
from pathlib import Path
import os
import sys
from google import genai
from dotenv import load_dotenv

class GeminiKeyError(Exception):
    """Raised when the Gemini API returns a 403 / leaked key error."""
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.models as models
import __main__
__main__.WCEmsemblePredictor = models.WCEmsemblePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

def compute_all_shap():
    """
    Computes and caches top 5 SHAP values for the 2026 inference teams.
    We will explain The Statistician's XGBoost model for simplicity.
    """
    with open(MODEL_DIR / "ensemble_model.pkl", "rb") as f:
        ensemble = pickle.load(f)
        
    df = pd.read_csv(PROC_DIR / "model_features.csv")
    df_2026 = df[df['year'] == 2026].reset_index(drop=True)
    
    # We explain target 1 (Underdog Score) — gives a natural mix of positive/negative SHAP values
    # Positive = feature pushes team toward being an underdog surprise
    # Negative = feature says they're too established to surprise
    xgb_estimator = ensemble.statistician.estimators_[1]
    X_stat = df_2026[ensemble.stat_features]
    
    explainer = shap.TreeExplainer(xgb_estimator)
    shap_values = explainer.shap_values(X_stat)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Extract top 5 features per team
    top_5_shap_dict = {}
    
    for i, row in df_2026.iterrows():
        team = row['team']
        vals = shap_values[i, :]
        feature_names = ensemble.stat_features
        
        # Sort features by absolute SHAP value impact
        impact = np.abs(vals)
        top_indices = np.argsort(impact)[::-1][:5]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": feature_names[idx],
                "shap_value": float(vals[idx]),
                "direction": "positive" if vals[idx] > 0 else "negative"
            })
            
        top_5_shap_dict[team] = top_features
        
    # Also pre-calculate scores for all teams to make app faster
    scores_dict = {}
    preds = ensemble.predict_proba_ensemble(df_2026)
    for i, row in df_2026.iterrows():
        team = row['team']
        scores_dict[team] = {
            "Underperformer Risk": float(preds['label_underperformer_risk'][i]),
            "Underdog Score": float(preds['label_underdog_score'][i]),
            "Dark Horse Probability": float(preds['label_dark_horse'][i]),
        }
        
    # Save cache
    with open(MODEL_DIR / "inference_cache.pkl", "wb") as f:
        pickle.dump({"shap": top_5_shap_dict, "scores": scores_dict}, f)
        
    return top_5_shap_dict, scores_dict

def generate_narrative(team_name, scores, shap_top5, api_key_override=None):
    """
    Calls the Google Gemini API to generate a scouting report.
    Key priority: UI override → st.secrets → .env (local)
    """
    if api_key_override and api_key_override.strip():
        api_key = api_key_override.strip()
    else:
        # Try Streamlit Cloud Secrets first (production), then fall back to .env (local)
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            api_key = None
        
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key or api_key == "your_gemini_api_key_here":
        return ("[MVP PLACEHOLDER NARRATIVE]\nGemini API key not found. Please provide an API key in the sidebar."
                " This team shows strong indicators in recent scoring history but struggles against top 20 opponents. "
                "The psychological profile suggests high volatility. Proceed with caution when betting.")
        
    # Format SHAP features
    shap_text = ", ".join([f"{f['feature']} ({f['direction']} impact: {f['shap_value']:.2f})" for f in shap_top5])
    
    prompt = f"""
    You are an expert, highly analytical football pundit and scout AI module called "Lastor - The Last Dance".
    Analyze the World Cup 2026 prospects for {team_name}.
    
    The ML ensemble generated these scores (0-100):
    - Underperformer Risk: {scores['Underperformer Risk']:.1f}
    - Underdog Score: {scores['Underdog Score']:.1f}
    - Dark Horse Probability: {scores['Dark Horse Probability']:.1f}
    
    The top 5 statistical features driving their model prediction are:
    {shap_text}
    
    Write a highly engaging scouting report in Markdown format using this exact structure:
    ### 🏆 Lastor's Statistical Verdict
    (One punchy, bolded sentence summarizing what the model metrics indicate, without making absolute future guarantees)

    ### 🔥 Tactical Data Strengths
    (2 bullet points heavily referencing the positive SHAP values)

    ### ⚠️ Statistical Red Flags
    (2 bullet points heavily referencing the negative SHAP values or the Underperformer risk)

    Adopt an enthusiastic persona but strictly ground all your observations in the provided SHAP data and model scores. You MUST speak from the data. DO NOT make definitive personal predictions about tournament outcomes, only explain how the model analyzed the team. Use appropriate emojis. 
    """
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        err = str(e)
        # Raise a typed exception so the caller can show a key-entry prompt
        if "403" in err or "PERMISSION_DENIED" in err or "leaked" in err.lower():
            raise GeminiKeyError(err)
        return f"Failed to generate narrative via API: {err}"

if __name__ == "__main__":
    print("Computing SHAP values for all 2026 teams...")
    compute_all_shap()
    print("Saved SHAP cache to models/inference_cache.pkl")
