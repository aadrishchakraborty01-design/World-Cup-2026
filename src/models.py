import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class WCEmsemblePredictor:
    def __init__(self):
        # Base estimators wrapped in MultiOutputClassifier for 3 targets
        self.statistician = MultiOutputClassifier(xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42))
        self.scout = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        self.psychologist = MultiOutputClassifier(LogisticRegression(max_iter=500, random_state=42))
        
        self.stat_features = [
            'win_rate_12m', 'goals_scored_12m', 'goals_conceded_12m', 'clean_sheet_rate_12m', 
            'win_rate_24m', 'goals_scored_24m', 'goals_conceded_24m', 'clean_sheet_rate_24m', 
            'fifa_rank', 'fifa_rank_delta_12m', 'win_rate_vs_top20'
        ]
        
        self.scout_features = [
            'market_value_m', 'squad_avg_age', 'top_5_league_players', 'xg_for', 'xga_against'
        ]
        
        # Soft features including all generated one-hot encoded cols
        self.psych_features_base = [
            'is_host', 'coach_tenure_months', 'tournament_experience'
        ]
        
        self.targets = ['label_underperformer_risk', 'label_underdog_score', 'label_dark_horse']
        self.all_psych_features = []
        
    def fit(self, df_train):
        # Filter purely to training years
        df_train = df_train.dropna(subset=self.targets)
        
        # Determine actual psych features dynamically based on one-hot columns in df
        conf_cols = [c for c in df_train.columns if c.startswith('conf_')]
        self.all_psych_features = self.psych_features_base + conf_cols
        
        X_stat = df_train[self.stat_features]
        X_scout = df_train[self.scout_features]
        # Fill missing values just in case
        X_psych = df_train[self.all_psych_features].fillna(0)
        
        y = df_train[self.targets]
        
        print("Training The Statistician (XGBoost)...")
        self.statistician.fit(X_stat, y)
        
        print("Training The Scout (Random Forest)...")
        self.scout.fit(X_scout, y)
        
        print("Training The Psychologist (Logistic Regression)...")
        self.psychologist.fit(X_psych, y)
        
    def predict_proba_ensemble(self, df_inference):
        # Ensure one-hot columns exist
        for f in self.all_psych_features:
            if f not in df_inference.columns:
                df_inference[f] = 0
                
        X_stat = df_inference[self.stat_features]
        X_scout = df_inference[self.scout_features]
        X_psych = df_inference[self.all_psych_features].fillna(0)
        
        # Each model returns a list of arrays (one for each target).
        # We need to extract the positive class probability [:, 1] for each target.
        def extract_probs(model, X):
            preds = model.predict_proba(X)
            out = {}
            for i, target in enumerate(self.targets):
                arr = preds[i]
                if arr.shape[1] == 1:
                    out[target] = np.zeros(arr.shape[0])
                else:
                    out[target] = arr[:, 1]
            return out
            
        prob_stat = extract_probs(self.statistician, X_stat)
        prob_scout = extract_probs(self.scout, X_scout)
        prob_psych = extract_probs(self.psychologist, X_psych)
        
        # Weighted ensemble (40% Stat, 35% Scout, 25% Psych)
        ensemble_probs = {}
        for target in self.targets:
            ensemble_probs[target] = (
                0.40 * prob_stat[target] + 
                0.35 * prob_scout[target] + 
                0.25 * prob_psych[target]
            ) * 100 # Scale to 0-100
            
        return ensemble_probs

    def save(self):
        with open(MODEL_DIR / "ensemble_model.pkl", "wb") as f:
            pickle.dump(self, f)

def run():
    print("Loading engineered features...")
    df = pd.read_csv(PROC_DIR / "model_features.csv")
    
    df_train_2018 = df[df['year'] == 2018]
    df_train_2022 = df[df['year'] == 2022]
    df_inference_2026 = df[df['year'] == 2026]
    
    # Backtesting definition
    # Train pre-2018, Test 2018
    # Actually, we synthesized 1 dataset per team per year, so we can't easily cross-val 
    # unless we use 2018 to predict 2022. But MVP says "Backtest by training only on data up to 2017, then scoring predictions against 2018 World Cup actual results."
    # Since we built rows indexed by (team, year: 2018), those *are* the pre-2018 features!
    print("Backtesting on 2018...")
    # Train on 2018 (pretend it's 2014 or something if we had it. Since we only have 2018, 2022, we will train on 2018 target and evaluate on 2018? No, train on 2018, test on 2022).
    # Since we need output, let's train on 2018 and check 2022
    predictor = WCEmsemblePredictor()
    predictor.fit(df_train_2018)
    
    # Evaluate on 2022
    y_true_2022 = df_train_2022[predictor.targets]
    y_pred_2022 = predictor.predict_proba_ensemble(df_train_2022)
    
    from sklearn.metrics import brier_score_loss, recall_score
    
    for t in predictor.targets:
        if not y_true_2022[t].isnull().all() and y_true_2022[t].sum() > 0:
            brier = brier_score_loss(y_true_2022[t], y_pred_2022[t]/100.0)
            # Recall for top-N
            # Pretend top-8 highest prob are called positive
            threshold = pd.Series(y_pred_2022[t]).nlargest(8).iloc[-1]
            binary_preds = (y_pred_2022[t] >= threshold).astype(int)
            recall = recall_score(y_true_2022[t], binary_preds)
            print(f"Validation 2022 - {t}: Brier={brier:.3f}, Top-8 Recall={recall:.3f}")
            
    # Final model for 2026 inference trained on all past data
    print("Training final model on all historical data (2018 + 2022)...")
    final_predictor = WCEmsemblePredictor()
    final_predictor.fit(pd.concat([df_train_2018, df_train_2022]))
    final_predictor.save()
    print("Final model saved to models/ensemble_model.pkl")

if __name__ == "__main__":
    run()
