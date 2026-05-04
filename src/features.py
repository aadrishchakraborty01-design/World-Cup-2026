import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

QUALIFIED_48_TEAMS = [
    "United States", "Canada", "Mexico", "Argentina", "Brazil", "France", "England", "Spain", 
    "Germany", "Portugal", "Italy", "Netherlands", "Belgium", "Croatia", "Uruguay", "Colombia", 
    "Morocco", "Senegal", "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia", "Nigeria", 
    "Egypt", "Algeria", "Ivory Coast", "Tunisia", "Mali", "Cameroon", "Ecuador", "Peru", 
    "Chile", "Venezuela", "Serbia", "Switzerland", "Denmark", "Sweden", "Poland", "Wales", 
    "Ukraine", "Costa Rica", "Panama", "Jamaica", "New Zealand", "Qatar", "United Arab Emirates", "Iraq"
]

CONFEDERATIONS = {
    # Approx mapping for MVP
    "UEFA": ["France", "England", "Spain", "Germany", "Portugal", "Italy", "Netherlands", "Belgium", "Croatia", "Serbia", "Switzerland", "Denmark", "Sweden", "Poland", "Wales", "Ukraine"],
    "CONMEBOL": ["Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Peru", "Chile", "Venezuela"],
    "CONCACAF": ["United States", "Canada", "Mexico", "Costa Rica", "Panama", "Jamaica"],
    "CAF": ["Morocco", "Senegal", "Nigeria", "Egypt", "Algeria", "Ivory Coast", "Tunisia", "Mali", "Cameroon"],
    "AFC": ["Japan", "South Korea", "Iran", "Australia", "Saudi Arabia", "Qatar", "United Arab Emirates", "Iraq"],
    "OFC": ["New Zealand"]
}

def get_confederation(team_name):
    for confed, teams in CONFEDERATIONS.items():
        if team_name in teams: return confed
    return "UNKNOWN"

def build_features_for_year(year, results_df, ranks_df, squad_df, sb_df):
    """
    Builds the feature matrix for a specific World Cup year (2018, 2022, 2026).
    """
    # Cutoff date for features
    if year == 2018: cutoff_date = pd.to_datetime("2018-06-01")
    elif year == 2022: cutoff_date = pd.to_datetime("2022-11-01")
    else: cutoff_date = pd.to_datetime("2026-06-01")

    hist_results = results_df[results_df['date'] < cutoff_date]
    hist_ranks = ranks_df[ranks_df['rank_date'] < cutoff_date]

    rows = []
    
    for team in QUALIFIED_48_TEAMS:
        row = {"team": team, "year": year}

        # ----- 1. The Statistician Features (Hard Stats) -----
        # Matches in last 12 / 24 months
        date_12m = cutoff_date - pd.DateOffset(months=12)
        date_24m = cutoff_date - pd.DateOffset(months=24)
        
        team_matches = hist_results[(hist_results['home_team'] == team) | (hist_results['away_team'] == team)]
        
        m_12 = team_matches[team_matches['date'] >= date_12m]
        m_24 = team_matches[team_matches['date'] >= date_24m]
        
        def compute_stats(df, t):
            if len(df) == 0: return {"win_rate": 0.5, "goals_scored": 1.0, "goals_conceded": 1.0, "clean_sheets": 0.0}
            wins, gs, gc, cs = 0, 0, 0, 0
            for _, r in df.iterrows():
                is_home = (r['home_team'] == t)
                t_score = r['home_score'] if is_home else r['away_score']
                o_score = r['away_score'] if is_home else r['home_score']
                
                if t_score > o_score: wins += 1
                gs += t_score
                gc += o_score
                if o_score == 0: cs += 1
                
            return {
                "win_rate": wins / len(df),
                "goals_scored": gs / len(df),
                "goals_conceded": gc / len(df),
                "clean_sheets": cs / len(df)
            }
            
        stats_12 = compute_stats(m_12, team)
        stats_24 = compute_stats(m_24, team)
        
        row['win_rate_12m'] = stats_12['win_rate']
        row['goals_scored_12m'] = stats_12['goals_scored']
        row['goals_conceded_12m'] = stats_12['goals_conceded']
        row['clean_sheet_rate_12m'] = stats_12['clean_sheets']
        
        row['win_rate_24m'] = stats_24['win_rate']
        row['goals_scored_24m'] = stats_24['goals_scored']
        row['goals_conceded_24m'] = stats_24['goals_conceded']
        row['clean_sheet_rate_24m'] = stats_24['clean_sheets']
        
        # FIFA Rankings
        team_ranks = hist_ranks[hist_ranks['country_full'] == team].sort_values('rank_date')
        if not team_ranks.empty:
            curr_rank = team_ranks.iloc[-1]['rank']
            # rank 12 months ago roughly
            rank_12m_ago = team_ranks[team_ranks['rank_date'] <= date_12m]
            if not rank_12m_ago.empty:
                prev_rank = rank_12m_ago.iloc[-1]['rank']
            else:
                prev_rank = curr_rank
            rank_delta = prev_rank - curr_rank # positive means improved
        else:
            curr_rank = 50 # Default middle rank
            rank_delta = 0
            
        row['fifa_rank'] = curr_rank
        row['fifa_rank_delta_12m'] = rank_delta
        
        # Performance vs Top 20 (Synthetic/Heuristic for MVP)
        row['win_rate_vs_top20'] = max(0.0, stats_24['win_rate'] - 0.2 + np.random.uniform(-0.1, 0.1))

        # ----- 2. The Scout Features (Squad Data) -----
        # In a real app we'd load correct year squad. For MVP, we align/synthesize.
        sq_row = squad_df[squad_df['team'] == team]
        if not sq_row.empty:
            market_val = sq_row.iloc[0]['market_value_m']
            avg_age = sq_row.iloc[0]['avg_age']
            top_5_players = sq_row.iloc[0].get('top_5_league_players', np.random.randint(0, 15))
        else:
            market_val = 150.0 + np.random.uniform(-50, 50)
            avg_age = 26.5 + np.random.uniform(-1, 1)
            top_5_players = np.random.randint(0, 15)
            
        row['market_value_m'] = market_val * (1 if year == 2026 else (0.8 if year==2022 else 0.6)) 
        row['squad_avg_age'] = avg_age
        row['top_5_league_players'] = top_5_players
        row['xg_for'] = row['goals_scored_12m'] * np.random.uniform(0.9, 1.1)
        row['xga_against'] = row['goals_conceded_12m'] * np.random.uniform(0.9, 1.1)

        # ----- 3. The Psychologist Features (Soft Signals) -----
        row['is_host'] = 1 if (year == 2026 and team in ["United States", "Canada", "Mexico"]) or \
                              (year == 2022 and team == "Qatar") else 0
        
        row['coach_tenure_months'] = np.random.randint(6, 60) # PLACEHOLDER
        row['tournament_experience'] = np.random.randint(15, 60) # PLACEHOLDER
        row['confederation'] = get_confederation(team)
        
        # ----- TARGET LABELS FOR TRAINING (Only meaningful for past years) -----
        if year in [2018, 2022]:
            # Synthesize targets based on true ranking heuristically:
            # Underperformer if high rank but bad heuristic score
            # Underdog if low rank but good heuristic score
            base_score = row['win_rate_24m'] * 2 - (row['fifa_rank'] / 100)
            row['label_underperformer_risk'] = 1 if (row['fifa_rank'] < 15 and base_score < 0.5) else 0
            row['label_underdog_score'] = 1 if (row['fifa_rank'] > 25 and base_score > 0.8) else 0
            row['label_dark_horse'] = 1 if (row['fifa_rank'] > 20 and base_score > 1.0) else 0
            # Add some noise to labels
            if np.random.rand() < 0.1: row['label_underperformer_risk'] = 1 - row['label_underperformer_risk']
            if np.random.rand() < 0.1: row['label_underdog_score'] = 1 - row['label_underdog_score']
            if np.random.rand() < 0.1: row['label_dark_horse'] = 1 - row['label_dark_horse']
        else:
            row['label_underperformer_risk'] = np.nan
            row['label_underdog_score'] = np.nan
            row['label_dark_horse'] = np.nan

        rows.append(row)
        
    return pd.DataFrame(rows)

def run():
    print("Loading raw data...")
    # Read raw datasets (or create empty if they don't exist, though scraper should have made them)
    try:
        results_df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=['date'])
    except:
        print("Warning: results.csv missing, creating empty df.")
        results_df = pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
        
    try:
        ranks_df = pd.read_csv(DATA_DIR / "fifa_ranking.csv", parse_dates=['rank_date'])
    except:
        # Check if there is another file like fifa_ranking-202X.csv
        try:
            fk = list(DATA_DIR.glob("fifa_ranking*.csv"))[0]
            ranks_df = pd.read_csv(fk, parse_dates=['rank_date'])
        except:
            print("Warning: fifa ranking missing, creating empty df.")
            ranks_df = pd.DataFrame(columns=['rank_date', 'country_full', 'rank'])

    try:
        squad_df = pd.read_csv(DATA_DIR / "squad_values.csv")
    except:
        squad_df = pd.DataFrame(columns=['team', 'market_value_m', 'avg_age'])

    try:
        sb_df = pd.read_csv(DATA_DIR / "statsbomb_matches.csv")
    except:
        sb_df = pd.DataFrame()

    print("Building features for 2018 (Train), 2022 (Train/Val), and 2026 (Inference)...")
    df_2018 = build_features_for_year(2018, results_df, ranks_df, squad_df, sb_df)
    df_2022 = build_features_for_year(2022, results_df, ranks_df, squad_df, sb_df)
    df_2026 = build_features_for_year(2026, results_df, ranks_df, squad_df, sb_df)
    
    # Combine everything
    final_df = pd.concat([df_2018, df_2022, df_2026], ignore_index=True)
    
    # One-hot encode the confederation
    final_df = pd.get_dummies(final_df, columns=['confederation'], prefix='conf')
    
    # Save the processed dataset
    output_path = PROC_DIR / "model_features.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved to {output_path}")

if __name__ == "__main__":
    run()
