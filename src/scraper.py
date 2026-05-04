import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import zipfile
import re
from pathlib import Path
from dotenv import load_dotenv

import statsbombpy.sb as sb

load_dotenv()

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

QUALIFIED_48_TEAMS = [
    "United States", "Canada", "Mexico", "Argentina", "Brazil", "France", "England", "Spain", 
    "Germany", "Portugal", "Italy", "Netherlands", "Belgium", "Croatia", "Uruguay", "Colombia", 
    "Morocco", "Senegal", "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia", "Nigeria", 
    "Egypt", "Algeria", "Ivory Coast", "Tunisia", "Mali", "Cameroon", "Ecuador", "Peru", 
    "Chile", "Venezuela", "Serbia", "Switzerland", "Denmark", "Sweden", "Poland", "Wales", 
    "Ukraine", "Costa Rica", "Panama", "Jamaica", "New Zealand", "Qatar", "United Arab Emirates", "Iraq"
]

def fetch_kaggle_dataset(dataset_slug, output_filename):
    """
    Fetches a dataset using the Kaggle API. 
    Falls back to synthetic data if no Kaggle credentials are set or download fails.
    """
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        print(f"Kaggle credentials not found. Falling back to synthetic data for {dataset_slug}.")
        _generate_synthetic_kaggle_data(dataset_slug)
        return

    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

        # This requires KAGGLE_USERNAME and KAGGLE_KEY environment variables to be set
        # or ~/.kaggle/kaggle.json to exist
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {dataset_slug}...")
        
        # Download the dataset
        api.dataset_download_files(dataset_slug, path=DATA_DIR, unzip=True)
        print(f"Successfully downloaded {dataset_slug}")
    except Exception as e:
        print(f"Kaggle download failed for {dataset_slug}: {e}")
        print("Generating synthetic data as a # PLACEHOLDER instead.")
        _generate_synthetic_kaggle_data(dataset_slug)

def _generate_synthetic_kaggle_data(dataset_slug):
    """Fallback method to generate synthetic Kaggle data # PLACEHOLDER."""
    if "international-football-results" in dataset_slug:
        # Synthetic match results
        dates = pd.date_range(start="2010-01-01", end="2023-12-31", freq='W')
        data = []
        for d in dates:
            home = np.random.choice(QUALIFIED_48_TEAMS)
            away = np.random.choice(QUALIFIED_48_TEAMS)
            if home == away: continue
            data.append({
                "date": d,
                "home_team": home,
                "away_team": away,
                "home_score": np.random.poisson(1.5),
                "away_score": np.random.poisson(1.2),
                "tournament": "Friendly" if np.random.rand() > 0.3 else "FIFA World Cup qualification",
                "city": "Unknown",
                "country": home,
                "neutral": False
            })
        df = pd.DataFrame(data)
        df.to_csv(DATA_DIR / "results.csv", index=False)
    
    elif "fifa-ranking" in dataset_slug:
        # Synthetic FIFA rankings with realistic tiers
        REALISTIC_TIERS = {
            "Argentina": 2100, "France": 2050, "Spain": 2040, "England": 2030, 
            "Brazil": 2000, "Belgium": 1950, "Netherlands": 1940, "Portugal": 1930, 
            "Colombia": 1920, "Italy": 1910, "Uruguay": 1900, "Croatia": 1890, 
            "Germany": 1880, "Morocco": 1870, "Switzerland": 1860, "United States": 1850,
            "Senegal": 1840, "Japan": 1830, "Mexico": 1820, "Iran": 1810, "Denmark": 1800,
            "South Korea": 1790, "Australia": 1780
        }
        
        dates = pd.date_range(start="2010-01-01", end="2026-06-01", freq='ME')
        data = []
        for team in QUALIFIED_48_TEAMS:
            # Assign baseline points: exact if known, otherwise random lower tier
            base_points = REALISTIC_TIERS.get(team, np.random.uniform(1400, 1750))
            
            for d in dates:
                # Add slight random walk month over month
                base_points += np.random.normal(0, 2)
                data.append({
                    "rank_date": d,
                    "country_full": team,
                    "total_points": base_points
                })
        df = pd.DataFrame(data)
        # Calculate strict deterministic rank per date based on total_points
        df.sort_values(by=["rank_date", "total_points"], ascending=[True, False], inplace=True)
        df["rank"] = df.groupby("rank_date")["total_points"].rank(ascending=False).astype(int)
        df.to_csv(DATA_DIR / "fifa_ranking_synthetic.csv", index=False)


def fetch_statsbomb_data():
    """Fetches high-level stats from StatsBomb."""
    print("Fetching StatsBomb data...")
    try:
        # Attempt to get World Cup 2022 matches (competition 43, season 106)
        matches = sb.matches(competition_id=43, season_id=106)
        matches.to_csv(DATA_DIR / "statsbomb_matches.csv", index=False)
        print("Successfully fetched StatsBomb matches.")
    except Exception as e:
        print(f"StatsBomb fetch failed: {e}")
        print("Generating synthetic statsbomb data. # PLACEHOLDER")
        df = pd.DataFrame([{"home_team": t, "away_team": "Other", "home_score": 1, "away_score": 0} for t in QUALIFIED_48_TEAMS])
        df.to_csv(DATA_DIR / "statsbomb_matches.csv", index=False)


def fetch_transfermarkt_data():
    """
    Scrapes or synthesizes squad values from Transfermarkt.
    In a real system, you would iterate over team URLs and parse HTML.
    Transfermarkt heavily blocks automated requests without headers and proxying.
    We will provide a basic scraper framework but fallback to synthetic if blocked.
    """
    print("Fetching Transfermarkt squad values...")
    squad_data = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Try a sample URL to see if it works, otherwise generate synthetic
        url = "https://www.transfermarkt.com/weltmeisterschaft-2022/teilnehmer/pokalwettbewerb/WM22/saison_id/2021"
        response = requests.get(url, headers=headers, timeout=5)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        rows = soup.select("table.items tbody tr.even, table.items tbody tr.odd")
        
        if len(rows) == 0:
            raise ValueError("No rows found. Transfermarkt blocked the request or the layout changed.")

        for row in rows:
            tds = row.find_all('td')
            if len(tds) > 4:
                team_name = tds[1].get_text(strip=True)
                # This depends on the exact columns, sample extraction:
                squad_size = tds[3].get_text(strip=True)
                avg_age = tds[4].get_text(strip=True)
                market_value = tds[6].get_text(strip=True)
                
                # Basic parsing
                market_value_num = 0.0
                if "bn" in market_value:
                    market_value_num = float(market_value.replace("€", "").replace("bn", "").strip()) * 1000
                elif "m" in market_value:
                    market_value_num = float(market_value.replace("€", "").replace("m", "").strip())
                
                squad_data.append({
                    "team": team_name,
                    "squad_size": int(squad_size) if squad_size.isdigit() else 26,
                    "avg_age": float(avg_age.replace(',', '.')) if avg_age else 26.0,
                    "market_value_m": market_value_num
                })
        print("Successfully fetched Transfermarkt data.")
    except Exception as e:
        print(f"Transfermarkt scraper failed/blocked: {e}")
        print("Generating synthetic Transfermarkt data. # PLACEHOLDER")
        for team in QUALIFIED_48_TEAMS:
            squad_data.append({
                "team": team,
                "squad_size": np.random.randint(23, 27),
                "avg_age": np.round(np.random.uniform(24.0, 29.5), 1),
                "market_value_m": np.round(np.random.uniform(20.0, 1000.0), 1),
                "top_5_league_players": np.random.randint(0, 20)
            })
            
    df = pd.DataFrame(squad_data)
    df.to_csv(DATA_DIR / "squad_values.csv", index=False)

def run():
    print("Starting data gathering pipeline...")
    
    # 1. Matches dataset
    fetch_kaggle_dataset("martj42/international-football-results-and-goalscorers", "results.csv")
    
    # 2. FIFA Ranking dataset
    # We will just fetch the dataset name, often the downloaded file has a specific name
    fetch_kaggle_dataset("adityajn105/fifa-ranking", "fifa_ranking.csv")
    
    # 3. StatsBomb
    fetch_statsbomb_data()
    
    # 4. Transfermarkt
    fetch_transfermarkt_data()

    print("Data gathering complete.")

if __name__ == "__main__":
    run()
