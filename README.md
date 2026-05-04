# WC2026 Predictor

An ML-powered prediction system and Streamlit dashboard that predicts Underperformer Risk, Underdog Score, and Dark Horse Probability for all 48 teams in the 2026 World Cup.

## Project Structure
- `data/raw/`: Raw downloaded datasets
- `data/processed/`: Engineered and cleaned features
- `notebooks/`: Jupyter notebooks for EDA and experimentation
- `src/`: Core Python modules for scraping, feature engineering, modeling, and explaining
- `models/`: Saved `.pkl` model files
- `app/`: Streamlit dashboard

## Setup Instructions

1. **Install Dependencies**
   It's recommended to create a virtual environment first.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
   You will need:
   - `ANTHROPIC_API_KEY`: For generating scouting reports via Claude API.
   - `KAGGLE_USERNAME` & `KAGGLE_KEY`: For downloading Kaggle datasets.

3. **Kaggle API Setup**
   - Create an account on Kaggle.
   - Go to "Settings" -> "API" and click "Create New Token". This downloads a `kaggle.json` file.
   - You can place the contents (`username` and `key`) into the `.env` file, OR place `kaggle.json` into `~/.kaggle/kaggle.json`.

## Running the Project

First, gather data, build features, train models, and generate explainability outputs:
```bash
python src/scraper.py
python src/features.py
python src/models.py
python src/explain.py
```

Then, run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```
