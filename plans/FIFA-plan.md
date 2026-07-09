Here is the complete blueprint, merging the 3-day Machine Learning Development Life Cycle (MDLC) with the exact folder structure, file explanations, code snippets, and dataset links.

Organizing a project like this is exactly what catches the eye of hackathon judges. It shows you understand modular software engineering, separating your data pipeline from your front-end logic, rather than stuffing everything into a single script.

### 🏷️ Repository Name Suggestions

* **Professional:** `fifa-match-outcome-predictor`, `international-football-ml-pipeline`, `ai-scout-report-generator`
* **Snappy/Techy:** `PitchPredictor`, `ScoutAI`, `MatchMind`, `GoalGraph`
* **One-Word:** `Gaffer`, `Scout`, `Fixture`

### 💾 The Kaggle Datasets

Download these directly into your `data/raw/` folder.

1. **Match History:** [International football results from 1872 to 2024 (by Mart Jürisoo)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
2. **FIFA Rankings:** [FIFA World Ranking 1992-2024](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)

---

### 🗓️ Day 1: Data Engineering (MDLC: Data Prep)

When solving daily algorithmic problems in Java, you are likely used to writing `for` loops to iterate through arrays. In Python data science, you must shift your mindset to "vectorization"—applying operations to entire columns at once using Pandas.

#### `notebooks/01_data_prep.ipynb`

* **Why & What:** A Jupyter Notebook acting as your scratchpad.
* **How & Use:** Use this to visually experiment with Pandas `merge()` and `apply()` functions on the Kaggle CSVs without breaking your main application.

#### `src/data_processor.py`

* **Why & What:** The formal script that handles your data cleaning. It ensures your application always processes data consistently.
* **How & Use:** It loads the raw Kaggle datasets, calculates the Rank Difference, creates the target `Outcome` variable, and exports a clean `training_data.csv` to your `data/processed/` folder.

```python
import pandas as pd

def process_football_data(matches_path, rankings_path, output_path):
    # Load raw data
    df_matches = pd.read_csv(matches_path)
    df_rankings = pd.read_csv(rankings_path)
    
    # Filter for modern matches (e.g., year > 2000)
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    df_matches = df_matches[df_matches['date'].dt.year >= 2000]
    
    # Feature Engineering: Create Target Variable (1=Win, 0=Draw, -1=Loss)
    df_matches['Outcome'] = df_matches.apply(
        lambda x: 1 if x['home_score'] > x['away_score'] else (0 if x['home_score'] == x['away_score'] else -1), 
        axis=1
    )
    
    # In a real scenario, you would merge df_rankings here to get 'Rank_Diff'
    df_matches['Rank_Diff'] = 5 # Placeholder for merged ranking logic
    
    # Save the processed data
    df_matches[['home_team', 'away_team', 'Rank_Diff', 'Outcome']].to_csv(output_path, index=False)
    print("Data processed and saved!")

if __name__ == "__main__":
    process_football_data('../data/raw/results.csv', '../data/raw/fifa_ranking.csv', '../data/processed/training_data.csv')

```

---

### 🗓️ Day 2: Modeling & AI (MDLC: Model Training)

#### `src/model_trainer.py`

* **Why & What:** This script trains the Random Forest algorithm.
* **How & Use:** It reads your cleaned data, learns the patterns, and exports the intelligence as a `.pkl` file so your web app doesn't have to retrain the math every time a user refreshes the page.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model(data_path, model_output_path):
    df = pd.read_csv(data_path)
    
    # Define Features (X) and Target (y)
    X = df[['Rank_Diff']] # Add more features like Is_Friendly later
    y = df['Outcome']
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train_and_save_model('../data/processed/training_data.csv', '../models/rf_classifier.pkl')

```

#### `context/team_news.csv`

* **Why & What:** A manually curated database representing your "live" football knowledge.
* **How & Use:** It replaces the need for complex web scraping. Format it simply:
```csv
Team,Recent_Form
Spain,Unbeaten in last 5 matches. Star midfielder injured.
Brazil,Struggling to score away from home. Manager under pressure.

```



#### `src/rag_engine.py`

* **Why & What:** The bridge between your data and the Gen AI API.
* **How & Use:** It searches `team_news.csv` for the selected teams and constructs a highly specific prompt for the LLM.

```python
import pandas as pd
import google.generativeai as genai
import os

# Ensure your .env loads the API key before this runs
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def generate_scout_report(home_team, away_team, win_prob):
    df_news = pd.read_csv('context/team_news.csv')
    
    # Retrieve context (Simple RAG)
    home_context = df_news[df_news['Team'] == home_team]['Recent_Form'].values
    away_context = df_news[df_news['Team'] == away_team]['Recent_Form'].values
    
    context_str = f"{home_team} news: {home_context}. {away_team} news: {away_context}."
    
    prompt = f"""
    You are an expert football scout. {home_team} is playing {away_team}. 
    Our ML model gives {home_team} a {win_prob}% chance of winning. 
    Context: {context_str}
    Write a punchy, 3-sentence tactical preview based strictly on this context and probability.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

```

---

### 🗓️ Day 3: The Front-End (MDLC: Deployment)

#### `.env` and `requirements.txt`

* **Why & What:** Environment controls.
* **How & Use:** `.env` safely holds your `GEMINI_API_KEY="your_key_here"` so it isn't pushed to GitHub. `requirements.txt` lists your libraries (`streamlit`, `pandas`, `scikit-learn`, `google-generativeai`) so others can install them instantly.

#### `app.py`

* **Why & What:** The Streamlit dashboard. This is the only file the user interacts with.
* **How & Use:** It imports your saved model and your RAG function, providing dropdowns and a clean visual interface.

```python
import streamlit as st
import joblib
from src.rag_engine import generate_scout_report

# Load the pre-trained model
model = joblib.load('models/rf_classifier.pkl')

st.title("⚽ AI Football Match Predictor")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", ["Spain", "Brazil", "France"])
with col2:
    away_team = st.selectbox("Away Team", ["Spain", "Brazil", "France"])

# Mock function: In reality, you'd calculate this from the live FIFA rankings CSV
mock_rank_diff = 5 

if st.button("Predict Outcome"):
    # 1. Get ML Probabilities
    probs = model.predict_proba([[mock_rank_diff]])[0]
    win_prob = round(probs[2] * 100, 1) # Index 2 is usually the '1' (Win) class
    
    st.subheader(f"{home_team} Win Probability: {win_prob}%")
    st.progress(int(win_prob))
    
    # 2. Get AI Scout Report
    st.subheader("🤖 AI Scout Report")
    with st.spinner("Analyzing team form..."):
        report = generate_scout_report(home_team, away_team, win_prob)
        st.write(report)

```