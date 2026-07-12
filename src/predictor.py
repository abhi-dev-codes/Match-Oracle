# src/predictor.py
import joblib
import pandas as pd

def get_latest_rank(rankings_df, team):
    team_rows = rankings_df[rankings_df["country_full"] == team]
    return team_rows.sort_values("rank_date").iloc[-1]["rank"]

def predict_outcome(model_path, rankings_df, home_team, away_team, neutral=0):
    model = joblib.load(model_path)
    scaler = joblib.load("models/scaler.pkl")

    home_rank = get_latest_rank(rankings_df, home_team)
    away_rank = get_latest_rank(rankings_df, away_team)
    rank_diff = away_rank - home_rank

    X_input = pd.DataFrame([[rank_diff, neutral]], columns=["rank_diff", "neutral_venue"])
    X_scaled = scaler.transform(X_input)
    probs = model.predict_proba(X_scaled)[0]
    return {"away_win": probs[0], "draw": probs[1], "home_win": probs[2]}