# src/predictor.py
import joblib
import pandas as pd
from datetime import datetime, timedelta

def get_latest_rank(rankings_df, team):
    team_rows = rankings_df[rankings_df["country_full"] == team]
    return team_rows.sort_values("rank_date").iloc[-1]["total_points"]

def compute_h2h_at_inference(results_df, home_team, away_team, window_years=4):
    """Compute H2H features for a specific matchup at inference time.

    Looks at all matches between home_team and away_team (in either order)
    within the last `window_years` years from today.
    Returns (h2h_home_win_rate, h2h_match_count).
    """
    now = pd.Timestamp(datetime.now())
    cutoff = now - timedelta(days=window_years * 365)

    # Find all matches between these two teams (either direction)
    mask = (
        ((results_df["home_team"] == home_team) & (results_df["away_team"] == away_team)) |
        ((results_df["home_team"] == away_team) & (results_df["away_team"] == home_team))
    )
    h2h = results_df[mask & (results_df["date"] >= cutoff)].copy()

    if len(h2h) == 0:
        return 0.5, 0   # neutral fallback

    wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team:
            if row["home_score"] > row["away_score"]:
                wins += 1
        else:
            # home_team was the away team in this match
            if row["away_score"] > row["home_score"]:
                wins += 1

    return wins / len(h2h), len(h2h)

def predict_outcome(model_path, rankings_df, home_team, away_team, neutral=0, results_df=None):
    model = joblib.load(model_path)
    scaler = joblib.load("models/scaler.pkl")

    home_points = get_latest_rank(rankings_df, home_team)
    away_points = get_latest_rank(rankings_df, away_team)
    points_diff = home_points - away_points

    # Compute H2H features
    if results_df is not None:
        h2h_rate, h2h_count = compute_h2h_at_inference(results_df, home_team, away_team)
    else:
        h2h_rate, h2h_count = 0.5, 0  # fallback if no results data

    X_input = pd.DataFrame(
        [[points_diff, neutral, h2h_rate, h2h_count]],
        columns=["points_diff", "neutral_venue", "h2h_home_win_rate", "h2h_match_count"]
    )
    X_scaled = scaler.transform(X_input)
    probs = model.predict_proba(X_scaled)[0]
    return {"away_win": probs[0], "draw": probs[1], "home_win": probs[2]}