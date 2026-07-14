# src/data_processor.py
import pandas as pd

NAME_FIXES = {
    "Brunei": "Brunei Darussalam",
    "Cape Verde": "Cabo Verde",
    "China": "China PR",
    "Taiwan": "Chinese Taipei",
    "DR Congo": "Congo DR",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "North Korea": "Korea DPR",
    "South Korea": "Korea Republic",
    "Kyrgyzstan": "Kyrgyz Republic",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "St. Kitts and Nevis": "St Kitts and Nevis",
    "St. Lucia": "St Lucia",
    "St. Vincent and the Grenadines": "St Vincent and the Grenadines",
    "Gambia": "The Gambia",
    "United States": "USA",
}

def load_and_merge(matches_path, rankings_path):
    matches = pd.read_csv(matches_path, parse_dates=["date"])
    rankings = pd.read_csv(rankings_path, parse_dates=["rank_date"])

    matches["home_team"] = matches["home_team"].replace(NAME_FIXES)
    matches["away_team"] = matches["away_team"].replace(NAME_FIXES)

    matches = matches[matches["date"].dt.year >= 2000].copy()
    rankings = rankings.sort_values("rank_date")
    matches = matches.sort_values("date").reset_index(drop=True)

    home_stats = pd.merge_asof(
    matches,
    rankings.rename(columns={
        "country_full": "home_team",
        "rank": "home_rank",
        "total_points": "home_points",   # NEW — carry points through the same merge
    }),
    left_on="date", right_on="rank_date", by="home_team", direction="backward"
    )[["home_rank", "home_points"]]

    away_stats = pd.merge_asof(
        matches,
        rankings.rename(columns={
            "country_full": "away_team",
            "rank": "away_rank",
            "total_points": "away_points",
        }),
        left_on="date", right_on="rank_date", by="away_team", direction="backward"
    )[["away_rank", "away_points"]]

    matches["home_rank"] = home_stats["home_rank"]
    matches["home_points"] = home_stats["home_points"]
    matches["away_rank"] = away_stats["away_rank"]
    matches["away_points"] = away_stats["away_points"]
    return matches.dropna(subset=["home_rank", "away_rank", "home_points", "away_points"])


def engineer_features(df):
    df["rank_diff"] = df["away_rank"] - df["home_rank"]
    df["points_diff"] = df["home_points"] - df["away_points"]
    df["neutral_venue"] = df["neutral"].astype(int)

    def outcome(row):
        if row["home_score"] > row["away_score"]:
            return 2   # home win
        if row["home_score"] == row["away_score"]:
            return 1   # draw
        return 0       # away win

    df["outcome"] = df.apply(outcome, axis=1)
    return df[["home_team", "away_team", "rank_diff", "points_diff", "neutral_venue", "outcome"]]


def process_football_data(matches_path, rankings_path, output_path):
    merged = load_and_merge(matches_path, rankings_path)
    features = engineer_features(merged)
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} rows to {output_path}")


if __name__ == "__main__":
    process_football_data(
        "data/raw/results.csv",
        "data/raw/fifa_ranking.csv",
        "data/processed/training_data.csv",
    )