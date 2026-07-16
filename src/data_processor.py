# src/data_processor.py
import pandas as pd
from datetime import timedelta

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


def compute_h2h_features(df, window_years=4):
    """Compute rolling H2H win rate and match count for each row.
    """
    df = df.sort_values("date").reset_index(drop=True)
    window = timedelta(days=window_years * 365) 

    # Create a canonical pair key so A-vs-B and B-vs-A are the same matchup
    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )

    h2h_win_rates = []  # home_win/total_matches
    h2h_counts = []     # total_matches

    # Group by pair for efficiency — only iterate within each matchup
    for pair, group in df.groupby("pair"):
        group = group.sort_values("date")
        idxs = group.index.tolist()
        dates = group["date"].values
        home_teams = group["home_team"].values
        home_scores = group["home_score"].values
        away_scores = group["away_score"].values

        for i, idx in enumerate(idxs):
            current_date = dates[i]
            cutoff = current_date - window
            current_home = home_teams[i]

            # Look at all prior matches in this pair within the window
            wins = 0
            total = 0
            for j in range(i - 1, -1, -1):
                if dates[j] < cutoff:
                    break
                total += 1
                # Did the current home team win this prior match?
                if home_teams[j] == current_home:
                    # Same orientation: home team was home
                    if home_scores[j] > away_scores[j]:
                        wins += 1
                else:
                    # Flipped orientation: home team was away
                    if away_scores[j] > home_scores[j]:
                        wins += 1

            if total > 0:
                h2h_win_rates.append((idx, wins / total))
                h2h_counts.append((idx, total))
            else:
                h2h_win_rates.append((idx, 0.5))   # neutral fallback
                h2h_counts.append((idx, 0))

    # Map results back to the dataframe
    rate_map = dict(h2h_win_rates)
    count_map = dict(h2h_counts)
    df["h2h_home_win_rate"] = df.index.map(rate_map)
    df["h2h_match_count"] = df.index.map(count_map)
    df.drop(columns=["pair"], inplace=True)
    return df


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

    # Compute H2H features (4-year rolling window)
    df = compute_h2h_features(df, window_years=4)

    return df[["home_team", "away_team", "rank_diff", "points_diff",
               "neutral_venue", "h2h_home_win_rate", "h2h_match_count", "outcome"]]


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