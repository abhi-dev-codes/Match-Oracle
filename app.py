import streamlit as st
import pandas as pd
from src.predictor import predict_outcome
from src.rag_engine import generate_scout_report

st.set_page_config(page_title="Match Oracle", page_icon="⚽")
st.title("⚽ Match Oracle — AI Football Outcome Predictor")

rankings_df = pd.read_csv("data/raw/fifa_ranking.csv", parse_dates=["rank_date"])
teams = sorted(rankings_df["country_full"].unique())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", teams, index=teams.index("Spain") if "Spain" in teams else 0)
with col2:
    away_team = st.selectbox("Away team", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)

if st.button("Predict outcome", type="primary"):
    if home_team == away_team:
        st.error("Pick two different teams.")
    else:
        probs = predict_outcome("models/classifier.pkl", rankings_df, home_team, away_team)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} win", f"{probs['home_win']*100:.0f}%")
        c2.metric("Draw", f"{probs['draw']*100:.0f}%")
        c3.metric(f"{away_team} win", f"{probs['away_win']*100:.0f}%")

        st.subheader("🤖 AI scout report")
        with st.spinner("Analyzing recent form..."):
            report = generate_scout_report(home_team, away_team, round(probs["home_win"]*100, 1))
        st.write(report)