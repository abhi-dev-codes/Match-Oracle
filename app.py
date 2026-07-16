import streamlit as st
import pandas as pd
from src.predictor import predict_outcome
from src.rag_engine import generate_scout_report
from src.utils import get_flag_url

st.set_page_config(page_title="Match Oracle", page_icon="⚽", layout="centered")

# Custom CSS for the Google-like UI
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.match-header {
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 20px 0;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 20px;
}
.team-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 33%;
}
.team-flag {
    width: 60px;
    height: 40px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #ddd;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}
.team-name {
    font-size: 1.2rem;
    margin-top: 10px;
    font-family: sans-serif;
    text-align: center;
}
.vs-text {
    font-size: 1.1rem;
    color: #5f6368;
    margin-bottom: 25px;
    width: 33%;
    text-align: center;
}
.prob-card {
    border: 1px solid #dadce0;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.prob-title {
    font-size: 0.85rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}
.prob-stats {
    display: flex;
    justify-content: space-between;
    font-weight: bold;
    margin-bottom: 10px;
}
.prob-stats span {
    display: flex;
    flex-direction: column;
}
.prob-stats .left { text-align: left; }
.prob-stats .center { text-align: center; }
.prob-stats .right { text-align: right; }
.prob-val {
    font-weight: normal;
    color: #1a73e8;
    margin-top: 4px;
}
.prob-val.right-val { color: #d93025; }
.prob-val.center-val { color: #5f6368; }

.progress-container {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 8px;
}
.progress-home { background-color: #4285f4; border-right: 2px solid white; }
.progress-draw { background-color: #e0e0e0; border-right: 2px solid white; }
.progress-away { background-color: #ea4335; }

.scout-card {
    background-color: #202124;
    color: white;
    border-radius: 12px;
    padding: 24px;
    margin-top: 30px;
    min-height: 150px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.scout-title {
    font-weight: 900;
    font-size: 1.8rem;
    text-transform: uppercase;
    font-family: 'Helvetica Neue', sans-serif;
    margin-bottom: 15px;
    color: #f1f3f4;
}
</style>
""", unsafe_allow_html=True)

import base64

def set_bg_gif(gif_path):
    with open(gif_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/gif;base64,{encoded_string}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_gif("public/aidan-yelamos-berbel-footballnight.gif")

st.title("⚽ Match Oracle")

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/fifa_ranking.csv", parse_dates=["rank_date"])

@st.cache_data
def load_results():
    from src.data_processor import NAME_FIXES
    df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
    df["home_team"] = df["home_team"].replace(NAME_FIXES)
    df["away_team"] = df["away_team"].replace(NAME_FIXES)
    return df

rankings_df = load_data()
results_df = load_results()
teams = sorted(rankings_df["country_full"].unique())

# Initialize session state for swapping
if "home_team" not in st.session_state:
    st.session_state["home_team"] = "Argentina"
if "away_team" not in st.session_state:
    st.session_state["away_team"] = "England"

def swap_teams():
    st.session_state["home_team"], st.session_state["away_team"] = (
        st.session_state["away_team"],
        st.session_state["home_team"],
    )

# 1. MATCH HEADER HTML (ALWAYS VISIBLE)
home_state = st.session_state["home_team"]
away_state = st.session_state["away_team"]
h_flag = get_flag_url(home_state)
a_flag = get_flag_url(away_state)

st.markdown(f"""
<div class="match-header">
    <div class="team-block">
        <img src="{h_flag}" class="team-flag">
        <div class="team-name">{home_state}</div>
    </div>
    <div class="vs-text">vs</div>
    <div class="team-block">
        <img src="{a_flag}" class="team-flag">
        <div class="team-name">{away_state}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Selectors
col1, col2, col3 = st.columns([5, 2, 5])
with col1:
    st.selectbox("Home team", teams, key="home_team", label_visibility="collapsed")
with col2:
    st.button("⇄", on_click=swap_teams, use_container_width=True)
with col3:
    st.selectbox("Away team", teams, key="away_team", label_visibility="collapsed")

is_neutral = st.checkbox("Neutral Venue (e.g., World Cup match)", value=True)

if st.button("Predict outcome", type="primary"):
    home = st.session_state["home_team"]
    away = st.session_state["away_team"]

    if home == away:
        st.error("Pick two different teams.")
    else:
        probs = predict_outcome("models/classifier.pkl", rankings_df, home, away, neutral=int(is_neutral), results_df=results_df)
        
        h_prob = int(round(probs['home_win']*100))
        d_prob = int(round(probs['draw']*100))
        a_prob = int(round(probs['away_win']*100))

        # Adjust to ensure exactly 100% total due to rounding
        total = h_prob + d_prob + a_prob
        if total != 100:
            a_prob += (100 - total)

        # 2. WIN PROBABILITY BAR HTML
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-title">WIN PROBABILITY (90 MIN)</div>
            <div class="prob-stats">
                <span class="left">{home}<br><span class="prob-val">{h_prob}%</span></span>
                <span class="center">Draw<br><span class="prob-val center-val">{d_prob}%</span></span>
                <span class="right">{away}<br><span class="prob-val right-val">{a_prob}%</span></span>
            </div>
            <div class="progress-container">
                <div class="progress-home" style="width: {h_prob}%;"></div>
                <div class="progress-draw" style="width: {d_prob}%;"></div>
                <div class="progress-away" style="width: {a_prob}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. AI SCOUT REPORT (replaces video thumbnails)
        with st.spinner("Analyzing recent form for scout report..."):
            report = generate_scout_report(home, away, h_prob)

        st.markdown(f"""
        <div class="scout-card">
            <div class="scout-title">MATCH STORY</div>
            <div style="font-size: 1.05rem; line-height: 1.6; color: #e8eaed;">{report}</div>
        </div>
        """, unsafe_allow_html=True)

# 4. FLOATING BUTTONS
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    github_svg = get_base64_image("public/github.svg")
    gif_svg = get_base64_image("public/filetype-gif.svg")

    st.markdown(f"""
    <style>
    .floating-buttons {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        display: flex;
        gap: 15px;
        z-index: 9999;
    }}
    .floating-btn {{
        width: 45px;
        height: 45px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s, background-color 0.2s;
    }}
    .floating-btn:hover {{
        transform: scale(1.1);
        background-color: white;
    }}
    .floating-btn img {{
        width: 24px;
        height: 24px;
    }}
    </style>
    <div class="floating-buttons">
        <a href="https://github.com/abhi-dev-codes/Match-Oracle" target="_blank" class="floating-btn" title="GitHub Repository">
            <img src="data:image/svg+xml;base64,{github_svg}" alt="GitHub">
        </a>
        <a href="https://cdna.artstation.com/p/assets/images/images/075/142/478/original/aidan-yelamos-berbel-footballnight.gif?1713861751" target="_blank" class="floating-btn" title="Background GIF Source">
            <img src="data:image/svg+xml;base64,{gif_svg}" alt="GIF Source">
        </a>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    pass
