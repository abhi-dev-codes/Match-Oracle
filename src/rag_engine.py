# Create the RAG pipeline
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTORSTORE_PATH = "vectorstore/team_news_index"

def build_vectorstore(csv_path="context/team_news.csv"):
    df = pd.read_csv(csv_path)
    docs = [
        Document(page_content=row["recent_form"], metadata={"team": row["team"]})
        for _, row in df.iterrows()
    ]
    store = FAISS.from_documents(docs, EMBEDDINGS)
    store.save_local(VECTORSTORE_PATH)
    return store

def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        return build_vectorstore()
    return FAISS.load_local(VECTORSTORE_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)

PROMPT = ChatPromptTemplate.from_template(
    """You are a football match analyst. Be direct, factual, and concise.

MATCH: {home_team} vs {away_team}
VENUE: {venue_status}

BASE PROBABILITIES (from our statistical model):
- {home_team} Win: {base_home_prob}%
- Draw: {base_draw_prob}%
- {away_team} Win: {base_away_prob}%

NEWS HEADLINES:
{context}

STEP 1 — TRIAGE THE HEADLINES:
Classify each headline into one of these categories:
- HIGH PRIORITY (use for probability adjustment): Confirmed injuries to current squad players, suspensions, official lineup changes, manager tactical statements about this specific match.
- LOW PRIORITY (mention in preview but do NOT adjust probabilities): Off-field news (legal, financial, political), headlines about a different team's players, historical stories about retired/deceased players, fan stories, post-match celebrations.
- NOISE (ignore entirely): Clickbait with no factual content, headlines completely unrelated to either team.

STEP 2 — HANDLE CONTRADICTIONS:
If headlines contradict each other (e.g., one says "no new injuries" while another names a specific injured player), always trust the more specific headline. Do NOT mention both sides — just go with the specific fact.

STEP 3 — ADJUST PROBABILITIES:
- Only adjust based on HIGH PRIORITY headlines.
- Maximum adjustment: +/- 10% per team from the base.
- If no high-priority headlines exist, return base probabilities unchanged.
- Adjusted probabilities MUST sum to exactly 100.

STEP 4 — WRITE THE PREVIEW:
- Write exactly 3 sentences.
- Reference specific player names and facts from the headlines.
- You may briefly mention low-priority context if it adds color, but keep focus on match-relevant facts.
- NEVER invent facts not found in the headlines.
- NEVER use generic filler phrases like: "closely contested", "looking to capitalize", "battle for supremacy".

Respond ONLY with valid JSON:
{{
    "adjusted_home_win": <number>,
    "adjusted_draw": <number>,
    "adjusted_away_win": <number>,
    "tactical_preview": "<your 3 sentences>"
}}
"""
)

# --- Gemini (primary) ---
llm_gemini = ChatGoogleGenerativeAI(model="gemini-3.5-flash") # gemini 3.5 flash recomends no temp change

# --- Groq (fallback) ---
from langchain_groq import ChatGroq
llm_groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

PROMPT_GROQ = ChatPromptTemplate.from_template(
    """You are a football match analyst. {home_team} vs {away_team}. Venue: {venue_status}.

Base probabilities: {home_team} {base_home_prob}% | Draw {base_draw_prob}% | {away_team} {base_away_prob}%

News headlines:
{context}

RULES:
- Only adjust probabilities (up to +/-10%) for CONFIRMED current injuries, suspensions, or lineup changes. Everything else: keep base probabilities unchanged.
- Write exactly 3 sentences. Be specific — name players. Do NOT contradict yourself.
- If one headline says "no injuries" and another says "player X is injured", pick the more specific one and go with that.
- Do NOT write generic filler. No "closely contested", no "looking to capitalize".
- Adjusted probabilities MUST sum to exactly 100.

Respond ONLY with JSON:
{{"adjusted_home_win": 0, "adjusted_draw": 0, "adjusted_away_win": 0, "tactical_preview": "..."}}
"""
)

def generate_scout_report(home_team, away_team, base_home, base_draw, base_away, is_neutral=True):
    # Dynamically update news for the selected teams
    import subprocess
    import shutil
    try:
        subprocess.run(["python", "update_news.py", home_team, away_team], check=True)
    except Exception as e:
        print(f"Failed to update news: {e}")

    # Force rebuild vectorstore since CSV has changed
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)

    store = load_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": 2})

    hits = retriever.invoke(f"{home_team} {away_team} recent form")
    context = "\n".join(f"- {d.metadata['team']}: {d.page_content}" for d in hits)

    venue_status = "Neutral Ground" if is_neutral else f"Home advantage for {home_team}"

    invoke_args = {
        "home_team": home_team,
        "away_team": away_team,
        "venue_status": venue_status,
        "base_home_prob": base_home,
        "base_draw_prob": base_draw,
        "base_away_prob": base_away,
        "context": context,
    }

    # Try Gemini first, fall back to Groq on any error (rate limit, API issues, etc.)
    try:
        chain = PROMPT | llm_gemini | JsonOutputParser()
        result_dict = chain.invoke(invoke_args)
    except Exception as e:
        print(f"[FALLBACK] Gemini failed ({type(e).__name__}: {e}), switching to Groq...")
        chain = PROMPT_GROQ | llm_groq | JsonOutputParser()
        result_dict = chain.invoke(invoke_args)

    return result_dict