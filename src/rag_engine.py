# Create the RAG pipeline
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
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

# PROMPT = ChatPromptTemplate.from_template(
#     """You are an expert football news reporter. {home_team} is playing {away_team}.
# Our model gives {home_team} a {win_prob}% chance of winning.

# Retrieved context:
# {context}

# Write a punchy, trendy 3-sentence tactical preview based strictly on this context and probability.
# Justify the probability.
# Do not invent facts that aren't in the context."""
# )
PROMPT = ChatPromptTemplate.from_template(
    """You are a football match analyst. {home_team} vs {away_team}. Venue: {venue_status}.

Base probabilities: {home_team} {base_home_prob}% | Draw {base_draw_prob}% | {away_team} {base_away_prob}%

News headlines:
{context}

RULES:
- Only adjust probabilities (up to +/-10%) for CONFIRMED current injuries, suspensions, or lineup changes. Everything else: keep base probabilities unchanged.
- Write exactly 3 sentences. Be specific — name players and cite the headline. Do NOT contradict yourself.
- If one headline says "no injuries" and another says "player X is injured", pick the more specific one (the named player) and go with that.
- Do NOT write generic filler. No "closely contested", no "poised to be", no "looking to capitalize".

EXAMPLE of good output:
{{"adjusted_home_win": 42, "adjusted_draw": 28, "adjusted_away_win": 30, "tactical_preview": "England's Jordan Henderson is ruled out after a celebration injury, leaving a gap in midfield that Jack Rice may need to fill. Argentina report no new fitness concerns heading into the semi-final. With England missing a key midfielder, the odds tilt slightly toward Argentina."}}

Now write yours. Respond ONLY with JSON:
{{"adjusted_home_win": 0, "adjusted_draw": 0, "adjusted_away_win": 0, "tactical_preview": "..."}}
"""
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

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
    
    chain = PROMPT | llm | JsonOutputParser()
    result_dict = chain.invoke({
        "home_team": home_team,
        "away_team": away_team,
        "venue_status": venue_status,
        "base_home_prob": base_home,
        "base_draw_prob": base_draw,
        "base_away_prob": base_away,
        "context": context,
    })
    return result_dict