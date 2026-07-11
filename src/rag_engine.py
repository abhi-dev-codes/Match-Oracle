# Create the RAG pipeline
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

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
    """You are an expert football scout. {home_team} is playing {away_team}.
Our model gives {home_team} a {win_prob}% chance of winning.

Retrieved context:
{context}

Write a punchy, 3-sentence tactical preview based strictly on this context and probability.
Do not invent facts that aren't in the context."""
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

def generate_scout_report(home_team, away_team, win_prob):
    store = load_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": 4})

    hits = retriever.invoke(f"{home_team} {away_team} recent form")
    context = "\n".join(f"- {d.metadata['team']}: {d.page_content}" for d in hits)

    chain = PROMPT | llm
    response = chain.invoke({
        "home_team": home_team,
        "away_team": away_team,
        "win_prob": win_prob,
        "context": context,
    })
    return response.content