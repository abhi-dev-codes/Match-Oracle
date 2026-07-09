# Project Structure Guide — AI-Car-Pricing-RAG-Agent

## Redesigned Folder Structure

The original structure was solid but had one real problem: the feature-engineering logic (one-hot encoding, column alignment) would need to exist in *both* `train_model.py` and `app.py`, copy-pasted. If you change it in one place and forget the other, your model breaks silently. The fix is a shared `preprocessing.py` module. I've also split `data/` into `raw/` vs `processed/`, which is standard practice and signals you understand data pipelines, not just scripts.

```
AI-Car-Pricing-RAG-Agent/
├── data/
│   ├── raw/
│   │   ├── car_price_data.csv        # CarDekho dataset, untouched
│   │   └── car_reviews.csv           # Edmunds dataset, untouched
│   └── processed/
│       └── car_reviews_filtered.csv  # cleaned subset used for RAG (generated, not hand-edited)
├── models/
│   ├── price_model.pkl               # trained regression model
│   └── model_columns.pkl             # column order the model expects (needed to align app.py inputs)
├── vectorstore/
│   └── faiss_index/                  # persisted FAISS index (auto-created, don't hand-edit)
├── src/
│   ├── __init__.py                   # makes src/ importable as a package
│   ├── config.py                     # shared file paths & constants — single source of truth
│   ├── preprocessing.py              # shared feature-engineering function — used by BOTH training and the app
│   ├── train_model.py                # Step 2: regression pipeline
│   ├── build_vectorstore.py          # Step 3: ingest reviews → embeddings → FAISS
│   └── rag_chain.py                  # Step 4: LangChain retrieval + generation chain
├── app.py                            # Step 5: Streamlit UI, ties everything together
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## `data/` — the raw material

**What:** Two CSVs you download once from Kaggle and never edit by hand.
- `data/raw/car_price_data.csv` → CarDekho dataset: `https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho`
- `data/raw/car_reviews.csv` → Edmunds dataset: `https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews`

**Why:** Keeping raw data separate from generated/processed data means if you ever mess up a cleaning step, you re-run a script instead of re-downloading or manually fixing a corrupted file. `data/processed/car_reviews_filtered.csv` is *generated* by `build_vectorstore.py` (see below) — you never write to it directly.

**How:** Just download both CSVs from the links above and drop them into `data/raw/`. Nothing to code here.

---

## `models/` — trained artifacts

**What:** Two `joblib` files produced by `src/train_model.py`:
- `price_model.pkl` — the fitted `LinearRegression` object
- `model_columns.pkl` — the exact list/order of columns the model was trained on

**Why `model_columns.pkl` matters:** `pd.get_dummies()` only creates columns for categories *present in the data it sees*. If your Streamlit form submits a car with `Fuel_Type = "CNG"` but your training data only had a handful of CNG rows, the one-hot columns won't line up between training and inference unless you explicitly save and reuse the training-time column list. This is a classic bug in student ML projects — solving it properly is a good interview talking point.

**How:** Created automatically when you run `train_model.py` — nothing to hand-build.

---

## `vectorstore/faiss_index/` — the RAG index

**What:** A FAISS vector index of embedded review chunks, saved to disk by `src/build_vectorstore.py`.

**Why:** Embedding text is slow-ish and you don't want to redo it every time the Streamlit app starts. Build it once, persist it, and `rag_chain.py` just loads it from disk.

**How:** Auto-created by running `build_vectorstore.py` once. Don't commit this folder to Git (it can get large) — add it to `.gitignore` and let each teammate regenerate it locally, or generate it as part of a setup script.

---

## `src/config.py` — single source of truth for paths (new file)

**What / Why:** Every other file needs to know where the data, model, and vectorstore live. Hardcoding `"data/raw/car_price_data.csv"` in three different files means renaming or moving anything breaks silently in whichever file you forgot. One config file fixes that.

**How:**
```python
# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PRICE_DATA = BASE_DIR / "data" / "raw" / "car_price_data.csv"
RAW_REVIEWS_DATA = BASE_DIR / "data" / "raw" / "car_reviews.csv"
PROCESSED_REVIEWS = BASE_DIR / "data" / "processed" / "car_reviews_filtered.csv"

PRICE_MODEL_PATH = BASE_DIR / "models" / "price_model.pkl"
MODEL_COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"

FAISS_INDEX_DIR = BASE_DIR / "vectorstore" / "faiss_index"

# Demo scope — filter the huge Edmunds dataset down to these brands to keep embedding fast
DEMO_BRANDS = ["Honda", "Toyota", "Ford", "BMW", "Hyundai"]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"
```

---

## `src/preprocessing.py` — shared feature engineering (new file)

**What:** One function, `build_features()`, that converts raw car details into the exact feature row the regression model expects. Used by *both* `train_model.py` (on the full training set) and `app.py` (on a single user-submitted row).

**Why:** This is the fix for the duplication problem mentioned above. Define the transformation logic once; both training and inference call the same function, so they can never drift out of sync.

**How:**
```python
# src/preprocessing.py
import pandas as pd

CATEGORICAL_COLS = ["Fuel_Type", "Seller_Type", "Transmission"]

def build_features(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """Turn raw CarDekho-style rows into model-ready features.
    Works on a full training DataFrame or a single-row DataFrame from the UI."""
    df = df.copy()
    df["Car_Age"] = reference_year - df["Year"]
    df = df.drop(columns=["Year"], errors="ignore")
    df = df.drop(columns=["Car_Name"], errors="ignore")
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df
```

---

## `src/train_model.py` — Step 2: the regression pipeline

**What:** Loads the CarDekho CSV, cleans it, trains `LinearRegression`, saves the model and its column list.

**Why:** This is the part you already know from your Multiple Regression coursework — no new concepts here, just wiring it into the shared `preprocessing.py`.

**How:**
```python
# src/train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from config import RAW_PRICE_DATA, PRICE_MODEL_PATH, MODEL_COLUMNS_PATH
from preprocessing import build_features

df = pd.read_csv(RAW_PRICE_DATA)
df = df.dropna()

features = build_features(df)
X = features.drop(columns=["Selling_Price"])
y = features["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("R² on test set:", model.score(X_test, y_test))

joblib.dump(model, PRICE_MODEL_PATH)
joblib.dump(list(X.columns), MODEL_COLUMNS_PATH)
print("Model and column list saved.")
```
Run with: `python src/train_model.py` (from the project root, or `python -m src.train_model` if you added `__init__.py` and run as a package).

---

## `src/build_vectorstore.py` — Step 3: RAG ingestion

**What:** Loads the Edmunds reviews CSV, filters to your demo brands, splits review text into chunks, embeds them, and saves a FAISS index.

**Why:** This is the new RAG piece. Filtering to `DEMO_BRANDS` first keeps embedding fast (the full Edmunds set is large — you don't need all 60+ brands for a demo).

**How:**
```python
# src/build_vectorstore.py
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import RAW_REVIEWS_DATA, PROCESSED_REVIEWS, FAISS_INDEX_DIR, DEMO_BRANDS, EMBEDDING_MODEL

# Load and filter — Edmunds columns: Company, Model, Year, Reviewer, Date, Title, Rating, Review
df = pd.read_csv(RAW_REVIEWS_DATA)
df = df[df["Company"].isin(DEMO_BRANDS)]
df = df[["Company", "Model", "Review"]].dropna()
df["page_content"] = df["Company"] + " " + df["Model"] + ": " + df["Review"]

# Save the filtered/processed version — this is what data/processed/ is for
df.to_csv(PROCESSED_REVIEWS, index=False)

loader = DataFrameLoader(df, page_content_column="page_content")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(str(FAISS_INDEX_DIR))

print(f"Indexed {len(chunks)} review chunks from {df['Company'].nunique()} brands.")
```
Run once (offline, not on every app start): `python src/build_vectorstore.py`

---

## `src/rag_chain.py` — Step 4: retrieval + generation chain

**What:** Loads the saved FAISS index, builds a LangChain retriever, and exposes one function `generate_pitch()` that retrieves relevant reviews and generates a grounded marketing pitch.

**Why:** This is where retrieval (RAG) and generation (LLM) actually connect. Keeping it as one importable function means `app.py` doesn't need to know anything about LangChain internals — clean separation of concerns.

**How:**
```python
# src/rag_chain.py
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from config import FAISS_INDEX_DIR, EMBEDDING_MODEL, LLM_MODEL

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(
    str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    """You are an expert car salesperson writing a short, punchy Facebook
Marketplace listing.

Car: {year} {brand} {model}
Mileage: {mileage} km
ML-predicted fair price: ${predicted_price}

Real owner feedback retrieved for this brand/model (use these as concrete
selling points, don't invent features not supported here):
{context}

Write a 3-4 sentence marketing pitch that highlights the price as a great
deal and weaves in 1-2 real selling points from the feedback above."""
)

def format_docs(docs):
    return "\n".join(f"- {d.page_content}" for d in docs)

def generate_pitch(brand, model_name, year, mileage, predicted_price):
    query = f"{brand} {model_name} review"
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    chain = prompt | llm | StrOutputParser()
    pitch = chain.invoke({
        "brand": brand, "model": model_name, "year": year,
        "mileage": mileage, "predicted_price": predicted_price,
        "context": context,
    })
    return pitch, retrieved_docs
```

---

## `app.py` — Step 5: Streamlit frontend

**What:** The UI. Collects car details, calls `preprocessing.build_features()` + the saved model to predict price, then calls `rag_chain.generate_pitch()` to write the listing.

**Why:** Note it imports `build_features` from the *same* `preprocessing.py` that training used — this is the whole point of the redesign. No duplicated encoding logic.

**How:**
```python
# app.py
import streamlit as st
import joblib
import pandas as pd

from src.config import PRICE_MODEL_PATH, MODEL_COLUMNS_PATH
from src.preprocessing import build_features
from src.rag_chain import generate_pitch

st.set_page_config(page_title="AI Car Pricing & Marketing Agent")
st.title("🚗 Hybrid Car Pricing & Marketing Agent")
st.caption("Regression pricing model + RAG-grounded marketing copy")

model = joblib.load(PRICE_MODEL_PATH)
model_columns = joblib.load(MODEL_COLUMNS_PATH)

brand = st.text_input("Brand", "Honda")
car_model = st.text_input("Model", "Civic")
year = st.number_input("Year", min_value=1995, max_value=2026, value=2019)
mileage = st.number_input("Kms Driven", min_value=0, value=45000)
present_price = st.number_input("Current Ex-Showroom Price (lakhs)", value=10.0)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

if st.button("Generate Pitch"):
    raw_row = pd.DataFrame([{
        "Year": year, "Present_Price": present_price, "Kms_Driven": mileage,
        "Owner": 0, "Fuel_Type": fuel_type, "Seller_Type": seller_type,
        "Transmission": transmission,
    }])
    features = build_features(raw_row).reindex(columns=model_columns, fill_value=0)
    predicted_price = round(model.predict(features)[0], 2)

    st.metric("Predicted Fair Price", f"${predicted_price} lakhs")

    with st.spinner("Retrieving reviews and writing pitch..."):
        pitch, sources = generate_pitch(brand, car_model, year, mileage, predicted_price)

    st.subheader("Generated Marketing Pitch")
    st.write(pitch)

    with st.expander("Grounding evidence (retrieved reviews)"):
        for doc in sources:
            st.markdown(f"- {doc.page_content}")
```
Run with: `streamlit run app.py`

---

## Root-level files

**`requirements.txt`** — generated with `pip freeze > requirements.txt` after installing everything. Anyone cloning the repo runs `pip install -r requirements.txt` to get an identical environment.

**`.env.example`** — a template showing what secrets are needed, without the actual values:
```
GOOGLE_API_KEY=your_key_here
```
Each dev copies this to a real `.env` (which is git-ignored) and fills in their own key.

**`.gitignore`** — prevents secrets and large generated files from being committed:
```
.env
vectorstore/
models/*.pkl
data/processed/
__pycache__/
env/
```

**`README.md`** — the file recruiters/reviewers actually read first. Should include: problem statement, architecture diagram (the input→process→output flow from earlier), setup instructions, and a screenshot or GIF of the running app.

---

## Suggested build order (maps to the 3-day plan)
1. `config.py` → `preprocessing.py` → `train_model.py` (Day 1)
2. `build_vectorstore.py` → `rag_chain.py`, tested standalone before touching Streamlit (Day 2)
3. `app.py`, wiring it all together, then README + polish (Day 3)
