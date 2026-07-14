# <div align="center"> ⚽ Match Oracle </div>
 
### <div align="center"> AI-Powered Football Outcome Predictor & LangChain RAG Scout Report Generator </div>

## <div align="center">![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) ![FAISS](https://img.shields.io/badge/FAISS-00599C?style=for-the-badge&logo=meta&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white) </div>
<!-- ![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white) -->
 

Match Oracle predicts international football match outcomes using a Logistic Regression model trained on 20+ years of FIFA ranking and match history data, then pairs that prediction with an AI-generated scouting report, built with a genuine **LangChain RAG pipeline** (FAISS vector search + prompt templates) instead of a bare API call. Pick two teams, get a win/draw/loss probability, and read a context-grounded tactical preview, all in one Streamlit page.
 
## <div align="center"><img style="border: 2px solid #626262ad; padding: 8px 1px; border-radius: 12px;" height="495" width="420" src = "./public/preview.png" /></div>
<!-- ![previewImg](./public/preview.png) -->

---
 
## Table of Contents
 
* [Features](#features)
* [Tech Stack](#tech-stack)
* [System Architecture](#system-architecture)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Environment Variables](#environment-variables)
* [API Reference](#api-reference)
* [Application Flow](#application-flow)
* [Known Issues & Lessons Learned](#known-issues--lessons-learned)
* [Sponsor](#sponsor)
* [License](#license)
---
 
## Features
 
- 🔮 **Match outcome prediction**: Logistic Regression classifier trained on 23,000+ international matches (2000–2024), predicting home win / draw / away win probabilities from FIFA rank difference and venue neutrality.
- 🤖 **AI scout reports**: a real LangChain RAG chain (not a raw prompt string) retrieves each team's current form from a curated knowledge base and generates a grounded, 3-sentence tactical preview.
- 📊 **Feature-scaled model**: inputs are standardized with `StandardScaler` so the model weighs rank difference on equal footing with categorical features, instead of one feature silently drowning out the other.
- 🌍 **Team-name reconciliation**: a mapping layer resolves spelling mismatches between the two source datasets (e.g. `Iran` ↔ `IR Iran`, `USA` ↔ `United States`) so real matches aren't silently dropped during the merge.
- 🖥️ **One-page Streamlit UI**: team dropdowns, live probability metrics, and an AI-generated report, no page reloads or extra tabs.
- 🔌 **Swappable LLM backend**: the RAG chain works identically with Google Gemini or Groq (Llama) behind a single `PROMPT | llm` LangChain expression, so switching providers is a two-line change.
---
 
## Tech Stack
 
| Layer | Technology | Purpose |
|---|---|---|
| Data processing | **Pandas**, **NumPy** | Cleaning, merging, and feature engineering on raw Kaggle CSVs |
| ML model | **scikit-learn** (Logistic Regression + StandardScaler) | Multi-class outcome prediction from rank difference and venue |
| Model persistence | **joblib** | Save/load trained model and scaler without retraining on every run |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) via `langchain-huggingface` | Local, free text embeddings for the RAG knowledge base |
| Vector store | **FAISS** | In-memory similarity search over team news/form data |
| LLM orchestration | **LangChain** (`ChatPromptTemplate`, LCEL chains) | Wires the retriever, prompt, and LLM into a single reusable chain |
| LLM provider | **Google Gemini** / **Groq (Llama)** | Generates the final natural-language scout report |
| Frontend | **Streamlit** | Single-page interactive web app |
| Config | **python-dotenv** | Loads API keys from `.env` without hardcoding secrets |
| Version control | **Git & GitHub** | Phase-by-phase commit history, branching |
 
---
 
## System Architecture
 
```mermaid
flowchart TD
    A["User picks Home & Away team<br/>(Streamlit UI)"] --> B["Latest FIFA rank lookup<br/>(pandas)"]
    B --> C["Feature scaling<br/>(StandardScaler)"]
    C --> D["Outcome prediction<br/>(Logistic Regression)"]
    D --> E["Win / Draw / Loss<br/>probabilities"]
 
    A --> F["Team news retrieval<br/>(FAISS similarity search)"]
    F --> G["Prompt construction<br/>(LangChain ChatPromptTemplate)"]
    G --> H["LLM generation<br/>(Gemini / Groq)"]
    H --> I["Grounded scout report"]
 
    E --> J["Results rendered<br/>(Streamlit UI)"]
    I --> J
```
 
**Two parallel pipelines feed one page:** the left branch is classical ML (rank data → scaled features → classifier → probabilities), the right branch is Gen AI (team news → vector retrieval → prompt → LLM → narrative). Both converge in `app.py`, which is intentionally the only file that knows about both halves.
 
---
 
## Project Structure
 
```
match-oracle/
├── .env                          # API keys (GROQ_API_KEY) 
├── .gitignore
├── .streamlit/
│   └── config.toml               # File watcher config (avoids transformers/torchvision import noise)
├── requirements.txt
├── README.md
├── app.py                        # Streamlit entrypoint: the only file touching the UI
├── data/
│   ├── raw/
│   │   ├── results.csv           # International match history (Kaggle)
│   │   └── fifa_ranking.csv      # FIFA World Ranking history (Kaggle)
│   └── processed/
│       └── training_data.csv     # Cleaned, merged, feature-engineered output
├── context/
│   └── team_news.csv             # Hand-curated "live" team form: the RAG knowledge base
├── vectorstore/
│   └── team_news_index/          # FAISS index, built once from team_news.csv, loaded at runtime
├── models/
│   ├── classifier.pkl            # Trained Logistic Regression model
│   └── scaler.pkl                # Fitted StandardScaler (must accompany the model)
├── notebooks/
│   └── 01_eda_and_prep.ipynb     # EDA scratchpad: class balance, rank_diff distribution checks
└── src/
    ├── __init__.py
    ├── data_processor.py         # CSV cleaning, team-name mapping, merge_asof, feature engineering
    ├── model_trainer.py          # Train/test split, scaling, Logistic Regression, evaluation
    ├── rag_engine.py             # FAISS vector store, prompt template, LangChain LLM chain
    └── predictor.py              # Rank lookup + scaled prediction: the glue app.py calls into
```
 
---
 
## Getting Started
 
### Prerequisites
- Python 3.11+ (or your installed version — see note on compatibility below)
- Git
- A free [Groq API key](https://console.groq.com/keys)
- Kaggle account (to download the two datasets)
### Installation
 
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/match-oracle.git
cd match-oracle
 
# 2. Create and activate a virtual environment
python -m venv env
source env/bin/activate        # Mac/Linux
env\Scripts\activate           # Windows PowerShell
 
# 3. Install dependencies
pip install -r requirements.txt
 
# 4. Download the datasets and place them here:
#    data/raw/results.csv       <- International football results (Kaggle: martj42)
#    data/raw/fifa_ranking.csv  <- FIFA World Ranking (Kaggle: cashncarry)
 
# 5. Set up your .env file (see Environment Variables below)
 
# 6. Run the data pipeline and train the model
python src/data_processor.py
python src/model_trainer.py
 
# 7. Launch the app
streamlit run app.py
```
 
The app opens at `http://localhost:8501`. On first prediction, the RAG pipeline builds and caches a FAISS index from `context/team_news.csv` — this takes a few extra seconds only on the very first run.
 
---
 
## Environment Variables
 
Create a `.env` file in the project root (already excluded from Git via `.gitignore`):
 
```env
GEMINI_API_KEY=your_gemini_key_here
# — or, if using the Groq backend instead —
GROQ_API_KEY=your_groq_key_here
```
 
`src/rag_engine.py` loads these automatically via `python-dotenv`'s `load_dotenv()` at import time — no manual wiring needed elsewhere in the codebase.
 
**For deployment (Streamlit Community Cloud):** `.env` is never read in the cloud. Add the same key(s) under your app's **Settings → Secrets** in TOML format instead.
 
---
 
## API Reference
 
This project doesn't expose a REST API — "API" here refers to the internal Python module functions each part of the app calls into.
 
### `src/data_processor.py`
| Function | Description |
|---|---|
| `load_and_merge(matches_path, rankings_path)` | Loads both CSVs, applies team-name fixes, and merges each match with the correct team's most recent FIFA rank using `merge_asof` |
| `engineer_features(df)` | Computes `rank_diff`, `neutral_venue`, and the 3-class `outcome` target |
| `process_football_data(matches_path, rankings_path, output_path)` | Runs the full pipeline and writes `training_data.csv` |
 
### `src/model_trainer.py`
| Function | Description |
|---|---|
| `train_and_save_model(data_path, model_output_path, scaler_output_path)` | Splits data, fits a `StandardScaler`, trains a `LogisticRegression` classifier, prints a `classification_report`, and saves both artifacts via `joblib` |
 
### `src/predictor.py`
| Function | Description |
|---|---|
| `get_latest_rank(rankings_df, team)` | Returns a team's most recent FIFA rank |
| `predict_outcome(model_path, rankings_df, home_team, away_team, neutral=0)` | Loads the model and scaler, computes `rank_diff`, and returns `{away_win, draw, home_win}` probabilities |
 
### `src/rag_engine.py`
| Function | Description |
|---|---|
| `build_vectorstore(csv_path)` | Embeds `team_news.csv` rows and saves a FAISS index |
| `load_vectorstore()` | Loads the saved index, or builds it if missing |
| `generate_scout_report(home_team, away_team, win_prob)` | Retrieves relevant team context, fills the prompt template, and returns the LLM-generated preview |
 
---
 
## Application Flow
 
```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit (app.py)
    participant P as predictor.py
    participant M as classifier.pkl
    participant R as rag_engine.py
    participant F as FAISS Index
    participant L as LLM (Gemini/Groq)
 
    U->>S: Select home & away team, click Predict
    S->>P: predict_outcome(home, away)
    P->>P: Look up latest FIFA ranks
    P->>M: predict_proba(scaled features)
    M-->>P: [away_win, draw, home_win]
    P-->>S: probability dict
    S->>R: generate_scout_report(home, away, win_prob)
    R->>F: similarity_search(team context)
    F-->>R: relevant form notes
    R->>L: PROMPT | llm chain invoke
    L-->>R: generated 3-sentence preview
    R-->>S: scout report text
    S-->>U: Render probabilities + report
```
 
---
 
<!-- ## Known Issues & Lessons Learned
 
Documenting this because working through it was most of the actual engineering effort:
 
- **Silent data loss from index misalignment (fixed):** the initial `merge_asof` pipeline was retaining only ~4% of eligible matches. The root cause was assigning `merge_asof`'s freshly-indexed output back onto a `DataFrame` that still carried its original, non-sequential index — pandas aligned by index label instead of row position, silently producing `NaN`s almost everywhere. Fixed with a single `.reset_index(drop=True)` before merging, recovering match retention to **91.8%**.
- **Unscaled features skewing the model (fixed):** `rank_diff` (range ≈ ±200) and `neutral_venue` (0 or 1) were fed into Logistic Regression without scaling, so the model's coefficients compensated by nearly ignoring `rank_diff` entirely. Fixed by fitting a `StandardScaler` on the training features and persisting it alongside the model.
- **Draw prediction remains weak:** with only two features, the model rarely predicts draws — a well-known hard case in football analytics, since draws don't correlate as cleanly with rank gap as decisive wins do. Planned improvement: add recent-form or head-to-head features.
- **Live model deprecation mid-build:** `gemini-1.5-flash` and later `gemini-2.5-flash-lite` were both retired by Google during development, requiring a mid-project provider swap — handled cleanly thanks to LangChain's provider-agnostic `PROMPT | llm` interface.
--- -->
 
## Sponsor
 
Match Oracle is an independent student project, not currently backed by any sponsor. If this project helped you learn something or you'd like to support future development, a star ⭐ on the repo goes a long way.
 
---
 
## License
 
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. In short: free to use, modify, and distribute, with attribution appreciated.
 
---
 
### <div align="center">Made with ❤️ using Python, Pandas, NumPy, scikit-learn, LangChain, FAISS, Streamlit & Groq by **Abhimanyu Kumar**, **Adrija Das**, and **Arpan Paul**</div>
