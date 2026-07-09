Project distilled into I/O/P terms:

**Input**
- User-entered car details via the Streamlit form: brand, model, year, mileage, present/showroom price, fuel type, transmission, seller type
- (Behind the scenes, two static datasets feed the system ahead of time: the CarDekho CSV trains the regression model, and the Edmunds reviews CSV is embedded into the FAISS vector store)

**Process**
1. **Feature formatting** — Pandas turns the raw form inputs into the one-hot-encoded, column-aligned format the regression model expects
2. **Price prediction** — the trained `LinearRegression` model (loaded via `joblib`) predicts a fair price from those features
3. **Retrieval (RAG)** — a query built from brand + model is embedded and searched against the FAISS index; the top-k real owner-review chunks for that brand/model come back
4. **Prompt assembly** — LangChain's `ChatPromptTemplate` combines the predicted price, car details, and retrieved review snippets into one prompt
5. **Generation** — the LLM (Gemini/OpenAI) runs the LCEL chain (`prompt | llm | parser`) and writes a short marketing pitch grounded in that retrieved evidence

**Output**
- The predicted fair price, shown as a Streamlit metric
- A 3–4 sentence marketing pitch for that car, displayed on the page
- (Optionally shown for transparency) the actual retrieved review snippets that grounded the pitch, in an expandable section
