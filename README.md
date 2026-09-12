# AutoML Agent 🤖

An agentic AutoML system that takes any CSV dataset and a natural language goal, runs a full ML pipeline autonomously, and returns a human-readable report with SHAP feature importance visualization — exposed as a remote MCP server and accessible via a Streamlit UI.

---

## What It Does

You upload a CSV and type a goal like "predict survival of passengers" or "forecast sales revenue". The agent does everything else:

→ Understands the problem type (classification, regression, clustering)  
→ Cleans and preprocesses the data intelligently  
→ Engineers new features using LLM reasoning  
→ Trains multiple ML models and selects the best one  
→ Generates SHAP explainability plots  
→ Returns a full structured report in plain English  

If any step fails, the self-correction loop catches the error, asks the LLM for a fix, and retries automatically — up to 3 times.

---

## Live Demo

**Streamlit UI:** Run `streamlit run app.py` locally — connects to the remote MCP server automatically  
**MCP Server:** `https://web-production-3340c.up.railway.app/mcp` — permanently deployed on Railway

---

## Architecture

```
User (Streamlit UI)
    → uploads CSV + enters goal
    → calls Remote MCP Server (Railway)
        → LangGraph Agent runs 7 nodes:
            1. understand_problem  → LLM identifies problem type, target, features
            2. clean_data          → LLM decides cleaning steps, pandas applies them
            3. feature_engineering → LLM suggests new features, eval() creates them
            4. train_models        → sklearn trains 3 models, evaluates each
            5. select_best_model   → picks best by accuracy (classification) or RMSE (regression)
            6. explain_with_shap   → generates SHAP values and summary plot
            7. generate_report     → LLM writes plain-English analysis report
        → self_correct node catches errors and retries (max 3 times)
    → returns report + SHAP plot to Streamlit UI
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangGraph | Agentic workflow orchestration and self-correction loop |
| LangChain + Groq (GPT-OSS 20B) | LLM for problem understanding, cleaning decisions, report generation |
| Scikit-learn | ML model training (LogisticRegression, RandomForest, GradientBoosting) |
| SHAP | Model explainability and feature importance |
| FastMCP | MCP server framework for exposing the agent as a callable tool |
| Railway | Remote deployment of the MCP server (permanent URL) |
| Streamlit | Frontend UI for CSV upload and results display |
| Pandas / NumPy | Data processing and feature engineering |
| Python-dotenv | Secure credential management |

---

## Why LangGraph?

Each step in the ML pipeline is a clearly defined node with explicit edges between them. The conditional self-correction loop — where any node can fail, get diagnosed by the LLM, and retry — would be deeply messy to implement in a plain script. LangGraph makes this clean, traceable, and extensible.

## Why MCP?

Exposing the agent as an MCP server means any MCP client (Claude Desktop, Cursor, or a custom Streamlit client like this one) can call it as a tool. The architecture separates the intelligence (server) from the interface (client) — the same agent can serve multiple frontends without changing a line of backend code.

## Why SHAP?

Accuracy alone doesn't explain WHY a model makes decisions. SHAP assigns credit to each feature for each prediction, making the model's reasoning transparent to non-technical stakeholders — a critical requirement in real-world ML deployments.

---

## Project Structure

```
Auto_ML/
├── agent.py          # LangGraph nodes, AgentState, graph wiring
├── server.py         # FastMCP server exposing analyse_dataset tool
├── app.py            # Streamlit client connecting to remote MCP server
├── requirements.txt  # Python dependencies
├── Procfile          # Railway deployment config
├── nixpacks.toml     # System dependencies for Railway (cairo, pango)
├── .env              # API credentials (never commit this)
├── .gitignore
└── README.md
```

---

## Key Design Decisions

**LLM-guided preprocessing:** Instead of hardcoding cleaning rules, the LLM analyzes df_info and returns JSON instructions (columns to drop, encoding strategy, null handling). This makes the pipeline work on ANY dataset, not just Titanic.

**available_features filter:** After feature engineering, column names in state may not match the cleaned CSV. A filter resolves this dynamically so training never crashes on missing columns.

**Base64 SHAP plot transfer:** The SHAP plot is generated on the server, encoded as base64, embedded in the return string, and decoded by the Streamlit client. No file system sharing needed between server and client.

**Temp file for CSV:** The Streamlit client reads the uploaded CSV as a string and sends it to the MCP server. The server writes it to a temporary file, processes it, then cleans up. This avoids filesystem path issues between client and server environments.

---

## What I'd Add Next

- LangSmith tracing and evaluation dashboard
- Support for clustering problems (currently classification and regression only)
- Model versioning — save trained models and allow reuse on new data
- Deploy Streamlit UI to Railway so the entire project runs in the cloud
- OAuth so multiple users can connect their own datasets securely

---

## Author

**Prarabdha Namdeo**
- GitHub: [@PrarabdhaNamdeo](https://github.com/PrarabdhaNamdeo)
- Threads: [@_prarabdha_namdeo_](https://www.threads.net/@_prarabdha_namdeo_)
