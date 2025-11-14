# GitHub Repo Q&A Agent (FastAPI + PydanticAI + optional LangChain RAG)

Answers repo questions like:
- "What was implemented in the last commit and by whom?"
- "When was the last deployment?"
- "When did we fix the favicon bug?"
- "When did we refactor the Homepage layout?"

Always prints UTC + Europe/Berlin timestamps and citations (commit/workflow/PR URLs).

## Architecture

- FastAPI API (slim): `app.py`
- Agent package (logic + tools): `agent/`
  - `agent/config.py` – env + constants (MODEL_NAME, DEFAULT_TZ, GITHUB_API)
  - `agent/utils.py` – headers, timestamp pairs, JSON/error handling
  - `agent/models.py` – `RepoAnswer` (output), `Deps` (token + timezone)
  - `agent/core.py` – creates the PydanticAI `agent` with instructions
  - `agent/tools_commits.py` – `last_commit`, `find_commit`
  - `agent/tools_deployments.py` – `last_deployment` (Deployments API → Actions fallback)
  - `agent/tools_prs.py` – `find_pr_merge` (merged PR search)
  - `agent/tools_files.py` – `last_file_change`, `introduced_line` (diff scan)
  - `agent/tools_rag.py` – `rag_find_change` (optional, if RAG index exists)
  - `agent/__init__.py` – exports and imports tools (auto-register)
- CLI client: `cli.py`
- Optional RAG builder: `rag_index.py`, loader/validator `rag.py`
- Runbook + demo: `RUN.md`, `demo.ps1`

### Data Flow (ASCII)

CLI/HTTP -> FastAPI `/chat` -> PydanticAI `agent` -> Calls tool(s) -> GitHub API
                                                          |        -> format answer
                                                          v
                                                    Citations + Timestamps

```
cli.py  -->  app.py (/chat)  -->  agent/core.py
                               \->  agent/tools_*  --> GitHub REST
RAG (opt): rag_index.py -> Chroma  -> rag.py -> tools_rag
```

## Quickstart (Windows 11, PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install tzdata
```

Pick one LLM path (choose one):

**OpenAI**
```powershell
$Env:OPENAI_API_KEY = "<openai-key>"
$Env:PYDANTIC_AI_MODEL = "openai:gpt-4o-mini"
```

**DeepSeek (OpenAI-compatible)**
```powershell
$Env:OPENAI_API_KEY = "<deepseek-key>"
$Env:OPENAI_BASE_URL = "https://api.deepseek.com"
$Env:PYDANTIC_AI_MODEL = "openai:<deepseek-model-id>"
```

**Local Ollama**
```powershell
# winget install --id Ollama.Ollama -e
ollama pull llama3.1
$Env:PYDANTIC_AI_MODEL = "ollama:llama3.1"
```

**OpenRouter (OpenAI-compatible)**
```powershell
$Env:OPENAI_API_KEY = "<openrouter-key>"
$Env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
$Env:PYDANTIC_AI_MODEL = "openai:google/gemini-2.5-flash-lite-preview-09-2025"
```
Note: Do not commit real API keys. Prefer `.env` or local environment variables.

GitHub token (read access):
```powershell
$Env:GITHUB_TOKEN = "<github-pat>"
```

Run API:
```powershell
uvicorn app:app --reload --port 8000
```

Ask from terminal:
```powershell
python cli.py --repo owner/repo "What was implemented in the last commit and by whom?"
python cli.py --repo owner/repo --environment prod "When was the last deployment?"
python cli.py --repo owner/repo "When did we fix the favicon bug?"
python cli.py --repo owner/repo "When did we refactor the Homepage layout?"
# File/code-aware examples
python cli.py --repo owner/repo "When did we last change README.md?"
python cli.py --repo owner/repo "When did we add @app.get('/health') in app.py?"
```

### Environment Setup Files
- `.env`: already included (loaded automatically by the app via `python-dotenv`). It contains the keys you provided.
- `env.ps1`: run in PowerShell to set session-local environment variables.

Usage (PowerShell):
```powershell
# One-time per new terminal session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./env.ps1

# Start API (uses .env and current session env vars)
uvicorn app:app --reload --port 8000

# In the same or another terminal (with env.ps1 loaded)
python cli.py --repo owner/repo "What was implemented in the last commit and by whom?"
```

### Health Check
```powershell
curl http://127.0.0.1:8000/health
# { "ok": true }
```

### One-Page Run Guide and Demo
- See `RUN.md` for a concise step-by-step runbook (two-terminal workflow, quoting tips, troubleshooting).
- Run the sample script after starting the API:
  ```powershell
  .\demo.ps1  # queries your repo with 4 example questions
  ```

## Repository Layout

- `app.py` – FastAPI entrypoint (health + chat)
- `agent/` – agent logic and tools
- `cli.py` – terminal client
- `rag.py`, `rag_index.py` – optional RAG (commit index builder and retriever)
- `RUN.md`, `README.md`, `demo.ps1` – docs and demo
- `.env.example` – template for environment variables (do not commit real keys)
- `.gitignore` – ignores `.env`, `env.ps1`, caches, venv, rag_store

## Pushing Changes

```powershell
git status
git add -A
git commit -m "refactor: split agent into package; docs + tools"
git push origin main
```
Notes:
- Never commit `.env` or `env.ps1` (secrets). `.gitignore` already protects them.
- If you want to track `.env.example`, ensure it contains placeholders only.

### Optional: Build the RAG index
```powershell
# OpenAI embeddings
$Env:OPENAI_API_KEY = "<openai-key>"
python rag_index.py --repo owner/repo --max-commits 1500

# OR local embeddings
$Env:EMBED_MODE = "local"
python rag_index.py --repo owner/repo --max-commits 1500
```

### Demo Script (suggested)
```powershell
uvicorn app:app --reload --port 8000
python cli.py --repo owner/repo "What was implemented in the last commit and by whom?"
python cli.py --repo owner/repo --environment prod "When was the last deployment?"
python cli.py --repo owner/repo "When did we fix the favicon bug?"
python cli.py --repo owner/repo "When did we refactor the Homepage layout?"
```
Start API: uvicorn app:app --reload --port 8000
Health: curl http://127.0.0.1:8000/health
Last commit: python cli.py --repo karthikGow/repo_qna_agent "What was implemented in the last commit and by whom?"
Last deployment (prod): python cli.py --repo karthikGow/repo_qna_agent --environment prod "When was the last deployment?"
Favicon fix: python cli.py --repo karthikGow/repo_qna_agent "When did we fix the favicon bug?"
Homepage refactor: python cli.py --repo karthikGow/repo_qna_agent "When did we refactor the Homepage layout?"
### Git Identity (optional)
```powershell
git config --global user.name "Your Name"
git config --global user.email "<your-noreply-or-safe-email>"
- Minor cleanup to support homepage layout refactor demo.
```

