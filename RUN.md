# Run Guide (Windows PowerShell)

This guide shows exactly how to run the Repo Q&A Agent locally and use the CLI to ask questions about a GitHub repo.

## Architecture (quick)

- API: `app.py` – minimal FastAPI routes (`/health`, `/chat`)
- Agent: `agent/` – config, utils, models, core agent, and tools
- CLI: `cli.py` – posts questions to `/chat`
- Optional RAG: `rag_index.py` (build), `rag.py` (retrieve)

ASCII overview:

```
cli.py  -->  app.py (/chat)  -->  agent/core.py
                               \->  agent/tools_*  --> GitHub REST
RAG (opt): rag_index.py -> Chroma  -> rag.py -> tools_rag
```

## Prerequisites
- Windows 11 with PowerShell
- Python 3.10+
- GitHub Personal Access Token (PAT) recommended even for public repos (better rate limits)

## 1) Create and activate a virtual environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies
```powershell
pip install -r requirements.txt
pip install tzdata
```

## 3) Configure environment
Choose ONE of the following:

- Option A: Session-local exports
  ```powershell
  ./env.ps1
  ```
  This sets variables like `OPENAI_API_KEY`, `PYDANTIC_AI_MODEL`, `GITHUB_TOKEN`, and `AGENT_TZ` for the current session.

- Option B: .env file (auto-loaded)
  ```powershell
  Copy-Item .env.example .env
  notepad .env   # fill your keys
  ```

## 4) Start the API (Terminal 1)
```powershell
uvicorn app:app --reload --port 8000
```
Keep this terminal open.

## 5) Health check (Terminal 2)
```powershell
.\.venv\Scripts\Activate.ps1
irm http://127.0.0.1:8000/health
# Expect: { "ok": true }
```

## 6) Ask questions with the CLI (Terminal 2)
Examples for repo `karthikGow/repo_qna_agent`:
```powershell
python cli.py --repo karthikGow/repo_qna_agent "What was implemented in the last commit and by whom?"
python cli.py --repo karthikGow/repo_qna_agent --environment prod "When was the last deployment?"
python cli.py --repo karthikGow/repo_qna_agent "When did we fix the favicon bug?"
python cli.py --repo karthikGow/repo_qna_agent "When did we refactor the homepage?"
# File/code-aware examples
python cli.py --repo karthikGow/repo_qna_agent "When did we last change README.md?"
python cli.py --repo karthikGow/repo_qna_agent "When did we add @app.get('/health') in app.py?"
```
Tips:
- Use single quotes if PowerShell confuses double quotes: `'When did we refactor the homepage?'`
- Interactive mode avoids quoting entirely:
  ```powershell
  python cli.py --repo karthikGow/repo_qna_agent
  > When did we refactor the homepage?
  ```

## 7) Optional: Build RAG index (better "fix/refactor" answers)
```powershell
# OpenAI embeddings
python rag_index.py --repo karthikGow/repo_qna_agent --max-commits 1500

# OR local embeddings
$Env:EMBED_MODE = "local"
python rag_index.py --repo karthikGow/repo_qna_agent --max-commits 1500
```
Restart the API after building the index.

## Pushing to GitHub

```powershell
git status
git add -A
git commit -m "refactor: split agent into package; docs + tools"
git push origin main
```

Secrets: `.env` and `env.ps1` are ignored by `.gitignore`. Never commit real keys.

## Troubleshooting
- Rate limit / 403/429: ensure `GITHUB_TOKEN` is set; public repos still benefit from a token.
- Private repos: token must have read access to the repo.
- Health returns but CLI fails: confirm API is running in Terminal 1 and the URL is `http://127.0.0.1:8000/chat` (default in CLI).
- Quoting issues (PowerShell): if you see a `>>` prompt, retype using `'single quotes'` or add `-- "question here"` at the end.

## Security Notes
- Do not commit real secrets to version control. `.gitignore` already excludes `.env` and `env.ps1`.
- Rotate tokens after your demo as best practice.
