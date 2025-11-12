# GitHub Repo Q&A Agent (FastAPI + PydanticAI + optional LangChain RAG)

Answers repo questions like:
- "What was implemented in the last commit and by whom?"
- "When was the last deployment?"
- "When did we fix the favicon bug?"
- "When did we refactor the Homepage layout?"

Always prints UTC + Europe/Berlin timestamps and citations (commit/workflow/PR URLs).

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

### Git Identity (optional)
```powershell
git config --global user.name "Your Name"
git config --global user.email "<your-noreply-or-safe-email>"
```
