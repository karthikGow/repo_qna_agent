# GitHub Repo Q&A Agent (FastAPI + PydanticAI + optional LangChain RAG)

Answers repo questions like:
- “What was implemented in the last commit and by whom?”
- “When was the last deployment?”
- “When did we fix the favicon bug?”
- “When did we refactor the Homepage layout?”

**Always** prints UTC + Europe/Berlin timestamps and **citations** (commit/workflow URLs).

## Quickstart (Windows 11, PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install tzdata
```

Pick one LLM path:

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
python cli.py --repo owner/repo "When was the last deployment?"
python cli.py --repo owner/repo "When did we fix the favicon bug?"
python cli.py --repo owner/repo "When did we refactor the Homepage layout?"
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
