## Repo-QnA-Agent — Copilot instructions

Short, actionable guidance for AI coding agents working on this repository.

1. Big picture
   - This is a small FastAPI service that answers questions about a GitHub repo using live GitHub REST calls and optional RAG (semantic index).
   - Key files:
     - `app.py` — Pydantic-AI Agent definition, tool implementations (last_commit, find_commit, last_deployment), FastAPI endpoints, and the `RepoAnswer` output model.
     - `cli.py` — simple Typer-based client that POSTs to `http://127.0.0.1:8000/chat` and prints timestamps & citations.
     - `rag_index.py` & `rag.py` — optional RAG index builder and retriever using LangChain + Chroma. RAG is optional and imported dynamically in `app.py`.
     - `README.md` — quickstart (PowerShell) and env var examples; use this for exact run commands.

2. Critical behavior to preserve
   - The Agent MUST fetch data from GitHub and always include exact citations (commit/workflow URLs). See the `agent.instructions` string in `app.py` — do not remove this requirement.
   - Answers must include UTC and local timezone timestamps. The default timezone is `Europe/Berlin` and is exposed via `AGENT_TZ` / deps.tz.
   - Do NOT hallucinate facts. If data is not found the code expects the agent to respond with: "I cannot verify this." and provide closest matches with citations.

3. How to run & dev workflows (copy from README)
   - Setup (PowerShell example): create venv, activate, pip install -r requirements.txt, then `uvicorn app:app --reload --port 8000`.
   - CLI usage: `python cli.py --repo owner/repo "What was implemented in the last commit and by whom?"` (the CLI assumes the API runs on port 8000).
   - Build RAG index: `python rag_index.py --repo owner/repo --max-commits 1500`. Use `EMBED_MODE=local` to switch to local HF embeddings.

4. Environment & integrations
   - Required for best functionality: `GITHUB_TOKEN` (read access). Without it, some GitHub REST endpoints will be rate-limited or unavailable.
   - Model selection controlled via `PYDANTIC_AI_MODEL` env var (default `openai:gpt-4o-mini`). README shows OpenAI, DeepSeek, or Ollama examples.
   - RAG persistence: `RAG_PERSIST_DIR` defaults to `rag_store`. Chroma expects persisted embeddings if using `rag.py` retriever.

5. Project-specific idioms and patterns
   - Agent tools are regular async functions decorated with `@agent.tool` and receive `RunContext[Deps]`. The `Deps` dataclass (in `app.py`) carries `github_token` and `tz`.
   - Optional imports: `from rag import rag_find_commits` is wrapped in a try/except — keep this pattern when adding optional dependencies.
   - HTTP clients: code uses `httpx` both async and sync; preserve explicit timeouts (e.g., 20s) and `_headers(token)` helper usage.
   - Output model: `RepoAnswer` (Pydantic) is the canonical output shape — if you change the shape, update `cli.py` and FastAPI response_model accordingly.
   - Timestamps: the helper `to_local_iso` converts GitHub ISO strings to the configured timezone; reuse it for any new timestamp presentation logic.

6. Editing guidance for contributors/AI agents
   - When adding new Agent tools:
     - Register with `@agent.tool` and accept a `RunContext[Deps]` if you need secrets/config.
     - Return serializable dicts compatible with the `RepoAnswer` shape or with intermediate tool outputs the main agent can consume.
   - When touching RAG:
     - `rag_index.py` builds embeddings and persists to Chroma. Keep `EMBED_MODE` logic intact and preserve `PERSIST_DIR` usage.
     - `rag.py` expects stored embeddings (embedding_function=None in Chroma) and verifies SHAs via GitHub commits endpoint.
   - When modifying network calls:
     - Keep `_headers()` / `_headers(token)` semantics so Authorization is consistently applied.
     - Preserve user-agent strings (useful for GitHub telemetry) or update them consciously.

7. Quick examples (exact lines to reference)
   - Use `last_commit` tool in `app.py` to fetch latest commit and cite the returned `html_url` (see function `last_commit`).
   - Use `last_deployment` to prefer the Deployments API, then fallback to Actions runs (see `_DEPLOY_KEYWORDS` and `last_deployment`).

8. Tests & checks
   - There are no unit tests in repo. Validate manually by running `uvicorn` + `python cli.py` and checking responses.
   - After edits, ensure `cli.py` can parse the FastAPI JSON shape (`text`, `timestamps`, `citations`).

If anything above is unclear or you'd like me to expand any section (examples for adding a new tool, test harness, or CI checks), tell me which area to refine.
