import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from zoneinfo import ZoneInfo

# Optional: RAG tool import (present if you add rag.py)
try:
    from rag import rag_find_commits  # type: ignore
except Exception:
    rag_find_commits = None  # RAG is optional

GITHUB_API = "https://api.github.com"
DEFAULT_TZ = os.getenv("AGENT_TZ", "Europe/Berlin")
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL", "openai:gpt-4o-mini")

# ---------- Utilities ----------

def _headers(token: Optional[str]) -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json, application/vnd.github.text-match+json",
        "User-Agent": "repo-qna-agent/1.0",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def to_local_iso(ts: str, tz_name: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(ZoneInfo(tz_name))
    return iso(local)

async def _json_or_error(resp: httpx.Response) -> Any:
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()

# ---------- Agent output ----------
class RepoAnswer(BaseModel):
    text: str = Field(description="Final answer, concise and factual.")
    citations: List[str] = Field(description="URLs used to produce the answer (commit, workflow run, etc.)")
    timestamps: List[str] = Field(description="Key timestamps included in the answer (UTC and local).")

# ---------- Agent dependencies ----------
@dataclass
class Deps:
    github_token: Optional[str]
    tz: str = DEFAULT_TZ

# ---------- Agent ----------
agent = Agent(
    MODEL_NAME,
    deps_type=Deps,
    output_type=RepoAnswer,
    instructions=(
        """
You are a precise GitHub repo Q&A assistant. You MUST:
- Use the provided tools to fetch data from GitHub. Never guess.
- Always cite exact sources (commit URL, workflow run URL, deployments page).
- Always include clear timestamps in BOTH UTC and the local timezone provided by deps.tz.
- If nothing is found, say "I cannot verify this." and offer the closest match with its citation.
- Keep answers concise and terminal-friendly.
        """
    ),
)

# ---------- Tools ----------
@agent.tool
async def last_commit(
    ctx: RunContext[Deps], *, owner: str, repo: str, branch: Optional[str] = None
) -> Dict[str, Any]:
    """Return info about the latest commit on a branch (default branch if not provided)."""
    async with httpx.AsyncClient(timeout=20) as client:
        # Resolve default branch if not provided
        if not branch:
            r = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers(ctx.deps.github_token))
            repo_json = await _json_or_error(r)
            branch = repo_json.get("default_branch", "main")
        r = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"sha": branch, "per_page": 1},
            headers=_headers(ctx.deps.github_token),
        )
        data = await _json_or_error(r)
        if not data:
            return {}
        c = data[0]
        sha = c["sha"]
        commit = c["commit"]
        author_name = commit.get("author", {}).get("name")
        author_login = (c.get("author") or {}).get("login")
        committed_date = commit.get("author", {}).get("date")  # ISO8601 UTC
        message = commit.get("message")
        html_url = f"https://github.com/{owner}/{repo}/commit/{sha}"
        return {
            "sha": sha,
            "message": message,
            "author_name": author_name,
            "author_login": author_login,
            "committed_date": committed_date,
            "html_url": html_url,
            "branch": branch,
        }

@agent.tool
async def find_commit(
    ctx: RunContext[Deps], *, owner: str, repo: str, query: str
) -> Dict[str, Any]:
    """Find the most recent commit whose message matches a keyword query."""
    async with httpx.AsyncClient(timeout=25) as client:
        # Try the dedicated commit search first
        q = f"repo:{owner}/{repo} {query}"
        r = await client.get(
            f"{GITHUB_API}/search/commits",
            params={"q": q, "sort": "committer-date", "order": "desc", "per_page": 1},
            headers=_headers(ctx.deps.github_token),
        )
        if r.status_code == 200:
            js = r.json()
            items = js.get("items", [])
            if items:
                it = items[0]
                sha = it["sha"]
                commit = it["commit"]
                message = commit.get("message")
                committed_date = commit.get("author", {}).get("date")
                author_name = commit.get("author", {}).get("name")
                author_login = (it.get("author") or {}).get("login")
                html_url = f"https://github.com/{owner}/{repo}/commit/{sha}"
                return {
                    "sha": sha,
                    "message": message,
                    "author_name": author_name,
                    "author_login": author_login,
                    "committed_date": committed_date,
                    "html_url": html_url,
                    "match_source": "search",
                }
        # Fallback: scan recent commits quickly
        r2 = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"per_page": 50},
            headers=_headers(ctx.deps.github_token),
        )
        data = await _json_or_error(r2)
        kw = [k.strip().lower() for k in re.split(r"\s+", query) if k.strip()]
        for c in data:
            msg = (c.get("commit", {}).get("message") or "").lower()
            if all(k in msg for k in kw):
                sha = c["sha"]
                commit = c["commit"]
                author_name = commit.get("author", {}).get("name")
                author_login = (c.get("author") or {}).get("login")
                committed_date = commit.get("author", {}).get("date")
                html_url = f"https://github.com/{owner}/{repo}/commit/{sha}"
                return {
                    "sha": sha,
                    "message": commit.get("message"),
                    "author_name": author_name,
                    "author_login": author_login,
                    "committed_date": committed_date,
                    "html_url": html_url,
                    "match_source": "list-scan",
                }
        return {}

_DEPLOY_KEYWORDS = [
    "deploy", "release", "publish", "vercel", "netlify", "render", "fly",
    "railway", "cloud run", "cloudrun", "prod", "production",
]

@agent.tool
async def last_deployment(
    ctx: RunContext[Deps], *, owner: str, repo: str, environment: Optional[str] = None
) -> Dict[str, Any]:
    """Find the last deployment time for a repo via Deployments API or Actions runs."""
    token = ctx.deps.github_token
    async with httpx.AsyncClient(timeout=25) as client:
        # (1) Deployments API
        params = {"per_page": 1}
        if environment:
            params["environment"] = environment
        r = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/deployments",
            params=params,
            headers=_headers(token),
        )
        if r.status_code == 200:
            items = r.json()
            if items:
                dep = items[0]
                created_at = dep.get("created_at")
                statuses_url = dep.get("statuses_url")
                target_url = None
                state = None
                if statuses_url:
                    rs = await client.get(statuses_url, headers=_headers(token))
                    if rs.status_code == 200:
                        sts = rs.json()
                        if sts:
                            s = sts[0]
                            state = s.get("state")
                            target_url = s.get("target_url") or s.get("log_url")
                return {
                    "when": created_at,
                    "html_url": target_url or f"https://github.com/{owner}/{repo}/deployments",
                    "source": "deployments",
                    "environment": dep.get("environment"),
                    "state": state,
                }
        # (2) Actions workflows fallback
        r2 = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs",
            params={"status": "success", "per_page": 50},
            headers=_headers(token),
        )
        data = await _json_or_error(r2)
        runs = data.get("workflow_runs", [])
        def looks_like_deploy(name: str) -> bool:
            n = (name or "").lower()
            return any(k in n for k in _DEPLOY_KEYWORDS)
        for run in runs:
            name = run.get("name") or run.get("display_title") or ""
            if looks_like_deploy(name):
                when = run.get("updated_at") or run.get("run_started_at") or run.get("created_at")
                return {
                    "when": when,
                    "html_url": run.get("html_url"),
                    "source": "actions",
                    "workflow_name": name,
                    "conclusion": run.get("conclusion"),
                }
        return {}

# ---------- Optional RAG Tool ----------
if rag_find_commits is not None:
    @agent.tool
    async def rag_find_change(
        ctx: RunContext[Deps], *, owner: str, repo: str, query: str, k: int = 4
    ) -> Dict[str, Any]:
        """Semantic search over indexed commit history (requires running rag_index.py first)."""
        results = rag_find_commits(owner, repo, query=query, k=k)  # type: ignore
        if not results:
            return {}
        best = sorted(results, key=lambda r: r.get("committed_date") or "", reverse=True)[0]
        return best

# ---------- FastAPI app ----------
app = FastAPI(title="Repo Q&A Agent", version="1.0.0")

class ChatRequest(BaseModel):
    repo: str = Field(..., description="owner/repo")
    question: str
    branch: Optional[str] = None
    environment: Optional[str] = None

@app.post("/chat", response_model=RepoAnswer)
async def chat(req: ChatRequest) -> RepoAnswer:
    token = os.getenv("GITHUB_TOKEN")
    if "/" not in req.repo:
        raise HTTPException(400, "repo must be in 'owner/repo' format")
    owner, repo = req.repo.split("/", 1)

    prompt = (
        f"Repo: {owner}/{repo}\n"
        f"Branch: {req.branch or '(default)'}\n"
        f"Environment: {req.environment or '(any)'}\n"
        f"Question: {req.question}"
    )

    result = await agent.run(prompt, deps=Deps(github_token=token))
    return result.output
