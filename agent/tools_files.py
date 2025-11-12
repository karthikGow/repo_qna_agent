"""File/code tools: last change to a file, and first commit introducing a pattern."""

from typing import Any, Dict, List, Optional

import httpx

from .core import agent
from .config import GITHUB_API
from .utils import _headers, _json_or_error, utc_local_pair
from .models import Deps
from pydantic_ai import RunContext


@agent.tool
async def last_file_change(
    ctx: RunContext[Deps], *, owner: str, repo: str, path: str, branch: Optional[str] = None
) -> Dict[str, Any]:
    """Return the most recent commit that modified the given file path."""
    token = ctx.deps.github_token
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            if not branch:
                r0 = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers(token))
                repo_json = await _json_or_error(r0)
                branch = repo_json.get("default_branch", "main")
            r = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/commits",
                params={"sha": branch, "per_page": 1, "path": path},
                headers=_headers(token),
            )
            if r.status_code >= 400:
                return {}
            data = r.json()
            if not data:
                return {}
            c = data[0]
            sha = c.get("sha")
            commit = c.get("commit", {})
            author = commit.get("author", {})
            committed_date = author.get("date")
            author_name = author.get("name")
            author_login = (c.get("author") or {}).get("login")
            message = commit.get("message")
            html_url = f"https://github.com/{owner}/{repo}/commit/{sha}"
            return {
                "sha": sha,
                "message": message,
                "author_name": author_name,
                "author_login": author_login,
                "committed_date": committed_date,
                "committed": utc_local_pair(committed_date, ctx.deps.tz) if committed_date else None,
                "html_url": html_url,
                "path": path,
                "branch": branch,
            }
        except Exception:
            return {}


@agent.tool
async def introduced_line(
    ctx: RunContext[Deps], *, owner: str, repo: str, path: str, pattern: str, branch: Optional[str] = None, max_commits: int = 200
) -> Dict[str, Any]:
    """Find the earliest commit (within a recent window) that introduced a line containing `pattern` in `path`."""
    token = ctx.deps.github_token
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            if not branch:
                r0 = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers(token))
                repo_json = await _json_or_error(r0)
                branch = repo_json.get("default_branch", "main")
            per_page = 100
            pages = max(1, (max_commits + per_page - 1) // per_page)
            commits: List[Dict[str, Any]] = []
            for page in range(1, pages + 1):
                r = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/commits",
                    params={"sha": branch, "per_page": per_page, "page": page, "path": path},
                    headers=_headers(token),
                )
                if r.status_code >= 400:
                    break
                batch = r.json() or []
                if not batch:
                    break
                commits.extend(batch)
                if len(batch) < per_page or len(commits) >= max_commits:
                    break
            if not commits:
                return {}
            for c in reversed(commits[:max_commits]):
                sha = c.get("sha")
                if not sha:
                    continue
                rc = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}",
                    headers=_headers(token),
                )
                if rc.status_code >= 400:
                    continue
                detail = rc.json()
                files = detail.get("files") or []
                introduced = False
                for f in files:
                    patch = f.get("patch")
                    if not patch:
                        continue
                    for line in patch.splitlines():
                        if line.startswith("+++") or line.startswith("---"):
                            continue
                        if line.startswith("+") and pattern in line:
                            introduced = True
                            break
                    if introduced:
                        break
                if introduced:
                    commit = detail.get("commit", {})
                    author = commit.get("author", {})
                    committed_date = author.get("date")
                    author_name = author.get("name")
                    author_login = (detail.get("author") or {}).get("login")
                    message = commit.get("message")
                    html_url = f"https://github.com/{owner}/{repo}/commit/{sha}"
                    return {
                        "sha": sha,
                        "message": message,
                        "author_name": author_name,
                        "author_login": author_login,
                        "committed_date": committed_date,
                        "committed": utc_local_pair(committed_date, ctx.deps.tz) if committed_date else None,
                        "html_url": html_url,
                        "path": path,
                        "pattern": pattern,
                        "branch": branch,
                    }
            return {}
        except Exception:
            return {}
