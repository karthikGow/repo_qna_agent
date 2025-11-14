"""Commit-related tools: latest commit and keyword-based commit search."""

from typing import Any, Dict, Optional, List
import re

import httpx

from .core import agent
from .config import GITHUB_API
from .utils import _headers, _json_or_error, utc_local_pair
from .models import Deps
from pydantic_ai import RunContext


# Very small stopword list to avoid matching on generic words like "when", "did", "we".
_STOPWORDS = {
    "when",
    "did",
    "do",
    "does",
    "we",
    "you",
    "the",
    "a",
    "an",
    "is",
    "was",
    "were",
    "to",
    "for",
    "in",
    "on",
    "of",
    "and",
    "or",
}


def _keywords_from_query(query: str) -> List[str]:
    """Extract meaningful lowercase keywords from a natural-language query."""
    raw = re.split(r"\s+", query)
    out: List[str] = []
    for k in raw:
        k = k.strip().lower()
        if not k:
            continue
        # Strip punctuation and non-alphanumerics.
        k = re.sub(r"[^a-z0-9]+", "", k)
        if not k:
            continue
        if k in _STOPWORDS:
            continue
        if len(k) < 3:
            continue
        out.append(k)
    return out


@agent.tool
async def last_commit(
    ctx: RunContext[Deps], *, owner: str, repo: str, branch: Optional[str] = None
) -> Dict[str, Any]:
    """Return info about the latest commit on a branch (default branch if not provided)."""
    async with httpx.AsyncClient(timeout=20) as client:
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
        committed_date = commit.get("author", {}).get("date")
        message = commit.get("message")
        html_url = f"https://github.com/{owner}/{repo}/commit/{sha}"
        return {
            "sha": sha,
            "message": message,
            "author_name": author_name,
            "author_login": author_login,
            "committed_date": committed_date,
            "committed": utc_local_pair(committed_date, ctx.deps.tz),
            "html_url": html_url,
            "branch": branch,
        }


@agent.tool
async def find_commit(
    ctx: RunContext[Deps], *, owner: str, repo: str, query: str
) -> Dict[str, Any]:
    """Find the most recent commit whose message matches a keyword query."""
    async with httpx.AsyncClient(timeout=25) as client:
        # Let GitHub search try with the raw query first.
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
                    "committed": utc_local_pair(committed_date, ctx.deps.tz),
                    "html_url": html_url,
                    "match_source": "search",
                }
        r2 = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"per_page": 50},
            headers=_headers(ctx.deps.github_token),
        )
        if r2.status_code >= 400:
            return {}
        data = r2.json()
        kw = _keywords_from_query(query)
        if not kw:
            return {}
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
                    "committed": utc_local_pair(committed_date, ctx.deps.tz),
                    "html_url": html_url,
                    "match_source": "list-scan",
                }
        return {}
