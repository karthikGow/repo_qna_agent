"""PR-related tool: search for a merged PR matching a query and return merge details."""

from typing import Any, Dict

import httpx

from .core import agent
from .config import GITHUB_API
from .utils import _headers, utc_local_pair
from .models import Deps
from pydantic_ai import RunContext


@agent.tool
async def find_pr_merge(
    ctx: RunContext[Deps], *, owner: str, repo: str, query: str
) -> Dict[str, Any]:
    """Find a merged PR matching a query and return merge details."""
    async with httpx.AsyncClient(timeout=25) as client:
        q = f"repo:{owner}/{repo} is:pr is:merged {query}"
        r = await client.get(
            f"{GITHUB_API}/search/issues",
            params={"q": q, "sort": "updated", "order": "desc", "per_page": 1},
            headers=_headers(ctx.deps.github_token),
        )
        if r.status_code >= 400:
            return {}
        js = r.json()
        items = js.get("items", [])
        if not items:
            return {}
        it = items[0]
        number = it.get("number")
        if not number:
            return {}
        pr_url = f"https://github.com/{owner}/{repo}/pull/{number}"
        r2 = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}",
            headers=_headers(ctx.deps.github_token),
        )
        if r2.status_code >= 400:
            when = it.get("closed_at") or it.get("updated_at") or it.get("created_at")
            return {
                "number": number,
                "title": it.get("title"),
                "author_login": (it.get("user") or {}).get("login"),
                "when": when,
                "when_pair": utc_local_pair(when, ctx.deps.tz) if when else None,
                "html_url": pr_url,
            }
        pr = r2.json()
        merged_at = pr.get("merged_at")
        merge_commit_sha = pr.get("merge_commit_sha")
        user = pr.get("user") or {}
        when = merged_at or it.get("closed_at") or it.get("updated_at") or it.get("created_at")
        out: Dict[str, Any] = {
            "number": number,
            "title": pr.get("title") or it.get("title"),
            "author_login": user.get("login") or (it.get("user") or {}).get("login"),
            "merged_at": merged_at,
            "when": when,
            "when_pair": utc_local_pair(when, ctx.deps.tz) if when else None,
            "html_url": pr_url,
        }
        if merge_commit_sha:
            out["merge_commit_sha"] = merge_commit_sha
            out["merge_commit_url"] = f"https://github.com/{owner}/{repo}/commit/{merge_commit_sha}"
        return out
