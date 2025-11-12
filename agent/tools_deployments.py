"""Deployment-related tool: find last deployment via Deployments API or Actions."""

from typing import Any, Dict, Optional

import httpx

from .core import agent
from .config import GITHUB_API
from .utils import _headers, _json_or_error, utc_local_pair

_DEPLOY_KEYWORDS = [
    "deploy",
    "release",
    "publish",
    "vercel",
    "netlify",
    "render",
    "fly",
    "railway",
    "cloud run",
    "cloudrun",
    "prod",
    "production",
    "pages-build-deployment",
]


@agent.tool
async def last_deployment(
    ctx: "Deps", *, owner: str, repo: str, environment: Optional[str] = None
) -> Dict[str, Any]:
    """Find the last deployment via Deployments API or Actions fallback."""
    token = ctx.github_token
    async with httpx.AsyncClient(timeout=25) as client:
        params: Dict[str, Any] = {"per_page": 1}
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
                when = created_at
                if statuses_url:
                    rs = await client.get(statuses_url, headers=_headers(token))
                    if rs.status_code == 200:
                        sts = rs.json()
                        if sts:
                            s = sts[0]
                            state = s.get("state")
                            target_url = s.get("target_url") or s.get("log_url")
                            when = s.get("updated_at") or s.get("created_at") or created_at
                return {
                    "when": when,
                    "when_pair": utc_local_pair(when, ctx.tz) if when else None,
                    "html_url": target_url or f"https://github.com/{owner}/{repo}/deployments",
                    "source": "deployments",
                    "environment": dep.get("environment"),
                    "state": state,
                }
        r2 = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs",
            params={"status": "success", "per_page": 50},
            headers=_headers(token),
        )
        if r2.status_code >= 400:
            return {}
        data = r2.json()
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
                    "when_pair": utc_local_pair(when, ctx.tz) if when else None,
                    "html_url": run.get("html_url"),
                    "source": "actions",
                    "workflow_name": name,
                    "conclusion": run.get("conclusion"),
                }
        return {}
