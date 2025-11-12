"""Utility helpers for HTTP headers, timestamp conversion, and JSON/error handling."""

from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
from fastapi import HTTPException
from zoneinfo import ZoneInfo

from .config import DEFAULT_TZ


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


def utc_local_pair(ts_utc: str, tz_name: str = DEFAULT_TZ) -> Dict[str, str]:
    dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc_iso = iso(dt.astimezone(timezone.utc))
    local_iso = dt.astimezone(ZoneInfo(tz_name)).isoformat()
    return {"utc": utc_iso, "local": f"{local_iso} ({tz_name})"}


async def _json_or_error(resp: httpx.Response) -> Any:
    if resp.status_code >= 400:
        detail_text = None
        try:
            j = resp.json()
            detail_text = j.get("message") or resp.text
        except Exception:
            detail_text = resp.text
        rl = resp.headers.get("X-RateLimit-Remaining")
        msg_lower = (detail_text or "").lower()
        if resp.status_code in (403, 429) and (rl == "0" or "rate limit" in msg_lower):
            raise HTTPException(
                status_code=429,
                detail="GitHub API rate limit exceeded. Provide GITHUB_TOKEN or retry later.",
            )
        raise HTTPException(status_code=resp.status_code, detail=detail_text or "GitHub request failed")
    try:
        return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON from GitHub")
