"""Optional RAG tool wrapper: semantic commit retrieval if index is present."""

from typing import Any, Dict

from .core import agent

try:
    from rag import rag_find_commits  # type: ignore
except Exception:
    rag_find_commits = None


if rag_find_commits is not None:
    @agent.tool
    async def rag_find_change(
        ctx, *, owner: str, repo: str, query: str, k: int = 4
    ) -> Dict[str, Any]:
        results = rag_find_commits(owner, repo, query=query, k=k)  # type: ignore
        if not results:
            return {}
        best = sorted(results, key=lambda r: r.get("committed_date") or "", reverse=True)[0]
        return best
