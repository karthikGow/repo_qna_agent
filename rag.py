"""RAG retrieval helper for commit history (optional).

Loads a pre-built Chroma collection and verifies retrieved commits via GitHub.
Used by agent.tools_rag if available.
"""

import os
from typing import List, Dict, Any
import httpx
from langchain_community.vectorstores import Chroma

GITHUB_API = "https://api.github.com"
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "rag_store")

def _headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "repo-qna-rag/1.0"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def load_retriever(owner: str, repo: str, k: int = 6):
    collection_name = f"{owner}/{repo}-commits"
    vs = Chroma(
        embedding_function=None,  # use stored embeddings
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )
    return vs.as_retriever(search_kwargs={"k": k})

def _verify_commit(owner: str, repo: str, sha: str) -> Dict[str, Any]:
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}", headers=_headers())
        if r.status_code != 200:
            return {}
        js = r.json()
        commit = js.get("commit", {})
        author = commit.get("author") or {}
        return {
            "sha": js.get("sha"),
            "message": commit.get("message"),
            "author_name": author.get("name"),
            "author_login": (js.get("author") or {}).get("login"),
            "committed_date": author.get("date"),
            "html_url": f"https://github.com/{owner}/{repo}/commit/{js.get('sha')}",
        }

def rag_find_commits(owner: str, repo: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
    retriever = load_retriever(owner, repo, k=k)
    docs = retriever.get_relevant_documents(query)
    out: List[Dict[str, Any]] = []
    for d in docs:
        sha = d.metadata.get("sha")
        if not sha:
            continue
        ver = _verify_commit(owner, repo, sha)
        if ver:
            out.append(ver)
    # unique by sha, keep order
    seen = set()
    uniq = []
    for x in out:
        if x["sha"] in seen:
            continue
        seen.add(x["sha"])
        uniq.append(x)
    return uniq
