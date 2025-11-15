"""
Server-side utility: build a local semantic index of a GitHub repo's commit
history for RAG.

- Fetches recent commits via GitHub REST (no git clone needed).
- Creates LangChain Documents with rich metadata (sha, author, date, url).
- Embeds with OpenAI (default) or a local HF model.
- Persists to a Chroma vector store under `rag_store/`.

Dependencies / flow:
- Reads `GITHUB_TOKEN` (for GitHub API), `OPENAI_API_KEY` / embedding env vars.
- The resulting index is later read by `rag.py`, which is wrapped by
  `agent.tools_rag` for optional semantic commit search.
"""
import os
import math
import httpx
import argparse
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


# Embeddings (choose OpenAI or local HF)
EMBED_MODE = os.getenv("EMBED_MODE", "openai").lower()
if EMBED_MODE == "local":
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDINGS = HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
else:
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

GITHUB_API = "https://api.github.com"
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "rag_store")

def _collection_name(owner: str, repo: str) -> str:
    """Return a Chroma-safe collection name for this repo.

    Must match the naming scheme used in `rag.py` to ensure we open
    the same collection.
    """
    return f"{owner}_{repo}-commits"

def _headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "repo-qna-rag/1.0"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def fetch_commits(owner: str, repo: str, max_commits: int = 1000) -> List[Dict[str, Any]]:
    commits: List[Dict[str, Any]] = []
    per_page = 100
    pages = max(1, (max_commits + per_page - 1) // per_page)
    with httpx.Client(timeout=20) as client:
        r0 = client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers())
        r0.raise_for_status()
        default_branch = r0.json().get("default_branch", "main")
        for page in range(1, pages + 1):
            r = client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/commits",
                params={"sha": default_branch, "per_page": per_page, "page": page},
                headers=_headers(),
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            commits.extend(batch)
            if len(batch) < per_page or len(commits) >= max_commits:
                break
    return commits[:max_commits]

def to_documents(owner: str, repo: str, rows: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for c in rows:
        sha = c.get("sha")
        commit = c.get("commit", {})
        msg = commit.get("message") or ""
        author = commit.get("author") or {}
        author_name = author.get("name")
        author_date = author.get("date")  # ISO-8601
        author_login = (c.get("author") or {}).get("login")
        url = f"https://github.com/{owner}/{repo}/commit/{sha}"
        content = (
            f"SHA: {sha}\n"
            f"Author: {author_name} ({author_login})\n"
            f"Date: {author_date}\n"
            f"Message:\n{msg}\n"
        )
        metadata = {
            "sha": sha,
            "author_name": author_name,
            "author_login": author_login,
            "date": author_date,
            "url": url,
            "repo": f"{owner}/{repo}",
        }
        docs.append(Document(page_content=content, metadata=metadata))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    out = []
    for d in docs:
        out.extend(splitter.split_documents([d]))
    return out

def build_index(owner: str, repo: str, max_commits: int = 1000) -> None:
    rows = fetch_commits(owner, repo, max_commits=max_commits)
    if not rows:
        raise SystemExit("No commits fetched. Check repo or token.")
    docs = to_documents(owner, repo, rows)
    collection_name = _collection_name(owner, repo)
    vs = Chroma.from_documents(
        documents=docs,
        embedding=EMBEDDINGS,
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )
    vs.persist()
    print(f"Indexed {len(docs)} chunks for {owner}/{repo} -> {PERSIST_DIR} (collection={collection_name})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--max-commits", type=int, default=1000)
    args = ap.parse_args()
    owner, repo = args.repo.split("/", 1)
    build_index(owner, repo, max_commits=args.max_commits)
