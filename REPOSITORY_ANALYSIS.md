# Comprehensive Repository Analysis: Repo Q&A Agent

## Repository Overview

This repository implements a **GitHub Repository Q&A Agent** - a sophisticated tool that answers natural language questions about GitHub repositories using a combination of FastAPI, PydanticAI, and optional Retrieval Augmented Generation (RAG) with LangChain. The system provides accurate, cited answers about repository history, deployments, commits, and code changes.

### Key Features
- **FastAPI Web API**: Lightweight REST API for handling questions
- **PydanticAI Agent**: AI-powered question answering with specialized tools
- **GitHub Integration**: Direct API calls to GitHub for real-time data
- **Optional RAG System**: Semantic search over commit history using vector embeddings
- **CLI Client**: Command-line interface for easy interaction
- **Timezone Support**: UTC and local timezone timestamps
- **Citation System**: Always includes exact source URLs

### Repository Goals
The primary goal is to provide developers with a natural language interface to query GitHub repository information, eliminating the need to manually browse through commits, PRs, and deployment history. It serves as a "repository assistant" that can answer questions like:
- "What was implemented in the last commit?"
- "When was the last deployment to production?"
- "When did we fix the favicon bug?"
- "When did we add this specific line of code?"

## Dependencies and Installation

### Core Dependencies
All dependencies are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

#### Python Packages
- **fastapi>=0.115**: Web framework for the API endpoints
- **uvicorn>=0.30**: ASGI server to run the FastAPI application
- **httpx>=0.27**: HTTP client for making API calls to GitHub
- **pydantic>=2.8**: Data validation and serialization
- **pydantic-ai>=1.12**: AI agent framework with tool integration
- **typer>=0.12**: Command-line interface framework
- **python-dotenv>=1.0**: Environment variable loading from .env files
- **openai>=1.47**: OpenAI API client (for LLM integration)
- **tzdata>=2024.1**: Timezone data for accurate timestamp handling

#### RAG System Dependencies (Optional)
- **langchain>=0.3**: Framework for building LLM applications
- **langchain-community>=0.3**: Community extensions for LangChain
- **langchain-openai>=0.2**: OpenAI integration for LangChain
- **chromadb>=0.5**: Vector database for storing embeddings
- **faiss-cpu>=1.8.0.post1**: Vector similarity search library
- **sentence-transformers>=3.0**: Local embedding models
- **rank-bm25>=0.2.2**: BM25 ranking algorithm
- **tiktoken>=0.7**: Tokenization for OpenAI models

### Installation Instructions
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.venv\Scripts\Activate.ps1` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Install timezone data: `pip install tzdata`

### Environment Variables
The application requires several environment variables for proper operation:

#### Required
- **GITHUB_TOKEN**: GitHub Personal Access Token (read access) - improves rate limits
- **PYDANTIC_AI_MODEL**: AI model to use (e.g., "openai:gpt-4o-mini")
- **OPENAI_API_KEY**: API key for OpenAI (or compatible service)

#### Optional
- **AGENT_TZ**: Timezone for timestamps (default: "Europe/Berlin")
- **EMBED_MODE**: "openai" or "local" for RAG embeddings
- **RAG_PERSIST_DIR**: Directory for RAG index storage (default: "rag_store")

### Potential Version Conflicts
- **Pydantic versions**: Must use Pydantic v2.x (not v1.x) as required by pydantic-ai
- **LangChain versions**: All langchain-* packages should be compatible (all >=0.3)
- **ChromaDB**: Version 0.5+ required for latest features
- **OpenAI client**: Version 1.x required for current API compatibility

## Repository Structure and File Interconnections

### Core Architecture
```
CLI/HTTP Request → FastAPI (/chat) → PydanticAI Agent → Tools → GitHub API
                                      ↓
                                Citations + Timestamps
```

### File Relationships

#### Main Entry Points
- **`app.py`**: FastAPI application entry point
  - Imports: `agent` (from agent package), `RepoAnswer`, `Deps`, `DEFAULT_TZ`
  - Calls: `agent.run()` with user prompt and dependencies
  - Serves: `/health` and `/chat` endpoints

- **`cli.py`**: Command-line client
  - Depends on: `app.py` API endpoints
  - Uses: `httpx` for HTTP requests, `typer` for CLI interface
  - Purpose: Send questions to API and display formatted responses

#### Agent Package (`agent/`)
- **`agent/__init__.py`**: Package initializer
  - Imports: All tool modules (auto-registers @agent.tool functions)
  - Exports: `agent`, `RepoAnswer`, `Deps`, `DEFAULT_TZ`

- **`agent/core.py`**: AI agent configuration
  - Imports: `MODEL_NAME` from config, `RepoAnswer` and `Deps` from models
  - Creates: PydanticAI `Agent` instance with instructions and tool registry

- **`agent/config.py`**: Configuration management
  - Loads: Environment variables via python-dotenv
  - Defines: Constants like `GITHUB_API`, `DEFAULT_TZ`, `MODEL_NAME`

- **`agent/models.py`**: Data structures
  - Defines: `RepoAnswer` (output schema), `Deps` (dependency injection)
  - Imports: `DEFAULT_TZ` from config

- **`agent/utils.py`**: Shared utilities
  - Provides: HTTP headers, timestamp conversion, error handling
  - Used by: All tool modules

#### Tool Modules
All tool files follow similar patterns:
- Import: `agent` from core, `Deps`, utility functions
- Define: Async functions decorated with `@agent.tool`
- Use: `httpx` for GitHub API calls, `RunContext[Deps]` for dependencies

- **`agent/tools_commits.py`**: Commit-related queries
- **`agent/tools_deployments.py`**: Deployment information
- **`agent/tools_files.py`**: File change tracking
- **`agent/tools_prs.py`**: Pull request searches
- **`agent/tools_rag.py`**: Optional RAG integration

#### RAG System (Optional)
- **`rag_index.py`**: Index builder
  - Fetches: Commit data from GitHub API
  - Creates: LangChain Documents with embeddings
  - Stores: In Chroma vector database

- **`rag.py`**: Retrieval helper
  - Loads: Pre-built Chroma index
  - Provides: Semantic search over commits
  - Verifies: Results against GitHub API

### Data Flow
1. **User Input**: Question via CLI or HTTP POST to `/chat`
2. **API Processing**: FastAPI validates request, extracts repo/question/branch/environment
3. **Agent Execution**: PydanticAI agent receives prompt, selects appropriate tools
4. **Tool Execution**: Tools make GitHub API calls using httpx
5. **Response Formation**: Agent synthesizes answer with citations and timestamps
6. **Output**: Formatted response with text, citations, and timestamps

## Python Files Analysis with Inline Comments

### `app.py` - FastAPI Application Entry Point

```python
"""
FastAPI entrypoint: defines the API, health route, and chat endpoint.
All agent logic and GitHub tools live in the `agent/` package to keep this file slim.
"""
import os  # Standard library for environment variable access
from typing import Optional  # Type hints for optional parameters
from fastapi import FastAPI, HTTPException  # Web framework and error handling
from pydantic import BaseModel, Field  # Data validation models

from agent import agent, RepoAnswer, DEFAULT_TZ, Deps  # Agent package imports

# Create FastAPI application instance with metadata
app = FastAPI(title="Repo Q&A Agent", version="1.3.0")

# Health check endpoint - simple liveness probe
@app.get("/health")
async def health() -> dict:
    return {"ok": True}

# Pydantic model for chat endpoint request validation
class ChatRequest(BaseModel):
    repo: str = Field(..., description="owner/repo")  # Repository identifier
    question: str  # User's question about the repo
    branch: Optional[str] = None  # Optional branch specification
    environment: Optional[str] = None  # Optional deployment environment

# Main chat endpoint - handles user questions
@app.post("/chat", response_model=RepoAnswer)
async def chat(req: ChatRequest) -> RepoAnswer:
    # Get GitHub token from environment (may be None for public repos)
    token = os.getenv("GITHUB_TOKEN")
    # Get timezone setting, fallback to default
    tz = os.getenv("AGENT_TZ", DEFAULT_TZ)
    
    # Validate repository format (must be owner/repo)
    if "/" not in req.repo:
        raise HTTPException(400, "repo must be in 'owner/repo' format")
    owner, repo = req.repo.split("/", 1)  # Split into components

    # Construct prompt for the AI agent with context
    prompt = (
        f"Repo: {owner}/{repo}\n"
        f"Branch: {req.branch or '(default)'}\n"
        f"Environment: {req.environment or '(any)'}\n"
        f"Question: {req.question}"
    )

    try:
        # Execute agent with prompt and dependencies
        result = await agent.run(prompt, deps=Deps(github_token=token, tz=tz))
        return result.output  # Return the agent's response
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Wrap any other errors as 502 Bad Gateway
        raise HTTPException(502, f"Agent error: {e}")
```

**Purpose**: This file serves as the web API entry point, handling HTTP requests and delegating to the AI agent. It keeps the API layer thin by importing all complex logic from the `agent` package.

### `cli.py` - Command Line Interface Client

```python
"""Tiny CLI client for the Repo Q&A Agent.

Posts questions to the FastAPI server and prints the answer with timestamps
and citations. Supports interactive mode if no question is provided.
"""

from typing import Optional  # Type hints for optional CLI arguments
import typer  # CLI framework for building command-line applications
import httpx  # HTTP client for API communication

# Default API endpoint (localhost development)
DEFAULT_API_URL = "http://127.0.0.1:8000/chat"
# Create Typer application instance
cli = typer.Typer(add_completion=False)

# Main CLI command with argument definitions
@cli.command()
def main(
    repo: str = typer.Option(..., help="owner/repo"),  # Required repo identifier
    question: str = typer.Argument(None, help="Your question. If omitted, enter interactive mode."),  # Optional question
    branch: Optional[str] = typer.Option(None, help="Branch to consider for last commit, e.g. main"),  # Optional branch
    environment: Optional[str] = typer.Option(None, help="Deployment environment, e.g. prod or staging"),  # Optional environment
    api_url: str = typer.Option(DEFAULT_API_URL, help="Agent API URL"),  # API endpoint override
):
    # If no question provided, enter interactive mode
    if question is None:
        typer.echo(f"Chatting with repo {repo}. Type 'exit' to quit.\n")
        while True:
            try:
                # Prompt user for input
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break  # Exit on Ctrl+C or EOF
            # Check for exit commands
            if not q or q.lower() in {"exit", "quit"}:
                break
            # Send question to API
            _ask(repo, q, api_url=api_url, branch=branch, environment=environment)
    else:
        # Send single question to API
        _ask(repo, question, api_url=api_url, branch=branch, environment=environment)

# Helper function to send question to API
def _ask(repo: str, q: str, *, api_url: str, branch: Optional[str], environment: Optional[str]):
    # Prepare request payload
    payload = {"repo": repo, "question": q}
    if branch:
        payload["branch"] = branch
    if environment:
        payload["environment"] = environment
    
    try:
        # Send POST request to API with 30-second timeout
        resp = httpx.post(api_url, json=payload, timeout=30.0)
        # Check for HTTP errors
        if resp.status_code >= 400:
            typer.secho(f"Error: {resp.status_code} {resp.text}", fg=typer.colors.RED)
            return
        # Parse JSON response
        data = resp.json()
        # Display answer in green
        typer.secho(data["text"], fg=typer.colors.GREEN)
        # Display timestamps if present
        if data.get("timestamps"):
            typer.echo("\nTimestamps:")
            for t in data["timestamps"]:
                typer.echo(f"  - {t}")
        # Display citations if present
        if data.get("citations"):
            typer.echo("\nCitations:")
            for c in data["citations"]:
                typer.echo(f"  - {c}")
        typer.echo("")  # Add spacing
    except Exception as e:
        # Display any client-side errors
        typer.secho(f"Client error: {e}", fg=typer.colors.RED)

# Standard Python CLI entry point
if __name__ == "__main__":
    cli()
```

**Purpose**: Provides a user-friendly command-line interface to interact with the Repo Q&A Agent API. Supports both single questions and interactive chat mode.

### `rag_index.py` - RAG Index Builder

```python
"""
Build a local semantic index of a GitHub repo's commit history for RAG.

- Fetches recent commits via GitHub REST (no git clone needed)
- Creates LangChain Documents with rich metadata (sha, author, date, url)
- Embeds with OpenAI (default) or a local HF model
- Persists to Chroma at ./rag_store
"""
import os  # Environment variable access
import math  # Mathematical calculations for pagination
import httpx  # HTTP client for GitHub API
import argparse  # Command-line argument parsing
from typing import List, Dict, Any  # Type hints
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Text chunking
from langchain_community.vectorstores import Chroma  # Vector database
try:
    from langchain_core.documents import Document  # Document structure
except ImportError:
    from langchain.schema import Document  # Fallback import

# Determine embedding mode from environment (openai or local)
EMBED_MODE = os.getenv("EMBED_MODE", "openai").lower()
if EMBED_MODE == "local":
    # Use local HuggingFace embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDINGS = HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
else:
    # Use OpenAI embeddings (default)
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

# GitHub API base URL
GITHUB_API = "https://api.github.com"
# Directory for storing the RAG index
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "rag_store")

# Function to generate HTTP headers for GitHub API
def _headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "repo-qna-rag/1.0"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"  # Add auth if token present
    return h

# Fetch commits from GitHub API with pagination
def fetch_commits(owner: str, repo: str, max_commits: int = 1000) -> List[Dict[str, Any]]:
    commits: List[Dict[str, Any]] = []
    per_page = 100  # GitHub API limit per page
    pages = max(1, (max_commits + per_page - 1) // per_page)  # Calculate pages needed
    
    with httpx.Client(timeout=20) as client:
        # First, get repo info to find default branch
        r0 = client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers())
        r0.raise_for_status()
        default_branch = r0.json().get("default_branch", "main")
        
        # Fetch commits page by page
        for page in range(1, pages + 1):
            r = client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/commits",
                params={"sha": default_branch, "per_page": per_page, "page": page},
                headers=_headers(),
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break  # No more commits
            commits.extend(batch)
            if len(batch) < per_page or len(commits) >= max_commits:
                break  # Reached limit or last page
    return commits[:max_commits]  # Return up to max_commits

# Convert commit data to LangChain Documents
def to_documents(owner: str, repo: str, rows: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for c in rows:
        sha = c.get("sha")
        commit = c.get("commit", {})
        msg = commit.get("message") or ""
        author = commit.get("author", {})
        author_name = author.get("name")
        author_date = author.get("date")  # ISO-8601 format
        author_login = (c.get("author") or {}).get("login")
        url = f"https://github.com/{owner}/{repo}/commit/{sha}"
        
        # Create rich content string with commit details
        content = (
            f"SHA: {sha}\n"
            f"Author: {author_name} ({author_login})\n"
            f"Date: {author_date}\n"
            f"Message:\n{msg}\n"
        )
        
        # Add metadata for retrieval
        metadata = {
            "sha": sha,
            "author_name": author_name,
            "author_login": author_login,
            "date": author_date,
            "url": url,
            "repo": f"{owner}/{repo}",
        }
        docs.append(Document(page_content=content, metadata=metadata))
    
    # Split documents into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    out = []
    for d in docs:
        out.extend(splitter.split_documents([d]))
    return out

# Main function to build and persist the RAG index
def build_index(owner: str, repo: str, max_commits: int = 1000) -> None:
    rows = fetch_commits(owner, repo, max_commits=max_commits)
    if not rows:
        raise SystemExit("No commits fetched. Check repo or token.")
    
    docs = to_documents(owner, repo, rows)
    collection_name = f"{owner}/{repo}-commits"
    
    # Create Chroma vector store with embeddings
    vs = Chroma.from_documents(
        documents=docs,
        embedding=EMBEDDINGS,
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )
    vs.persist()  # Save to disk
    
    print(f"Indexed {len(docs)} chunks for {owner}/{repo} -> {PERSIST_DIR} (collection={collection_name})")

# Command-line interface
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--max-commits", type=int, default=1000)
    args = ap.parse_args()
    owner, repo = args.repo.split("/", 1)
    build_index(owner, repo, max_commits=args.max_commits)
```

**Purpose**: Builds a semantic search index of repository commit history using vector embeddings, enabling more sophisticated queries through the RAG system.

### `rag.py` - RAG Retrieval Helper

```python
"""RAG retrieval helper for commit history (optional).

Loads a pre-built Chroma collection and verifies retrieved commits via GitHub.
Used by agent.tools_rag if available.
"""

import os  # Environment variables
from typing import List, Dict, Any  # Type hints
import httpx  # HTTP client
from langchain_community.vectorstores import Chroma  # Vector database

# GitHub API base URL
GITHUB_API = "https://api.github.com"
# RAG index storage directory
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "rag_store")

# Generate headers for GitHub API requests
def _headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "repo-qna-rag/1.0"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

# Load retriever from persisted Chroma index
def load_retriever(owner: str, repo: str, k: int = 6):
    collection_name = f"{owner}/{repo}-commits"
    vs = Chroma(
        embedding_function=None,  # Use stored embeddings
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )
    return vs.as_retriever(search_kwargs={"k": k})

# Verify commit details against GitHub API
def _verify_commit(owner: str, repo: str, sha: str) -> Dict[str, Any]:
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}", headers=_headers())
        if r.status_code != 200:
            return {}  # Return empty if not found
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

# Main RAG search function
def rag_find_commits(owner: str, repo: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
    retriever = load_retriever(owner, repo, k=k)
    docs = retriever.get_relevant_documents(query)
    out: List[Dict[str, Any]] = []
    
    for d in docs:
        sha = d.metadata.get("sha")
        if not sha:
            continue
        # Verify commit exists and get fresh data
        ver = _verify_commit(owner, repo, sha)
        if ver:
            out.append(ver)
    
    # Remove duplicates by SHA, preserve order
    seen = set()
    uniq = []
    for x in out:
        if x["sha"] in seen:
            continue
        seen.add(x["sha"])
        uniq.append(x)
    return uniq
```

**Purpose**: Provides retrieval functionality for the RAG system, loading pre-built indexes and performing semantic search over commit history with verification against live GitHub data.

### `agent/__init__.py` - Agent Package Initializer

```python
"""Agent package public API.

Exposes the configured `agent`, output and dependency models, and default timezone.
Imports tool modules so their `@agent.tool` registrations execute at import time.
"""

from .core import agent  # The main PydanticAI agent instance
from .models import RepoAnswer, Deps  # Data models
from .config import DEFAULT_TZ  # Default timezone constant

# Import tool modules so their @agent.tool registrations run on import
from . import tools_commits  # noqa: F401 - Commit-related tools
from . import tools_deployments  # noqa: F401 - Deployment tools
from . import tools_prs  # noqa: F401 - Pull request tools
from . import tools_files  # noqa: F401 - File change tools
from . import tools_rag  # noqa: F401 - RAG integration (optional)
```

**Purpose**: Serves as the public interface for the agent package, ensuring all tools are registered with the agent upon import.

### `agent/config.py` - Configuration Management

```python
"""Configuration and environment defaults for the agent.

Loads environment variables (.env via python-dotenv) and exposes constants used
across tools and the agent core.
"""

import os  # Environment variable access
from dotenv import load_dotenv  # Load .env files

load_dotenv()  # Load environment variables from .env file

# GitHub API base URL
GITHUB_API = "https://api.github.com"
# Default timezone for timestamp conversions
DEFAULT_TZ = os.getenv("AGENT_TZ", "Europe/Berlin")
# AI model to use for the agent
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL", "openai:gpt-4o-mini")
```

**Purpose**: Centralizes configuration management, loading environment variables and providing default values for the entire agent system.

### `agent/core.py` - Agent Core Configuration

```python
"""Agent core: creates and configures the PydanticAI `agent`.

The tools import this object and register their functions via `@agent.tool`.
"""

from pydantic_ai import Agent  # AI agent framework

from .config import MODEL_NAME  # Model configuration
from .models import RepoAnswer, Deps  # Data models

# Create and configure the PydanticAI agent
agent = Agent(
    MODEL_NAME,  # AI model to use
    deps_type=Deps,  # Dependency injection type
    output_type=RepoAnswer,  # Expected output structure
    instructions=(
        """
You are a precise GitHub repo Q&A assistant. You MUST:
- Use the provided tools to fetch data from GitHub. Never guess.
- Always cite exact sources (commit URL, workflow run URL, PR URL) in `citations`.
- Always include clear timestamps in BOTH UTC and the local timezone from deps.tz.
- When a tool returns fields like `committed.utc/local` or `when_pair.utc/local`, USE THEM directly.
- If the request or the user mentions an environment (e.g., prod/staging), pass it to last_deployment(environment=...).
        - For "fix/refactor" questions, first try find_commit; if not found, try find_pr_merge; if nothing is found, say: "I cannot verify this." and offer the closest match with its citation.
- For file-specific questions, try last_file_change(path=...) to get the latest change to a file. For code-introduction questions (e.g., "when did we add @app.get('/health')?"), try introduced_line(path=..., pattern=...).
- Keep answers concise and terminal-friendly.
        """
    ),
)
```

**Purpose**: Defines the core AI agent with specific instructions for how to behave when answering repository questions, emphasizing accuracy, citations, and proper tool usage.

### `agent/models.py` - Data Models

```python
"""Data models used by the agent: output schema and dependency container."""

from dataclasses import dataclass  # Simple data structures
from typing import List, Optional  # Type hints

from pydantic import BaseModel, Field  # Data validation

from .config import DEFAULT_TZ  # Default timezone

# Output model for agent responses
class RepoAnswer(BaseModel):
    text: str = Field(description="Final answer, concise and factual.")  # Main answer text
    citations: List[str] = Field(description="URLs used to produce the answer (commit, workflow run, etc.)")  # Source URLs
    timestamps: List[str] = Field(description="Key timestamps included in the answer (UTC and local).")  # Formatted timestamps

# Dependency injection container
@dataclass
class Deps:
    github_token: Optional[str]  # GitHub API token (optional)
    tz: str = DEFAULT_TZ  # Timezone for timestamps
```

**Purpose**: Defines the data structures used throughout the agent system for type safety and validation.

### `agent/tools_commits.py` - Commit Tools

```python
"""Commit-related tools: latest commit and keyword-based commit search."""

from typing import Any, Dict, Optional  # Type hints
import re  # Regular expressions for text matching

import httpx  # HTTP client

from .core import agent  # Agent instance for tool registration
from .config import GITHUB_API  # GitHub API base URL
from .utils import _headers, _json_or_error, utc_local_pair  # Utility functions
from .models import Deps  # Dependency model
from pydantic_ai import RunContext  # Agent execution context

# Tool to get the latest commit on a branch
@agent.tool
async def last_commit(
    ctx: RunContext[Deps], *, owner: str, repo: str, branch: Optional[str] = None
) -> Dict[str, Any]:
    """Return info about the latest commit on a branch (default branch if not provided)."""
    async with httpx.AsyncClient(timeout=20) as client:
        if not branch:
            # Fetch default branch if not specified
            r = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers(ctx.deps.github_token))
            repo_json = await _json_or_error(r)
            branch = repo_json.get("default_branch", "main")
        
        # Fetch latest commit on the branch
        r = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"sha": branch, "per_page": 1},
            headers=_headers(ctx.deps.github_token),
        )
        data = await _json_or_error(r)
        if not data:
            return {}
        
        c = data[0]  # First (latest) commit
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
            "committed": utc_local_pair(committed_date, ctx.deps.tz),  # Convert to UTC + local
            "html_url": html_url,
            "branch": branch,
        }

# Tool to find commits by keyword search
@agent.tool
async def find_commit(
    ctx: RunContext[Deps], *, owner: str, repo: str, query: str
) -> Dict[str, Any]:
    """Find the most recent commit whose message matches a keyword query."""
    async with httpx.AsyncClient(timeout=25) as client:
        # First try GitHub's commit search API
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
        
        # Fallback: scan recent commits manually
        r2 = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            params={"per_page": 50},
            headers=_headers(ctx.deps.github_token),
        )
        
        if r2.status_code >= 400:
            return {}
        
        data = r2.json()
        # Split query into keywords and search case-insensitively
        kw = [k.strip().lower() for k in re.split(r"\s+", query) if k.strip()]
        
        for c in data:
            msg = (c.get("commit", {}).get("message") or "").lower()
            if all(k in msg for k in kw):  # All keywords must be present
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
        
        return {}  # No matching commit found
```

**Purpose**: Provides tools for querying commit information, including getting the latest commit and searching for commits by message content.

### `agent/tools_deployments.py` - Deployment Tools

```python
"""Deployment-related tool: find last deployment via Deployments API or Actions."""

from typing import Any, Dict, Optional  # Type hints

import httpx  # HTTP client

from .core import agent  # Agent instance
from .config import GITHUB_API  # API base URL
from .utils import _headers, _json_or_error, utc_local_pair  # Utilities
from .models import Deps  # Dependencies
from pydantic_ai import RunContext  # Execution context

# Keywords that indicate deployment-related workflows
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

# Tool to find the last deployment
@agent.tool
async def last_deployment(
    ctx: RunContext[Deps], *, owner: str, repo: str, environment: Optional[str] = None
) -> Dict[str, Any]:
    """Find the last deployment via Deployments API or Actions fallback."""
    token = ctx.deps.github_token
    async with httpx.AsyncClient(timeout=25) as client:
        params: Dict[str, Any] = {"per_page": 1}
        if environment:
            params["environment"] = environment
        
        # First try GitHub Deployments API
        r = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/deployments",
            params=params,
            headers=_headers(token),
        )
        
        if r.status_code == 200:
            items = r.json()
            if items:
                dep = items[0]  # Latest deployment
                created_at = dep.get("created_at")
                statuses_url = dep.get("statuses_url")
                target_url = None
                state = None
                when = created_at
                
                # Get deployment status if available
                if statuses_url:
                    rs = await client.get(statuses_url, headers=_headers(token))
                    if rs.status_code == 200:
                        sts = rs.json()
                        if sts:
                            s = sts[0]  # Latest status
                            state = s.get("state")
                            target_url = s.get("target_url") or s.get("log_url")
                            when = s.get("updated_at") or s.get("created_at") or created_at
                
                return {
                    "when": when,
                    "when_pair": utc_local_pair(when, ctx.deps.tz) if when else None,
                    "html_url": target_url or f"https://github.com/{owner}/{repo}/deployments",
                    "source": "deployments",
                    "environment": dep.get("environment"),
                    "state": state,
                }
        
        # Fallback to GitHub Actions workflow runs
        r2 = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs",
            params={"status": "success", "per_page": 50},
            headers=_headers(token),
        )
        
        if r2.status_code >= 400:
            return {}
        
        data = r2.json()
        runs = data.get("workflow_runs", [])

        # Helper function to check if workflow name indicates deployment
        def looks_like_deploy(name: str) -> bool:
            n = (name or "").lower()
            return any(k in n for k in _DEPLOY_KEYWORDS)

        # Find the most recent deployment-like workflow run
        for run in runs:
            name = run.get("name") or run.get("display_title") or ""
            if looks_like_deploy(name):
                when = run.get("updated_at") or run.get("run_started_at") or run.get("created_at")
                return {
                    "when": when,
                    "when_pair": utc_local_pair(when, ctx.deps.tz) if when else None,
                    "html_url": run.get("html_url"),
                    "source": "actions",
                    "workflow_name": name,
                    "conclusion": run.get("conclusion"),
                }
        
        return {}  # No deployment found
```

**Purpose**: Provides functionality to find the most recent deployment, using GitHub's Deployments API when available, with fallback to GitHub Actions workflow runs.

### `agent/tools_files.py` - File Tools

```python
"""File/code tools: last change to a file, and first commit introducing a pattern."""

from typing import Any, Dict, List, Optional  # Type hints

import httpx  # HTTP client

from .core import agent  # Agent instance
from .config import GITHUB_API  # API base URL
from .utils import _headers, _json_or_error, utc_local_pair  # Utilities
from .models import Deps  # Dependencies
from pydantic_ai import RunContext  # Execution context

# Tool to find the last change to a specific file
@agent.tool
async def last_file_change(
    ctx: RunContext[Deps], *, owner: str, repo: str, path: str, branch: Optional[str] = None
) -> Dict[str, Any]:
    """Return the most recent commit that modified the given file path."""
    token = ctx.deps.github_token
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            if not branch:
                # Get default branch
                r0 = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers(token))
                repo_json = await _json_or_error(r0)
                branch = repo_json.get("default_branch", "main")
            
            # Get commits that modified the file (GitHub API returns most recent first)
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
            
            c = data[0]  # Most recent commit that touched the file
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

# Tool to find when a code pattern was first introduced
@agent.tool
async def introduced_line(
    ctx: RunContext[Deps], *, owner: str, repo: str, path: str, pattern: str, branch: Optional[str] = None, max_commits: int = 200
) -> Dict[str, Any]:
    """Find the earliest commit (within a recent window) that introduced a line containing `pattern` in `path`."""
    token = ctx.deps.github_token
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            if not branch:
                # Get default branch
                r0 = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=_headers(token))
                repo_json = await _json_or_error(r0)
                branch = repo_json.get("default_branch", "main")
            
            per_page = 100  # GitHub API limit
            pages = max(1, (max_commits + per_page - 1) // per_page)
            commits: List[Dict[str, Any]] = []
            
            # Fetch recent commits that touched the file
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
            
            # Search backwards through commits (oldest to newest)
            for c in reversed(commits[:max_commits]):
                sha = c.get("sha")
                if not sha:
                    continue
                
                # Get detailed commit info including diff
                rc = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}",
                    headers=_headers(token),
                )
                
                if rc.status_code >= 400:
                    continue
                
                detail = rc.json()
                files = detail.get("files") or []
                introduced = False
                
                # Check diff for additions containing the pattern
                for f in files:
                    patch = f.get("patch")
                    if not patch:
                        continue
                    
                    for line in patch.splitlines():
                        if line.startswith("+++") or line.startswith("---"):
                            continue
                        # Look for added lines (+) containing the pattern
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
```

**Purpose**: Provides tools for analyzing file changes and code history, including finding when files were last modified and when specific code patterns were introduced.

### `agent/tools_prs.py` - Pull Request Tools

```python
"""PR-related tool: search for a merged PR matching a query and return merge details."""

from typing import Any, Dict  # Type hints

import httpx  # HTTP client

from .core import agent  # Agent instance
from .config import GITHUB_API  # API base URL
from .utils import _headers, utc_local_pair  # Utilities
from .models import Deps  # Dependencies
from pydantic_ai import RunContext  # Execution context

# Tool to find merged PRs by query
@agent.tool
async def find_pr_merge(
    ctx: RunContext[Deps], *, owner: str, repo: str, query: str
) -> Dict[str, Any]:
    """Find a merged PR matching a query and return merge details."""
    async with httpx.AsyncClient(timeout=25) as client:
        # Search for merged PRs matching the query
        q = f"repo:{owner}/{repo} is:pr is:merged {query}"
        r = await client.get(
            f"{GITHUB_API}/search/issues",
            params={"q": q, "sort": "updated", "order": "desc", "per_page": 1},
            headers=_headers(ctx.deps.github_token),
        )
        
        if r.status_code >= 400:
            return {}
        
       