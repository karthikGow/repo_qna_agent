# Repo Q&A Agent Documentation

## 1. Project Overview

The "Repo Q&A Agent" is a sophisticated tool designed to answer questions about GitHub repositories using a combination of FastAPI, PydanticAI, and an optional LangChain RAG (Retrieval Augmented Generation) system. Its primary purpose is to provide quick and accurate answers to common repository-related queries, such as:

- "What was implemented in the last commit and by whom?"
- "When was the last deployment?"
- "When did we fix the favicon bug?"
- "When did we refactor the Homepage layout?"

Key features include:
- A lightweight FastAPI API for handling requests.
- A PydanticAI-powered agent that leverages various tools to interact with the GitHub API.
- Support for both UTC and local (Europe/Berlin) timestamps in responses.
- Automatic citation of sources (commit URLs, workflow run URLs, PR URLs).
- A command-line interface (CLI) for easy interaction.
- An optional RAG system for semantic search over commit history, enhancing the agent's ability to answer complex queries.

## 2. Codebase Structure

The project is organized into several key components:

- `app.py`: The main FastAPI application entrypoint, defining API routes for health checks and chat interactions.
- `agent/`: A Python package containing the core logic and specialized tools for the agent.
    - `agent/config.py`: Manages environment variables and defines global constants.
    - `agent/core.py`: Initializes and configures the PydanticAI agent with its instructions and capabilities.
    - `agent/models.py`: Defines Pydantic data models for structured agent output and dependency injection.
    - `agent/utils.py`: Provides utility functions for HTTP request headers, timestamp conversions, and error handling.
    - `agent/tools_commits.py`: Contains tools for querying commit-related information (e.g., `last_commit`, `find_commit`).
    - `agent/tools_deployments.py`: Implements tools for retrieving deployment information (e.g., `last_deployment`).
    - `agent/tools_files.py`: Offers tools for file and code-specific queries (e.g., `last_file_change`, `introduced_line`).
    - `agent/tools_prs.py`: Includes tools for searching and retrieving Pull Request details (e.g., `find_pr_merge`).
    - `agent/tools_rag.py`: An optional module that integrates RAG capabilities for semantic commit search.
    - `agent/__init__.py`: Initializes the `agent` package, making core components and tools accessible.
- `cli.py`: A command-line interface client for interacting with the FastAPI agent.
- `rag.py`: A helper module for RAG retrieval, responsible for loading Chroma collections and verifying commits.
- `rag_index.py`: A script used to build and persist a local semantic index of a GitHub repository's commit history for RAG.
- `README.md`: The main project README, providing a high-level overview and quickstart instructions.
- `RUN.md`: A detailed runbook with step-by-step instructions for setting up and running the project.
- `demo.ps1`: A PowerShell script demonstrating example queries to the agent.
- `requirements.txt`: Lists all Python dependencies required for the project.
- `.env.example`: A template file for environment variables.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## 3. File-by-File Analysis

### `app.py`
This file serves as the FastAPI application's entry point. It defines the `/health` endpoint for basic service status checks and the `/chat` endpoint, which is the primary interface for user questions. It parses incoming requests, extracts repository details, questions, and optional branch/environment information, and then dispatches these to the PydanticAI agent for processing. Error handling for GitHub API rate limits and other issues is also managed here.

### `cli.py`
A lightweight command-line interface built with `typer`. It allows users to interact with the Repo Q&A Agent from their terminal. Users can provide a repository and a question, or enter an interactive mode for continuous questioning. The CLI communicates with the FastAPI server, formats the agent's responses, and displays the answer along with any timestamps and citations.

### `rag.py`
This module provides the core retrieval logic for the optional RAG system. It is responsible for loading a pre-built Chroma vector store collection of commit history. It also includes functionality to verify the existence and details of retrieved commits against the GitHub API, ensuring the accuracy of RAG results.

### `rag_index.py`
A standalone script designed to build a semantic index of a GitHub repository's commit history. It fetches commits from the GitHub API, converts them into LangChain `Document` objects with rich metadata, embeds these documents using either OpenAI or local HuggingFace models, and then stores them in a Chroma vector database for efficient retrieval by the RAG system.

### `agent/__init__.py`
This file acts as the public interface for the `agent` package. It imports and exposes the main `agent` instance, `RepoAnswer` and `Deps` models, and `DEFAULT_TZ` constant. Crucially, it imports all `agent/tools_*.py` modules, which automatically registers their `@agent.tool` decorated functions with the PydanticAI agent upon import.

### `agent/config.py`
Handles the project's configuration by loading environment variables (e.g., from a `.env` file) and defining constants. Key constants include `GITHUB_API` (the base URL for GitHub API), `DEFAULT_TZ` (the default timezone for timestamp conversions), and `MODEL_NAME` (the PydanticAI model to be used).

### `agent/core.py`
This module is where the PydanticAI `agent` is instantiated and configured. It sets up the agent with the chosen language model, defines the types for dependencies (`Deps`) and output (`RepoAnswer`), and provides a comprehensive set of instructions that guide the agent's behavior, emphasizing the use of tools, accurate citations, and precise timestamp handling.

### `agent/models.py`
Defines the data structures used throughout the agent. `RepoAnswer` is a Pydantic `BaseModel` that specifies the expected format of the agent's final response, including the answer text, a list of citations (URLs), and a list of timestamps. `Deps` is a `dataclass` used for dependency injection, primarily holding the GitHub token and timezone information.

### `agent/utils.py`
A collection of helper functions used across the `agent` package. This includes `_headers` for constructing HTTP headers with authentication, `iso` for formatting datetimes into ISO 8601 strings, `utc_local_pair` for converting UTC timestamps to local time and providing both formats, and `_json_or_error` for robustly handling HTTP responses and potential API errors.

### `agent/tools_commits.py`
This module provides two essential tools for querying GitHub commit data:
- `last_commit`: Retrieves details about the most recent commit on a specified branch (or the default branch if none is provided).
- `find_commit`: Searches for the most recent commit whose message matches a given keyword query, utilizing GitHub's search API or by scanning recent commits.

### `agent/tools_deployments.py`
Implements the `last_deployment` tool, which is designed to find the latest deployment event for a repository. It first attempts to use the GitHub Deployments API and falls back to scanning GitHub Actions workflow runs if no deployment API data is available. It supports filtering by environment.

### `agent/tools_files.py`
Contains tools for detailed file and code analysis within a repository:
- `last_file_change`: Identifies the most recent commit that modified a specific file path.
- `introduced_line`: Scans recent commit history to find the earliest commit that introduced a line containing a specified pattern within a given file.

### `agent/tools_prs.py`
Provides the `find_pr_merge` tool, which searches for merged Pull Requests that match a given query. It retrieves details about the PR, including its title, author, merge date, and the associated merge commit SHA and URL.

### `agent/tools_rag.py`
This module acts as an optional wrapper for the RAG functionality. If the RAG index is present, it exposes the `rag_find_change` tool, which leverages the semantic commit retrieval capabilities of the RAG system to answer queries more effectively by finding relevant commit information.

## 4. Dependencies

The project relies on the following Python packages, as specified in `requirements.txt`:

- `fastapi>=0.115`
- `uvicorn>=0.30`
- `httpx>=0.27`
- `pydantic>=2.8`
- `pydantic-ai>=1.12`
- `typer>=0.12`
- `python-dotenv>=1.0`
- `openai>=1.47`
- `tzdata>=2024.1`
- `langchain>=0.3` (for RAG)
- `langchain-community>=0.3` (for RAG)
- `langchain-openai>=0.2` (for RAG embeddings)
- `chromadb>=0.5` (for RAG vector store)
- `faiss-cpu>=1.8.0.post1` (for RAG vector store)
- `sentence-transformers>=3.0` (for local RAG embeddings)
- `rank-bm25>=0.2.2` (for RAG)
- `tiktoken>=0.7` (for RAG)

## 5. How to Run/Interact

This section outlines the basic steps to set up and run the Repo Q&A Agent. More detailed instructions can be found in `RUN.md`.

### Setup

1.  **Create a Python Virtual Environment**:
    ```powershell
    python -m venv .venv
    ```

2.  **Activate the Virtual Environment**:
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass # (Windows only, if needed)
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    pip install tzdata
    ```

### Configuration

Set the necessary environment variables. You can use a `.env` file (which is automatically loaded by `python-dotenv`) or set them directly in your session.

-   **LLM API Key and Model**: Choose one of the following:
    -   **OpenAI**:
        ```powershell
        $Env:OPENAI_API_KEY = "<openai-key>"
        $Env:PYDANTIC_AI_MODEL = "openai:gpt-4o-mini"
        ```
    -   **DeepSeek (OpenAI-compatible)**:
        ```powershell
        $Env:OPENAI_API_KEY = "<deepseek-key>"
        $Env:OPENAI_BASE_URL = "https://api.deepseek.com"
        $Env:PYDANTIC_AI_MODEL = "openai:<deepseek-model-id>"
        ```
    -   **Local Ollama**:
        ```powershell
        # winget install --id Ollama.Ollama -e
        ollama pull llama3.1
        $Env:PYDANTIC_AI_MODEL = "ollama:llama3.1"
        ```
    -   **OpenRouter (OpenAI-compatible)**:
        ```powershell
        $Env:OPENAI_API_KEY = "<openrouter-key>"
        $Env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
        $Env:PYDANTIC_AI_MODEL = "openai:google/gemini-2.5-flash-lite-preview-09-2025"
        ```
    *Note: Do not commit real API keys. Use `.env` or local environment variables.*

-   **GitHub Token (read access)**:
    ```powershell
    $Env:GITHUB_TOKEN = "<github-pat>"
    ```

### Running the API Server

Start the FastAPI application:
```powershell
uvicorn app:app --reload --port 8000
```

### Interacting via CLI

In a separate terminal (after activating the virtual environment and setting environment variables):

-   **Ask a single question**:
    ```powershell
    python cli.py --repo owner/repo "What was implemented in the last commit and by whom?"
    python cli.py --repo owner/repo --environment prod "When was the last deployment?"
    python cli.py --repo owner/repo "When did we fix the favicon bug?"
    python cli.py --repo owner/repo "When did we refactor the Homepage layout?"
    ```

-   **File/code-aware examples**:
    ```powershell
    python cli.py --repo owner/repo "When did we last change README.md?"
    python cli.py --repo owner/repo "When did we add @app.get('/health') in app.py?"
    ```

-   **Interactive mode (if no question is provided)**:
    ```powershell
    python cli.py --repo owner/repo
    ```
    Then type your questions at the `>` prompt. Type `exit` or `quit` to end the session.

### Health Check

Verify the API is running:
```powershell
curl http://127.0.0.1:8000/health
# Expected output: { "ok": true }
```

### Optional: Build the RAG Index

If you wish to use the RAG functionality, build the semantic index:

-   **With OpenAI embeddings**:
    ```powershell
    $Env:OPENAI_API_KEY = "<openai-key>"
    python rag_index.py --repo owner/repo --max-commits 1500
    ```

-   **With local embeddings**:
    ```powershell
    $Env:EMBED_MODE = "local"
    python rag_index.py --repo owner/repo --max-commits 1500