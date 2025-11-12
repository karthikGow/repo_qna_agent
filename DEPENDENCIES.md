# Project Dependencies and Pitfalls

This document outlines the key dependencies within the "Repo Q&A Agent" codebase and highlights potential pitfalls for developers new to the project.

## 1. Dependency Analysis

The "Repo Q&A Agent" project is structured to provide a natural language interface for querying GitHub repository information. It leverages a FastAPI application as its API, a CLI client for interaction, and a core AI agent with various tools for GitHub API interactions. An optional Retrieval Augmented Generation (RAG) system enhances its capabilities.

**Key Dependencies:**

*   **`app.py`**: The main FastAPI application.
    *   Depends on `agent.core` for the AI agent's execution.
    *   Depends on `agent.models` for defining request and response data structures (`RepoAnswer`, `Deps`).
    *   Depends on `agent.config` for accessing global configuration, such as `DEFAULT_TZ`.
    *   Utilizes `fastapi` for API endpoint definition and `pydantic` for data validation.

*   **`cli.py`**: The command-line interface client.
    *   Functionally depends on `app.py` by making HTTP requests to its FastAPI endpoints.
    *   Uses `typer` for building the command-line interface.
    *   Uses `httpx` for making asynchronous HTTP requests to the FastAPI server.

*   **`rag_index.py`**: Script for building the RAG semantic index.
    *   Depends on `httpx` for fetching commit data from the GitHub API.
    *   Uses `langchain_text_splitters` for processing text data from commits.
    *   Relies on `langchain_community.vectorstores.Chroma` for persisting the vector store.
    *   Conditionally depends on `langchain_community.embeddings.HuggingFaceEmbeddings` or `langchain_openai.OpenAIEmbeddings` for generating embeddings.

*   **`rag.py`**: Provides RAG retrieval functionality.
    *   Depends on `httpx` for verifying retrieved commits against the GitHub API.
    *   Uses `langchain_community.vectorstores.Chroma` to load and query the pre-built RAG index.

*   **`agent/__init__.py`**: The `agent` package initializer.
    *   Imports and exposes `agent` from `agent.core`.
    *   Imports `RepoAnswer` and `Deps` from `agent.models`.
    *   Imports `DEFAULT_TZ` from `agent.config`.
    *   Crucially, it imports all `agent/tools_*.py` modules (`agent.tools_commits`, `agent.tools_deployments`, `agent.tools_files`, `agent.tools_prs`, `agent.tools_rag`) to register their functions as tools with the `PydanticAI` agent upon package initialization.

*   **`agent/config.py`**: Manages project configuration.
    *   Depends on `os` for environment variable access.
    *   Uses `python-dotenv` for loading environment variables from a `.env` file.

*   **`agent/core.py`**: Defines the central AI agent.
    *   Instantiates the `PydanticAI Agent`.
    *   Depends on `agent.config` for `MODEL_NAME` and other configuration.
    *   Depends on `agent.models` for `RepoAnswer` and `Deps` to define agent output and context.

*   **`agent/models.py`**: Defines data models.
    *   Uses `dataclasses` and `typing` for type hinting and data structure definition.
    *   Uses `pydantic` for defining `RepoAnswer` (agent's output schema).
    *   Depends on `agent.config` for `DEFAULT_TZ` in some model definitions.

*   **`agent/tools_commits.py`, `agent/tools_deployments.py`, `agent/tools_files.py`, `agent/tools_prs.py`**: These modules define specific tools for GitHub API interaction.
    *   All depend on `httpx` for making HTTP requests to the GitHub API.
    *   All depend on `agent.core` to register their functions as agent tools.
    *   All depend on `agent.config` for GitHub API base URL (`GITHUB_API`).
    *   All depend on `agent.utils` for helper functions like `_headers`, `_json_or_error`, and `utc_local_pair`.
    *   All depend on `agent.models` for `Deps` to access contextual information.
    *   All depend on `pydantic_ai.RunContext` for agent execution context.
    *   `agent/tools_commits.py` additionally uses the `re` module for regular expressions.

*   **`agent/tools_rag.py`**: Integrates the RAG system as an agent tool.
    *   Conditionally imports `rag.rag_find_commits` from `rag.py`.
    *   Depends on `agent.core` to register itself as an agent tool.
    *   Depends on `agent.models` for `Deps`.
    *   Depends on `pydantic_ai.RunContext`.

*   **`agent/utils.py`**: Provides common utility functions.
    *   Depends on `datetime`, `typing`, `httpx`, `fastapi.HTTPException`, `zoneinfo`.
    *   Depends on `agent.config` for `DEFAULT_TZ`.

## 2. Codebase Dependency Diagram

```mermaid
graph TD
    subgraph Core Application
        A[app.py]
        B[cli.py]
    end

    subgraph Agent Module
        E[agent/__init__.py]
        F[agent/config.py]
        G[agent/core.py]
        H[agent/models.py]
        N[agent/utils.py]
        subgraph Agent Tools
            I[agent/tools_commits.py]
            J[agent/tools_deployments.py]
            K[agent/tools_files.py]
            L[agent/tools_prs.py]
            M[agent/tools_rag.py]
        end
    end

    subgraph RAG System
        C[rag_index.py]
        D[rag.py]
    end

    A --> G
    A --> H
    A --> F
    B --> A (API Call)

    E --> G
    E --> H
    E --> F
    E --> I
    E --> J
    E --> K
    E --> L
    E --> M

    G --> F
    G --> H

    H --> F

    I --> G
    I --> F
    I --> N
    I --> H
    J --> G
    J --> F
    J --> N
    J --> H
    K --> G
    K --> F
    K --> N
    K --> H
    L --> G
    L --> F
    L --> N
    L --> H

    M --> D
    M --> G
    M --> H

    N --> F

    C --> D (Creates Index)
    D --> C (Loads Index)
```

## 3. Potential Pitfalls for Novices

Working with this project, especially for those new to AI agents, FastAPI, or GitHub API interactions, can present several challenges:

1.  **GitHub API Rate Limits**: Frequent or unauthenticated requests to the GitHub API can quickly exhaust rate limits, leading to `403 Forbidden` errors. Novices should be aware of these limits and how to use personal access tokens for authenticated requests to increase limits.
2.  **Environment Variable Configuration**: The project relies heavily on environment variables (e.g., `GITHUB_TOKEN`, `OPENAI_API_KEY`, `MODEL_NAME`). Incorrectly set or missing environment variables, particularly in the `.env` file, are a common source of runtime errors.
3.  **RAG System Setup and Maintenance**:
    *   **Indexing Time**: Building the RAG index using `rag_index.py` can be time-consuming for large repositories.
    *   **Embedding Models**: Understanding the choice between OpenAI and local HuggingFace embedding models, and ensuring the correct setup for each, is crucial.
    *   **ChromaDB Persistence**: The `rag_store/` directory (where Chroma persists data) needs to be managed correctly. Issues with permissions or accidental deletion can break the RAG functionality.
4.  **Asynchronous Programming**: FastAPI is built on asynchronous Python (`async`/`await`). While it simplifies many aspects, understanding the flow of control in asynchronous code, especially when dealing with external I/O operations like HTTP requests, can be challenging for those unfamiliar with it.
5.  **PydanticAI Agent Logic**: The core of the agent is built with `PydanticAI`. Novices might find it complex to grasp how tools are registered, how the agent selects and executes tools based on prompts, and how to debug the agent's decision-making process.
6.  **Tool Integration and Signatures**: Each `agent/tools_*.py` module defines functions that are registered as tools. The `PydanticAI` agent expects specific function signatures and return types. Mismatches can lead to the agent failing to use a tool correctly or producing unexpected outputs.
7.  **Timezone Handling**: The project explicitly handles timezones for timestamps. Misunderstandings of UTC vs. local time, or incorrect timezone conversions, can lead to subtle but critical bugs when analyzing commit or deployment times.
8.  **Error Handling and Debugging**: The project includes custom error handling mechanisms (e.g., `_json_or_error` in `agent/utils.py`). Novices might need to spend time understanding these custom error flows to effectively debug issues.

## 4. Viewing and Interpreting `DEPENDENCIES.md`

To properly view and interpret the Mermaid diagram within this Markdown file, you will need a Markdown viewer that supports Mermaid rendering.

*   **VS Code**: If you are using Visual Studio Code, ensure you have a Markdown preview extension that supports Mermaid (e.g., the built-in Markdown Preview Enhanced or other extensions like "Markdown Preview Mermaid Support"). Simply open `DEPENDENCIES.md` in VS Code and then open the Markdown preview (usually by clicking the preview icon in the top right of the editor or using `Ctrl+Shift+V`).
*   **GitHub/GitLab**: When this file is pushed to a GitHub or GitLab repository, their web interfaces will automatically render the Mermaid diagram.
*   **Online Mermaid Editors**: You can copy the Mermaid code block (starting with `graph TD` and ending with ````) into an online Mermaid editor (e.g., [Mermaid Live Editor](https://mermaid.live/)) to view and experiment with the diagram.