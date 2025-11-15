"""Core data models for the agent layer.

This module defines:

- ``RepoAnswer``: the structured response the agent returns for every question.
  FastAPI uses this as the response_model for ``/chat``, and the CLI prints
  its fields (text, timestamps, citations) to the terminal.

- ``Deps``: per-request dependencies passed into tools via ``RunContext[Deps]``.
  ``app.py`` creates a ``Deps`` instance using values from environment
  variables (``GITHUB_TOKEN``, ``AGENT_TZ``) and hands it to
  ``agent.run(..., deps=Deps(...))``. Inside each tool, you read them as
  ``ctx.deps.github_token`` and ``ctx.deps.tz``.

Dependencies:
- Reads ``DEFAULT_TZ`` from ``agent.config``.
- Is imported by:
  - ``agent.core`` (to configure the agent)
  - all ``agent.tools_*`` modules (type for their ``RunContext[Deps]``)
  - ``app.py`` (to construct ``Deps`` for each request)

In short: this file defines the "data contracts" between the server (FastAPI),
the agent, and the tools.
"""

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from .config import DEFAULT_TZ


class RepoAnswer(BaseModel):
    text: str = Field(description="Final answer, concise and factual.")
    citations: List[str] = Field(description="URLs used to produce the answer (commit, workflow run, etc.)")
    timestamps: List[str] = Field(description="Key timestamps included in the answer (UTC and local).")


@dataclass
class Deps:
    # GitHub Personal Access Token (string) or None if not set.
    github_token: Optional[str]
    # Timezone name (e.g. "Europe/Berlin") used for local timestamps.
    tz: str = DEFAULT_TZ
