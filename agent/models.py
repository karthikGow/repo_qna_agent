"""Data models used by the agent: output schema and dependency container."""

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
    github_token: Optional[str]
    tz: str = DEFAULT_TZ
