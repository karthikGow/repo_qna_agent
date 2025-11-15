"""
Server-side entrypoint (FastAPI).

- Exposes `/health` and `/chat` HTTP endpoints.
- Reads environment (e.g. `GITHUB_TOKEN`, `AGENT_TZ`) and passes those values
  into the PydanticAI agent as `Deps`.
- Delegates all GitHub logic to the `agent/` package; this file stays thin and
  only handles HTTP and request/response wiring.
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import agent, RepoAnswer, DEFAULT_TZ, Deps


app = FastAPI(title="Repo Q&A Agent", version="1.3.0")


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


class ChatRequest(BaseModel):
    repo: str = Field(..., description="owner/repo")
    question: str
    branch: Optional[str] = None
    environment: Optional[str] = None


@app.post("/chat", response_model=RepoAnswer)
async def chat(req: ChatRequest) -> RepoAnswer:
    token = os.getenv("GITHUB_TOKEN")
    tz = os.getenv("AGENT_TZ", DEFAULT_TZ)
    if "/" not in req.repo:
        raise HTTPException(400, "repo must be in 'owner/repo' format")
    owner, repo = req.repo.split("/", 1)

    prompt = (
        f"Repo: {owner}/{repo}\n"
        f"Branch: {req.branch or '(default)'}\n"
        f"Environment: {req.environment or '(any)'}\n"
        f"Question: {req.question}"
    )

    try:
        result = await agent.run(prompt, deps=Deps(github_token=token, tz=tz))
        return result.output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Agent error: {e}")
