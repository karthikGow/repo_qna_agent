"""Configuration and environment defaults for the agent.

Loads environment variables (.env via python-dotenv) and exposes constants used
across tools and the agent core.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_API = "https://api.github.com"
DEFAULT_TZ = os.getenv("AGENT_TZ", "Europe/Berlin")
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL", "openai:gpt-4o-mini")
