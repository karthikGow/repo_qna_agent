"""Agent package public API.

Exposes the configured `agent`, output and dependency models, and default timezone.
Imports tool modules so their `@agent.tool` registrations execute at import time.
"""

from .core import agent
from .models import RepoAnswer, Deps
from .config import DEFAULT_TZ

# Import tool modules so their @agent.tool registrations run on import
from . import tools_commits  # noqa: F401
from . import tools_deployments  # noqa: F401
from . import tools_prs  # noqa: F401
from . import tools_files  # noqa: F401
from . import tools_rag  # noqa: F401
