"""A cost-guarded, hallucination-checked LLM pipeline for structured record enrichment."""

from .config import Limits, Settings
from .context import RunContext
from .errors import CreditOutError, PermanentError, TransientError
from .spend import SpendTracker

__version__ = "1.0.0"
__all__ = [
    "Settings", "Limits", "RunContext", "SpendTracker",
    "CreditOutError", "PermanentError", "TransientError",
]
