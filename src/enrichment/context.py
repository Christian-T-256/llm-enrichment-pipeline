"""The small bundle of run-scoped collaborators passed to each stage."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import Backend
from .config import Settings
from .spend import SpendTracker


@dataclass
class RunContext:
    backend: Backend
    tracker: SpendTracker
    settings: Settings
