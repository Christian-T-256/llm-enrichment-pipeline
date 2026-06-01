"""Central configuration: models, pricing, and tunable limits.

Everything that controls cost or model behavior lives here so it can be audited
in one place. Prices are USD per 1M tokens and reflect public list pricing for
the Claude 4 family at time of writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Models ───────────────────────────────────────────────────────────────────
# Cheap, fast model for the structured-extraction stages (4 calls per record).
EXTRACT_MODEL = "claude-haiku-4-5"
# Stronger model for the single free-text generation stage, run through Batch.
GENERATE_MODEL = "claude-sonnet-4-6"

# ── Pricing (USD per 1M tokens) ──────────────────────────────────────────────
PRICING = {
    "claude-haiku-4-5":  {"in": 1.0, "out": 5.0},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
}
# Anthropic's Message Batches API is billed at 50% of standard rates.
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class Limits:
    """Input-size caps. These are cost levers, not correctness levers — the
    extraction stages only need a representative slice of source text, and
    feeding the full thing was the single largest driver of per-record cost.
    """

    # Max characters of website text fed to the segment extractor.
    extract_source_chars: int = 8_000
    # Max characters of source text fed to the (cheaper, narrower) grounding
    # verifier. The verifier only needs enough to confirm/deny a claim.
    grounding_source_chars: int = 2_000
    # Hard ceiling on generated note length before we treat the output as the
    # model "writing prose at us" instead of a one-liner (refusal heuristic).
    note_max_words: int = 45


@dataclass
class Settings:
    """Per-run settings, populated from CLI flags."""

    max_cost: float                       # hard USD cap — required, no default
    dry_run: bool = False                 # use the offline mock backend
    resume_checkpoint: str | None = None  # path to a pre-generation checkpoint
    review_gate: bool = True              # block first run for human sample review
    workers: int = 8                      # reserved for I/O-bound enrichment fan-out
    limits: Limits = field(default_factory=Limits)
