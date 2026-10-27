"""LLM backends behind one small interface.

The pipeline talks to a `Backend`, never to a vendor SDK. Two implementations:

* `AnthropicBackend`: wraps the real `anthropic` SDK and, crucially, translates
  its exception zoo into our three-class taxonomy (`errors.py`). The credit-out
  sniff inspects the message body because the Batch endpoint sometimes returns a
  credit-balance error wrapped in a Cloudflare 5xx.
* `MockBackend`: fully offline, deterministic. Lets anyone clone the repo and
  run the entire pipeline (including a simulated Batch round-trip) end-to-end
  with no API key and zero spend, while still exercising every branch.

Keeping the surface this thin is what makes the retry/guard logic in `spend.py`
identical across both, and makes the whole pipeline trivially testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import CreditOutError, PermanentError, TransientError


# ── Lightweight response shapes (mirror the bits of the SDK we actually read) ──

@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class Message:
    content: list = field(default_factory=list)


@dataclass
class BatchStatus:
    ended: bool
    succeeded: int
    total: int


class Backend(Protocol):
    def create_message(self, *, model: str, max_tokens: int, system: str,
                       messages: list, tools: list | None = None,
                       tool_choice: dict | None = None) -> Message: ...

    def create_batch(self, requests: list) -> str: ...
    def poll_batch(self, batch_id: str) -> BatchStatus: ...
    def batch_results(self, batch_id: str) -> dict: ...
    def cancel_batch(self, batch_id: str) -> None: ...


_CREDIT_OUT_MARKERS = (
    "credit balance is too low",
    "credit_balance_too_low",
    "billing_hard_limit",
)


def _looks_like_credit_out(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _CREDIT_OUT_MARKERS)


# ── Real backend ──────────────────────────────────────────────────────────────

class AnthropicBackend:
    """Adapter over the official `anthropic` SDK."""

    def __init__(self, api_key: str):
        import anthropic
        self._ant = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _translate(self, exc: Exception) -> Exception:
        """Map an SDK exception onto our taxonomy. Credit-out wins over
        everything, including when it arrives disguised as a 5xx."""
        ant = self._ant
        if _looks_like_credit_out(exc):
            return CreditOutError(str(exc))
        if isinstance(exc, (ant.BadRequestError, ant.AuthenticationError,
                            ant.PermissionDeniedError, ant.NotFoundError)):
            return PermanentError(f"{type(exc).__name__}: {exc}")
        if isinstance(exc, (ant.InternalServerError, ant.APIStatusError,
                            ant.APIConnectionError, ant.RateLimitError)):
            return TransientError(f"{type(exc).__name__}: {exc}")
        return TransientError(f"{type(exc).__name__}: {exc}")

    def create_message(self, *, model, max_tokens, system, messages,
                       tools=None, tool_choice=None) -> Message:
        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens, system=system, messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        try:
            return self._client.messages.create(**kwargs)  # SDK msg is duck-compatible
        except Exception as exc:  # noqa: BLE001 (re-raised as our taxonomy)
            raise self._translate(exc) from exc

    def create_batch(self, requests) -> str:
        try:
            return self._client.messages.batches.create(requests=requests).id
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc

    def poll_batch(self, batch_id) -> BatchStatus:
        try:
            s = self._client.messages.batches.retrieve(batch_id)
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        c = s.request_counts
        done = c.succeeded + c.errored + c.canceled + c.expired
        return BatchStatus(ended=s.processing_status == "ended",
                           succeeded=c.succeeded, total=done + c.processing)

    def batch_results(self, batch_id) -> dict:
        out: dict[str, str] = {}
        try:
            for r in self._client.messages.batches.results(batch_id):
                if r.result.type == "succeeded":
                    out[r.custom_id] = r.result.message.content[0].text.strip()
                else:
                    out[r.custom_id] = ""
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc
        return out

    def cancel_batch(self, batch_id) -> None:
        try:
            self._client.messages.batches.cancel(batch_id)
        except Exception:  # noqa: BLE001 (best effort)
            pass


# ── Mock backend (offline, deterministic) ──────────────────────────────────────

_HELP_PAT = re.compile(
    r"(?:help|serve|work with|support|for)\s+([a-z][a-z &/\-]{4,40}?)(?:\s+(?:who|that|to|in|with|grow|scale|build|navigate)\b|[.,]|$)",
    re.I,
)
_FLUFF = ("cutting-edge", "world-class", "best-in-class", "synergy", "leading provider")


class MockBackend:
    """Deterministic, no-network stand-in. Emulates each tool's structured
    output from the record's own text so a dry run produces sensible,
    inspectable results and exercises both generation routes + grounding."""

    def __init__(self):
        self._batches: dict[str, list] = {}
        self._n = 0

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _user_text(messages: list) -> str:
        parts = []
        for m in messages:
            c = m.get("content", "")
            parts.append(c if isinstance(c, str) else str(c))
        return "\n".join(parts)

    @staticmethod
    def _guess_segment(text: str) -> str:
        m = _HELP_PAT.search(text or "")
        return m.group(1).strip().lower() if m else ""

    # -- interface -----------------------------------------------------------
    def create_message(self, *, model, max_tokens, system, messages,
                       tools=None, tool_choice=None) -> Message:
        text = self._user_text(messages)
        tool = (tool_choice or {}).get("name") if tool_choice else None

        if tool == "extract_segment_from_profile":
            seg = self._guess_segment(text)
            return Message([ToolUseBlock(name=tool, input={
                "segment": seg, "confidence": "high" if seg else "low"})])

        if tool == "extract_segment_from_website":
            seg = self._guess_segment(text)
            ev = next((ln.strip() for ln in text.splitlines()
                       if seg and seg in ln.lower()), "")
            return Message([ToolUseBlock(name=tool, input={
                "segment": seg, "confidence": "high" if seg else "low",
                "evidence": ev[:160]})])

        if tool == "reconcile_segments":
            segs = re.findall(r'"([^"]*)"', text)
            a = segs[0] if len(segs) > 0 else ""
            b = segs[1] if len(segs) > 1 else ""
            agree = bool(a) and bool(b) and (a in b or b in a)
            merged = max([a, b], key=len) if (a or b) else ""
            return Message([ToolUseBlock(name=tool, input={
                "agree": "yes" if agree else "no", "merged_segment": merged,
                "reasoning": "string overlap heuristic (mock)"})])

        if tool == "verify_segment_grounded":
            # Grounded if every word of the claimed segment appears in the source.
            claim = ""
            m = re.search(r"Claimed segment:\s*\"([^\"]*)\"", text)
            if m:
                claim = m.group(1).lower()
            src = text.lower()
            grounded = bool(claim) and all(w in src for w in claim.split() if len(w) > 3)
            return Message([ToolUseBlock(name=tool, input={
                "grounded": "yes" if grounded else "no",
                "reasoning": "token-presence heuristic (mock)"})])

        # No tool → free-text generation (the note stage).
        return Message([TextBlock(text=self._mock_note(text))])

    def _mock_note(self, prompt: str) -> str:
        company = self._field(prompt, "Company") or "the company"
        headline = self._field(prompt, "Headline")
        seg = self._field(prompt, "Served segment")
        obs = headline.split(".")[0][:90] if headline else f"the work at {company}"
        note = f"{company} stands out for {obs.lower()}."
        if seg:
            note += f" That lines up with what {seg} look for."
        # Mock never emits fluff; real outputs are post-validated for it.
        for f in _FLUFF:
            note = note.replace(f, "")
        return note

    @staticmethod
    def _field(prompt: str, label: str) -> str:
        m = re.search(rf"{re.escape(label)}:\s*(.+)", prompt)
        return m.group(1).strip() if m else ""

    def create_batch(self, requests) -> str:
        self._n += 1
        bid = f"mock_batch_{self._n}"
        self._batches[bid] = requests
        return bid

    def poll_batch(self, batch_id) -> BatchStatus:
        n = len(self._batches.get(batch_id, []))
        return BatchStatus(ended=True, succeeded=n, total=n)  # mock returns results immediately

    def batch_results(self, batch_id) -> dict:
        out = {}
        for req in self._batches.get(batch_id, []):
            params = req["params"]
            msg = self.create_message(
                model=params["model"], max_tokens=params["max_tokens"],
                system=params.get("system", ""), messages=params["messages"])
            out[req["custom_id"]] = msg.content[0].text
        return out

    def cancel_batch(self, batch_id) -> None:
        self._batches.pop(batch_id, None)
