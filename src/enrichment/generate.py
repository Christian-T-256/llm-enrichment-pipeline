"""Free-text generation via the Batch API, with the guards that wrap it.

This is the one stage that uses the stronger (and pricier) model, so it runs
through Anthropic's Message Batches API for the 50% discount. Three things make
it safe and reliable rather than just cheap:

* **Pre-batch healthcheck.** Before submitting N expensive requests, fire a
  single trivial one. If the account is out of credit or the endpoint is down,
  find out for ~$0.0001 instead of after committing the whole batch. This guard
  exists because the un-checked version once submitted thousands of requests
  into a credit-out and lost both the money and the hours of upstream work.

* **Refusal recovery.** When a record's two sources look like different
  entities, the model sometimes refuses and writes *prose about why it can't*.
  That meta-text must never ship as the result. `is_refusal` detects it and we
  regenerate down the simpler observation-only path.

* **Output validation.** Generated notes are scanned for banned filler; a clean
  one-liner is the contract, and we enforce it rather than hope for it.

Routing: a record with a confident, non-vague segment takes the segment-anchored
prompt; everything else falls back to a plain observation.
"""

from __future__ import annotations

import time

from .config import GENERATE_MODEL
from .context import RunContext
from .errors import CreditOutError
from .spend import guarded_call

# ── Prompts ────────────────────────────────────────────────────────────────────

SEGMENT_ANCHORED_SYSTEM = (
    "You write a two-sentence analyst note about a company. Sentence one is a "
    "concrete, specific observation about their actual work — a method, an "
    "unusual combination, a niche, a number. Sentence two names the segment they "
    "serve and how their positioning fits it. You are a peer who read their "
    "material, not a marketer. No filler, no superlatives, no brand slogans.")

SEGMENT_ANCHORED_PROMPT = (
    "Company: {company}\nHeadline: {headline}\nProfile: {profile}\n"
    "Website: {website}\nServed segment: {segment}\n\n"
    "Write the two-sentence note.")

OBSERVATION_SYSTEM = (
    "You write one concrete, specific sentence about what a company actually "
    "does — something a careful reader would notice. You are a peer, not a "
    "marketer. No filler, no superlatives, no brand slogans.")

OBSERVATION_PROMPT = (
    "Company: {company}\nHeadline: {headline}\nProfile: {profile}\n"
    "Website: {website}\n\nWrite the single observation sentence.")

# Banned filler — a generated note containing any of these fails validation.
BANNED_FILLER = ("cutting-edge", "world-class", "best-in-class", "synergy",
                 "leading provider", "industry-leading", "game-changing",
                 "next-level", "revolutionary")


def _is_anchored(record: dict) -> bool:
    return bool(record.get("segment", "").strip()) and \
        record.get("confidence", "low") in ("high", "medium")


def build_requests(records: list[dict]) -> list[dict]:
    """One Batch request per record, prompt chosen by final confidence."""
    reqs = []
    for r in records:
        company = r.get("company", "").strip() or r.get("name", "").strip()
        common = dict(
            company=company,
            headline=r.get("headline", "").strip() or "(none)",
            profile=r.get("profile_text", "").strip()[:500] or "(none)",
            website=r.get("_website_text", "").strip()[:600] or "(none)",
        )
        if _is_anchored(r):
            system = SEGMENT_ANCHORED_SYSTEM
            prompt = SEGMENT_ANCHORED_PROMPT.format(segment=r["segment"], **common)
        else:
            system = OBSERVATION_SYSTEM
            prompt = OBSERVATION_PROMPT.format(**common)
        reqs.append({
            "custom_id": r["_row_id"],
            "params": {"model": GENERATE_MODEL, "max_tokens": 120,
                       "system": system, "messages": [{"role": "user", "content": prompt}]},
        })
    return reqs


def healthcheck(ctx: RunContext) -> None:
    """Submit + cancel a 1-request mini-batch to confirm credits + endpoint."""
    print("  [healthcheck] probing batch endpoint + credit balance...", flush=True)
    bid = guarded_call(
        lambda: ctx.backend.create_batch([{
            "custom_id": "_healthcheck",
            "params": {"model": GENERATE_MODEL, "max_tokens": 5,
                       "system": "", "messages": [{"role": "user", "content": "ok"}]},
        }]),
        tracker=ctx.tracker, model=GENERATE_MODEL,
        in_tokens=10, out_tokens=5, batch=True, max_retries=2)
    ctx.backend.cancel_batch(bid)
    print("  [healthcheck] ok — endpoint reachable, credits available", flush=True)


def submit_and_poll(requests: list[dict], ctx: RunContext,
                    poll_interval: int = 30) -> dict[str, str]:
    """Healthcheck, submit the real batch, poll to completion, return results."""
    healthcheck(ctx)
    est_in = sum(len(str(r["params"]["messages"])) // 4 + 100 for r in requests)
    est_out = sum(r["params"]["max_tokens"] for r in requests)
    print(f"  submitting {len(requests)} requests to batch endpoint...", flush=True)
    batch_id = guarded_call(
        lambda: ctx.backend.create_batch(requests),
        tracker=ctx.tracker, model=GENERATE_MODEL,
        in_tokens=est_in, out_tokens=est_out, batch=True, max_retries=5)
    print(f"  batch id: {batch_id}", flush=True)
    while True:
        try:
            status = ctx.backend.poll_batch(batch_id)
        except CreditOutError:
            raise
        except Exception as e:  # noqa: BLE001 — transient poll error, keep waiting
            print(f"  transient poll error ({e}); retrying...", flush=True)
            time.sleep(poll_interval)
            continue
        print(f"  progress: {status.succeeded}/{status.total}", flush=True)
        if status.ended:
            break
        time.sleep(poll_interval)
    return ctx.backend.batch_results(batch_id)


# ── Post-processing + recovery ─────────────────────────────────────────────────

def clean(text: str) -> str:
    """Strip whole-line-wrapping quotes only. Never lowercase, never strip
    terminal periods — the two-sentence format depends on them, and an earlier
    `rstrip('.')` silently stripped end punctuation off most outputs."""
    s = (text or "").strip()
    if len(s) >= 4 and s.startswith('"') and s.endswith('"') and '"' not in s[1:-1]:
        s = s[1:-1].strip()
    return s


_REFUSAL_MARKERS = (
    "i cannot", "i can't", "unable to", "cannot write", "please provide",
    "the data", "appears unrelated", "mismatched", "defeats the purpose",
    "would be fabricated", "not enough information",
)


def is_refusal(text: str, max_words: int) -> bool:
    """True if the model wrote meta-prose instead of a note.

    Two signals: refusal phrasing, or simply being far longer than a one/two
    sentence note has any business being.
    """
    if not text:
        return True
    if len(text.split()) > max_words:
        return True
    low = text.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def has_filler(text: str) -> bool:
    low = (text or "").lower()
    return any(f in low for f in BANNED_FILLER)


def regenerate_as_observation(record: dict, ctx: RunContext) -> str:
    """Force the observation path for one record and re-call (non-batch).

    Temporarily blanks the segment so `build_requests` selects the observation
    prompt, then restores it so the record keeps its extracted metadata.
    """
    saved = record.get("segment", ""), record.get("confidence", "")
    record["segment"], record["confidence"] = "", "low"
    try:
        req = build_requests([record])[0]
    finally:
        record["segment"], record["confidence"] = saved
    p = req["params"]
    msg = guarded_call(
        lambda: ctx.backend.create_message(
            model=p["model"], max_tokens=p["max_tokens"],
            system=p["system"], messages=p["messages"]),
        tracker=ctx.tracker, model=p["model"],
        in_tokens=len(str(p["messages"])) // 4 + 100, out_tokens=p["max_tokens"])
    return clean(msg.content[0].text.strip())
