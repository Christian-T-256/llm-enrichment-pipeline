"""Structured extraction + reconciliation + grounding.

Each record carries two independent text sources: a short `profile_text`
(think: a self-description) and longer `website_text` (scraped marketing copy).
We extract a single attribute from each (the **segment** the entity serves),
then reconcile, sanity-check, and ground-verify it.

Two design decisions worth calling out:

* **Tool-use, not "respond only with JSON."** Every extraction is a forced
  tool call with an explicit `input_schema`. The prose-JSON approach this
  replaced produced a low-but-nonzero rate of unparseable output (truncation,
  stray preamble, and (the nasty one) a list where a string was promised that
  blew up `.strip()` three stages downstream). Schema enforcement deleted that
  entire failure class. `_coerce_to_str` stays as belt-and-suspenders anyway.

* **Two sources, then reconcile.** A single source lies confidently. Extracting
  from profile and website independently and reconciling turns a guess into a
  cross-checked claim, and the disagreement signal itself becomes confidence.
"""

from __future__ import annotations

from .config import EXTRACT_MODEL
from .context import RunContext
from .errors import CreditOutError
from .spend import guarded_call

# ── Tool schemas (forced structured output) ───────────────────────────────────

PROFILE_SEGMENT_TOOL = {
    "name": "extract_segment_from_profile",
    "description": ("Extract the market segment the entity explicitly says it "
                    "serves, from its self-description. State only what is "
                    "written; do not infer from a job title alone."),
    "input_schema": {
        "type": "object",
        "properties": {
            "segment": {"type": "string",
                        "description": "The served segment in one short phrase. "
                                       "Empty string if not clearly stated."},
            "confidence": {"type": "string", "enum": ["high", "low"]},
        },
        "required": ["segment", "confidence"],
    },
}

WEBSITE_SEGMENT_TOOL = {
    "name": "extract_segment_from_website",
    "description": ("Extract the market segment the entity serves, from its "
                    "website copy. Prefer case-study / services / about evidence "
                    "over a generic homepage tagline."),
    "input_schema": {
        "type": "object",
        "properties": {
            "segment": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "low"]},
            "evidence": {"type": "string",
                         "description": "One short quoted snippet supporting the "
                                        "claim, or empty string."},
        },
        "required": ["segment", "confidence", "evidence"],
    },
}

RECONCILE_TOOL = {
    "name": "reconcile_segments",
    "description": "Decide whether two segment descriptions for the same entity refer to the same target.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agree": {"type": "string", "enum": ["yes", "no"]},
            "merged_segment": {"type": "string",
                               "description": "Best single phrase; prefer the more specific."},
            "reasoning": {"type": "string"},
        },
        "required": ["agree", "merged_segment", "reasoning"],
    },
}

VERIFY_TOOL = {
    "name": "verify_segment_grounded",
    "description": ("Verify whether a segment claim is supported by the source "
                    "text. Be strict: if the claim needs inference beyond what "
                    "is written, answer no."),
    "input_schema": {
        "type": "object",
        "properties": {
            "grounded": {"type": "string", "enum": ["yes", "no"]},
            "reasoning": {"type": "string"},
        },
        "required": ["grounded", "reasoning"],
    },
}

# ── Generic-vagueness backstop (no LLM) ────────────────────────────────────────

_GENERIC_TERMS = {
    "businesses", "companies", "clients", "customers", "people", "professionals",
    "organizations", "teams", "brands", "founders", "owners", "leaders",
    "everyone", "anyone", "small businesses", "growing companies",
    "ambitious professionals", "modern teams",
}
_STOP = {"the", "and", "or", "for", "to", "a", "an", "that", "who", "with",
         "ambitious", "growing", "modern", "innovative", "great", "good"}


def _is_vague(segment: str) -> bool:
    """True if the segment is all generic roles/modifiers with no niche anchor.

    Cheap, deterministic guard that runs after reconciliation: it catches the
    'serves businesses'-class non-answers the LLM marks `high` because they're
    technically supported by the text. A vague segment makes the downstream
    note generic, so we downgrade its confidence rather than trust it.
    """
    s = (segment or "").strip().lower()
    if not s:
        return True
    if s in _GENERIC_TERMS:
        return True
    meaningful = [w for w in s.replace(",", " ").split()
                  if w not in _STOP and w not in _GENERIC_TERMS]
    return len(meaningful) == 0


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _tool_input(msg, tool_name: str) -> dict:
    """Pull the forced tool_use block's input. Empty dict = soft failure."""
    for block in getattr(msg, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            inp = getattr(block, "input", None)
            return inp if isinstance(inp, dict) else {}
    return {}


def _coerce_to_str(val) -> str:
    """Normalize a segment field. The model has historically returned a list
    when multiple segments applied; join those. Defensive layer behind the
    schema (which already pins `type: string`)."""
    if val is None:
        return ""
    if isinstance(val, list):
        return "; ".join(str(x).strip() for x in val if str(x).strip())
    if isinstance(val, dict):
        return ""
    return str(val).strip()


# ── Stage functions ────────────────────────────────────────────────────────────

def extract_from_profile(record: dict, ctx: RunContext) -> tuple[str, str]:
    fields = "\n".join(f"- {k}: {record.get(k, '')}"
                       for k in ("name", "headline", "profile_text") if record.get(k))
    prompt = ("Extract the segment this entity explicitly says it serves.\n\n"
              f"{fields}\n\nReturn an empty segment if none is clearly stated.")
    msg = guarded_call(
        lambda: ctx.backend.create_message(
            model=EXTRACT_MODEL, max_tokens=150,
            system="You extract an explicitly stated served segment. "
                   "Call extract_segment_from_profile.",
            messages=[{"role": "user", "content": prompt}],
            tools=[PROFILE_SEGMENT_TOOL],
            tool_choice={"type": "tool", "name": "extract_segment_from_profile"}),
        tracker=ctx.tracker, model=EXTRACT_MODEL,
        in_tokens=len(prompt) // 4 + 120, out_tokens=60)
    data = _tool_input(msg, "extract_segment_from_profile")
    return _coerce_to_str(data.get("segment")), str(data.get("confidence", "low")).lower()


def extract_from_website(website_text: str, ctx: RunContext) -> tuple[str, str, str]:
    sliced = (website_text or "")[: ctx.settings.limits.extract_source_chars]
    prompt = f"Website content:\n{sliced}\n\nExtract the segment this entity serves."
    msg = guarded_call(
        lambda: ctx.backend.create_message(
            model=EXTRACT_MODEL, max_tokens=300,
            system="You extract an explicitly served segment from website copy. "
                   "Call extract_segment_from_website.",
            messages=[{"role": "user", "content": prompt}],
            tools=[WEBSITE_SEGMENT_TOOL],
            tool_choice={"type": "tool", "name": "extract_segment_from_website"}),
        tracker=ctx.tracker, model=EXTRACT_MODEL,
        in_tokens=len(prompt) // 4 + 120, out_tokens=120)
    data = _tool_input(msg, "extract_segment_from_website")
    return (_coerce_to_str(data.get("segment")),
            str(data.get("confidence", "low")).lower(),
            str(data.get("evidence", "")))


def reconcile(profile_seg: str, profile_conf: str,
              website_seg: str, website_conf: str,
              ctx: RunContext) -> tuple[str, str, str]:
    """Fuse the two extractions into (segment, confidence, source).

    The disagreement signal is the point: two confident-but-different sources
    yield `medium`, not `high`. A lone source is trusted only as far as its own
    stated confidence. Confidence ladder is high > medium > low.
    """
    p, w = profile_seg.strip(), website_seg.strip()
    if not p and not w:
        return "", "low", "none"
    if p and not w:
        return p, ("high" if profile_conf == "high" else "low"), "profile"
    if w and not p:
        return w, ("high" if website_conf == "high" else "low"), "website"

    prompt = (f'Segment A (profile): "{p}"\nSegment B (website): "{w}"\n'
              "Do these describe the same target?")
    msg = guarded_call(
        lambda: ctx.backend.create_message(
            model=EXTRACT_MODEL, max_tokens=200,
            system="You compare two segment descriptions. Call reconcile_segments.",
            messages=[{"role": "user", "content": prompt}],
            tools=[RECONCILE_TOOL],
            tool_choice={"type": "tool", "name": "reconcile_segments"}),
        tracker=ctx.tracker, model=EXTRACT_MODEL,
        in_tokens=len(prompt) // 4 + 120, out_tokens=80)
    data = _tool_input(msg, "reconcile_segments")
    merged = _coerce_to_str(data.get("merged_segment")) or (w if len(w) >= len(p) else p)
    agree = str(data.get("agree", "no")).lower() == "yes"
    if agree:
        conf = "high" if profile_conf == "high" and website_conf == "high" else "medium"
    else:
        conf = "medium"  # both present but disagree → usable, not trusted
    return merged, conf, "reconciled"


def verify_grounded(record: dict, segment: str, website_text: str,
                    ctx: RunContext) -> bool:
    """Strict hallucination check: is `segment` actually supported by the source?

    Fails *open* on a verifier error (returns True). A flaky verifier should not
    silently drop good claims. The caller treats a `False` here as 'downgrade one
    notch', not 'discard', so a wrong claim still loses its `high`.
    """
    if not segment:
        return True
    cap = ctx.settings.limits.grounding_source_chars
    profile = "\n".join(f"- {k}: {record.get(k, '')[:400]}"
                        for k in ("headline", "profile_text") if record.get(k))
    site = (website_text or "")[:cap]
    source = (profile + ("\n\n[WEBSITE]\n" + site if site else ""))[: cap * 2]
    prompt = (f"Source text:\n{source}\n\nClaimed segment: \"{segment}\"\n"
              "Is the claim supported by the source?")
    try:
        msg = guarded_call(
            lambda: ctx.backend.create_message(
                model=EXTRACT_MODEL, max_tokens=250,
                system="You verify a segment claim against source text. "
                       "Call verify_segment_grounded.",
                messages=[{"role": "user", "content": prompt}],
                tools=[VERIFY_TOOL],
                tool_choice={"type": "tool", "name": "verify_segment_grounded"}),
            tracker=ctx.tracker, model=EXTRACT_MODEL,
            in_tokens=len(prompt) // 4 + 120, out_tokens=60)
    except CreditOutError:
        raise  # out of money is fatal to the whole run, not soft to one record
    except Exception as e:  # noqa: BLE001 (any other verifier glitch fails open)
        print(f"  [verify] soft error, failing open: {e}", flush=True)
        return True
    data = _tool_input(msg, "verify_segment_grounded")
    return str(data.get("grounded", "yes")).lower() == "yes"
