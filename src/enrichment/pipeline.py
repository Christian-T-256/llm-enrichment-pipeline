"""End-to-end orchestration.

    load → eligibility filter → [ per record: extract×2 → reconcile → vagueness
    backstop → grounding verify ] → checkpoint → review gate → batch generate →
    refusal recovery + output validation → save

Each stage is small and testable on its own; this module is just the wiring and
the policy decisions about how confidence flows between them.
"""

from __future__ import annotations

import csv
import os

from . import checkpoint, generate, review_gate
from .config import Settings
from .context import RunContext
from .extract import (
    _is_vague, extract_from_profile, extract_from_website, reconcile, verify_grounded,
)

# Final confidence ladder, low → high.
_LADDER = ["low", "medium", "high"]
OUTPUT_FIELDS = ["segment", "confidence", "source", "grounded", "evidence",
                 "note", "note_route", "note_flag"]


def _downgrade(conf: str) -> str:
    i = max(0, _LADDER.index(conf) - 1) if conf in _LADDER else 0
    return _LADDER[i]


def load_records(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r.setdefault("_row_id", f"rec_{i:05d}")
        r["_website_text"] = r.get("website_text", "")
    return rows


def save_records(records: list[dict], path: str) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base = [k for k in records[0].keys() if not k.startswith("_")]
    fields = base + [f for f in OUTPUT_FIELDS if f not in base]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  saved {len(records)} rows -> {path}", flush=True)


def is_eligible(record: dict) -> bool:
    """Cost gate, not a quality gate. We require website text because it is the
    load-bearing anchor for *both* generation routes; with neither website nor
    profile text there is nothing to extract or write from. We deliberately do
    NOT also require a profile — an over-strict AND-filter here drops records the
    observation fallback could have handled. Filter at the latest safe stage."""
    return bool((record.get("website_text") or "").strip() or
                (record.get("profile_text") or "").strip())


def _enrich_one(record: dict, ctx: RunContext) -> None:
    """Run the extraction stack on one record, writing results back onto it."""
    p_seg, p_conf = extract_from_profile(record, ctx)
    w_seg, w_conf, evidence = extract_from_website(record.get("_website_text", ""), ctx)
    segment, confidence, source = reconcile(p_seg, p_conf, w_seg, w_conf, ctx)

    if segment and _is_vague(segment):
        confidence = "low"  # generic non-answer the model over-trusted

    grounded = verify_grounded(record, segment, record.get("_website_text", ""), ctx)
    if confidence == "high" and not grounded:
        confidence = _downgrade(confidence)  # fail-soft: doubt it, don't trust it

    record.update(segment=segment, confidence=confidence, source=source,
                  grounded="yes" if grounded else "no", evidence=evidence)


def run(input_path: str, output_path: str, ctx: RunContext, *,
        dataset_key: str, gate_dir: str) -> dict:
    """Run the pipeline. Returns a small summary dict."""
    settings: Settings = ctx.settings

    # --- Generation-only resume path ---------------------------------------
    if settings.resume_checkpoint:
        records = checkpoint.load(settings.resume_checkpoint)
        for r in records:
            r["_website_text"] = r.get("website_text", r.get("_website_text", ""))
    else:
        raw = load_records(input_path)
        records = [r for r in raw if is_eligible(r)]
        dropped = len(raw) - len(records)
        print(f"  eligibility: {len(records)} eligible, {dropped} dropped "
              f"(no usable source text)", flush=True)

        for i, r in enumerate(records, 1):
            _enrich_one(r, ctx)
            if i % 25 == 0:
                print(f"  extracted {i}/{len(records)}", flush=True)

        # Human glance before anything expensive ships (first run only).
        if settings.review_gate and not review_gate.require_approval(
                records, gate_dir, dataset_key):
            return {"status": "AWAITING_REVIEW", "records": len(records)}

        checkpoint.write(records, output_path)

    # --- Batch generation ---------------------------------------------------
    requests = generate.build_requests(records)
    results = generate.submit_and_poll(requests, ctx)

    recovered = flagged = 0
    by_id = {r["_row_id"]: r for r in records}
    for cid, text in results.items():
        r = by_id.get(cid)
        if r is None:
            continue
        note = generate.clean(text)
        route = "anchored" if generate._is_anchored(r) else "observation"
        if generate.is_refusal(note, settings.limits.note_max_words):
            note = generate.regenerate_as_observation(r, ctx)
            route = "observation(recovered)"
            recovered += 1
        flag = ""
        if generate.has_filler(note):
            flag = "filler"            # surfaced for review rather than silently shipped
            flagged += 1
        r.update(note=note, note_route=route, note_flag=flag)

    save_records(records, output_path)
    ctx.tracker.summary()
    ctx.tracker.write_log("OK")
    return {"status": "OK", "records": len(records), "recovered_refusals": recovered,
            "flagged_notes": flagged, "estimated_cost": round(ctx.tracker.estimate(), 4)}
