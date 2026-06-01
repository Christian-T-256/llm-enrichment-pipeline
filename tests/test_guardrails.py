"""Tests for the parts that protect money and output quality."""

import csv

import pytest

from enrichment import errors
from enrichment.backends import MockBackend
from enrichment.config import Settings
from enrichment.context import RunContext
from enrichment.extract import _coerce_to_str, _is_vague, reconcile
from enrichment.generate import clean, has_filler, is_refusal
from enrichment.pipeline import run
from enrichment.spend import SpendCapExceeded, SpendTracker, guarded_call


# ── Spend cap ──────────────────────────────────────────────────────────────────

def test_spend_cap_aborts_when_estimate_exceeds_cap():
    t = SpendTracker(max_cost=0.0001)  # microscopic cap
    with pytest.raises(SpendCapExceeded) as exc:
        # 1M Sonnet input tokens ≈ $3, far over the cap.
        t.record("claude-sonnet-4-6", 1_000_000, 0, batch=False)
    assert exc.value.code == 2


def test_batch_discount_halves_cost():
    std = SpendTracker(max_cost=999)
    std.record("claude-sonnet-4-6", 1_000_000, 0, batch=False)
    bat = SpendTracker(max_cost=999)
    bat.record("claude-sonnet-4-6", 1_000_000, 0, batch=True)
    assert bat.estimate() == pytest.approx(std.estimate() * 0.5)


# ── guarded_call retry policy ───────────────────────────────────────────────────

def test_credit_out_is_never_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise errors.CreditOutError("credit balance is too low")

    t = SpendTracker(max_cost=999)
    with pytest.raises(errors.CreditOutError):
        guarded_call(fn, tracker=t, model="claude-haiku-4-5")
    assert calls["n"] == 1  # exactly one attempt, no retries


def test_permanent_error_is_never_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise errors.PermanentError("400 bad request")

    t = SpendTracker(max_cost=999)
    with pytest.raises(errors.PermanentError):
        guarded_call(fn, tracker=t, model="claude-haiku-4-5")
    assert calls["n"] == 1


def test_transient_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("enrichment.spend.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise errors.TransientError("503")
        return "ok"

    t = SpendTracker(max_cost=999)
    assert guarded_call(fn, tracker=t, model="claude-haiku-4-5") == "ok"
    assert calls["n"] == 3


# ── Output quality guards ────────────────────────────────────────────────────────

def test_coerce_list_to_string():
    assert _coerce_to_str(["startups", "agencies"]) == "startups; agencies"
    assert _coerce_to_str("founders") == "founders"
    assert _coerce_to_str(None) == ""


def test_vagueness_backstop():
    assert _is_vague("businesses")
    assert _is_vague("ambitious professionals")
    assert not _is_vague("debut science-fiction authors")


def test_clean_preserves_terminal_period_but_strips_wrapping_quotes():
    assert clean('"A clean one-liner."') == "A clean one-liner."
    assert clean("Ends with a period.") == "Ends with a period."


def test_refusal_detection():
    assert is_refusal("I cannot write this because the data is mismatched.", 45)
    assert is_refusal("word " * 60, 45)            # too long → prose
    assert not is_refusal("A crisp two-sentence note. It fits.", 45)


def test_filler_detection():
    assert has_filler("A world-class, cutting-edge solution.")
    assert not has_filler("A small studio that ships indie games.")


# ── Reconciliation policy ────────────────────────────────────────────────────────

def _ctx():
    return RunContext(MockBackend(), SpendTracker(max_cost=999),
                      Settings(max_cost=999, dry_run=True))


def test_reconcile_single_source_respects_its_confidence():
    seg, conf, src = reconcile("founders", "high", "", "low", _ctx())
    assert (seg, conf, src) == ("founders", "high", "profile")


def test_reconcile_both_present_disagree_yields_medium():
    seg, conf, src = reconcile("wedding couples", "high",
                               "hvac contractors", "high", _ctx())
    assert conf == "medium" and src == "reconciled"


# ── End-to-end dry run ─────────────────────────────────────────────────────────────

def test_dry_run_end_to_end_produces_notes(tmp_path):
    src = tmp_path / "in.csv"
    with open(src, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "company", "headline",
                                          "profile_text", "website_text"])
        w.writeheader()
        w.writerow({"name": "Larkspur", "company": "Larkspur Clinic",
                    "headline": "Sports PT",
                    "profile_text": "We help endurance runners recover.",
                    "website_text": "Larkspur helps endurance runners return from injury."})

    out = tmp_path / "out.csv"
    ctx = RunContext(MockBackend(), SpendTracker(max_cost=5),
                     Settings(max_cost=5, dry_run=True, review_gate=False))
    summary = run(str(src), str(out), ctx,
                  dataset_key="test", gate_dir=str(tmp_path / "gate"))

    assert summary["status"] == "OK"
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["note"]                       # a note was produced
    assert "endurance runners" in rows[0]["segment"]
    assert rows[0]["confidence"] in ("high", "medium")
    assert rows[0]["note_route"] == "anchored"   # confident segment → anchored route
