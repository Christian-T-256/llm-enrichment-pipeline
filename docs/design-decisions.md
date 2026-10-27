# Design decisions

Notes on *why* the pipeline is shaped the way it is. Each decision traces back to
a concrete failure mode in the production system this was distilled from. The
pattern throughout: **the demo version of an LLM pipeline is easy; the version
that survives 11,000 records and a metered API bill is mostly guardrails.**

---

## 1. Structured output via tool-use, not prose-JSON

**The naive version.** "Respond ONLY with valid JSON: `{"segment": ..., "confidence": ...}`" in the system prompt, then `json.loads()` the text.

**What broke.** At scale this fails a small but nonzero fraction of the time, in three flavors:
- truncation mid-object when the model bumps `max_tokens`,
- a stray sentence of preamble before the `{`,
- and the subtle one: a *list* where a string was promised (`["startups", "agencies"]`), which sailed through `json.loads()` and then crashed `.strip()` three stages later in the reconciler. Some of these didn't crash at all; they shipped as a silent blank.

**The fix.** Every extraction is a *forced* tool call (`tool_choice={"type": "tool", ...}`) with an explicit `input_schema` pinning `segment` to `type: string` and `confidence` to an `enum`. The model can no longer return malformed shapes, and `_tool_input()` reads the parsed block directly, with no `json.loads`, no markdown stripping. The entire parse-error class disappeared.

**Belt and suspenders.** `_coerce_to_str()` still normalizes a list-to-string just in case, because the cost of keeping a four-line defensive helper is nothing and the cost of a silent blank in a downstream artifact is high.

See `extract.py`.

---

## 2. Two independent extractions, then reconcile

A single source describes an entity the way *it* wants to be seen, and the model will confidently extract whatever's there. Pulling the attribute from **two** independent sources (a short self-description and longer marketing copy) and reconciling them does two things:

1. It catches the case where one source is wrong, stale, or about a different entity.
2. **The disagreement becomes the confidence score.** Two sources that agree and are both confident → `high`. Two confident sources that *disagree* → `medium`, never `high`. The conflict is information, not noise. A lone source is trusted only as far as its own stated confidence.

This is cheaper than it sounds (two Haiku calls) and it's the difference between "the model guessed" and "two sources cross-checked."

---

## 3. Grounding verifier: why it fails *soft*

After reconciliation, a separate strict pass asks one question: *is this segment actually supported by the source text, or did the model infer beyond what's written?* This is the hallucination guard.

The subtlety is in the failure handling:
- If a `high`-confidence claim comes back **not grounded**, it drops *one notch* (`high → medium`) rather than being discarded. The claim is probably still useful; it just shouldn't be trusted at the top tier.
- If the **verifier itself** errors (timeout, transient blip), it **fails open** and returns "grounded." A flaky check must not silently delete good data. The only thing that can halt the verifier hard is a credit-out, which is fatal to the whole run anyway.

The result is hallucination control that degrades gracefully instead of veering between "trust everything" and "discard everything."

---

## 4. Refusal recovery

When a record's two sources genuinely look like different entities (a name collision, the wrong website scraped), the model sometimes does the "honest" thing and *refuses*, returning a paragraph explaining why it can't write a good note.

That paragraph is a trap. A "non-empty output" validation passes it, and it ships **as the note**. You've now sent a customer-facing artifact that is literally a meta-complaint about data quality.

`is_refusal()` catches it on two signals: refusal phrasing ("I cannot", "appears unrelated", "mismatched"), or simply being far longer than a one/two-sentence note has any business being. On a hit, `regenerate_as_observation()` blanks the segment, forces the simpler observation-only prompt (which doesn't require the two sources to agree), and re-calls. The recovery is logged in the output (`note_route = observation(recovered)`) so the rate is auditable.

See `generate.py`.

---

## 5. The hard spend cap

`--max-cost` is **required**. There is intentionally no default and no way to start a run without declaring a ceiling. A running estimate (`SpendTracker`) updates after every call and, the moment it crosses the cap, the run aborts cleanly with exit code 2, dumps a spend summary, and writes the audit log.

Two deliberate choices:
- **Estimates are upper bounds, not actuals.** The point is to trip *before* money moves, which is exactly when you want to know. Reconciling against the real invoice after the fact is too late.
- **The cap is checked at the chokepoint**, not sprinkled around. Every model call routes through `guarded_call`, so there's exactly one place the cap can be bypassed: nowhere.

---

## 6. Credit-out fail-fast + pre-batch healthcheck

**Fail-fast.** Out-of-credit / over-billing-limit is *permanent*. Retrying it just burns the rate limit and the clock. So `CreditOutError` is raised immediately and never retried, and it aborts the whole run, not a single record. The trap: the Batch endpoint sometimes returns credit-out **wrapped in a Cloudflare 5xx**, which looks transient. So classification sniffs the *message body* for credit-balance markers before deciding, rather than trusting the HTTP status.

**Healthcheck.** Before submitting N expensive batch requests, fire one trivial request and cancel it. If the account is dry or the endpoint is down, you find out for ~$0.0001 instead of after committing the whole batch.

Both guards exist because the un-guarded version once retried a 5xx-wrapped credit-out through an entire batch window, losing the budget *and* the hours of upstream extraction that fed the batch.

See `spend.py` and `errors.py`.

---

## 7. Checkpoint/resume + first-run review gate

**Checkpoint.** The upstream extraction is the expensive part (four model calls per record). If the batch submission then dies, that work is gone unless it was written down. `checkpoint.write()` persists the post-extraction state to disk right before batch submission; `--resume-checkpoint` reloads it and skips straight to generation. A crash becomes a cheap restart instead of a full re-run.

**Review gate.** Generation pipelines drift silently: a prompt tweak or a model bump can degrade output without anyone noticing, and "the run completed" tells you nothing about quality. So the *first* run on any dataset stops, dumps a 30-row sample to CSV, and blocks until a human creates an approval marker. Subsequent runs auto-pass. It's deliberately low-tech (a CSV and a touch-file) because the goal is a *forced human glance* before thousands of records ship, not a UI.

See `checkpoint.py` and `review_gate.py`.

---

## Cross-cutting: a vendor-agnostic core

The pipeline never touches the vendor SDK directly. It talks to a small `Backend`
protocol; the real adapter maps the SDK's exception zoo onto a three-class
taxonomy (`CreditOutError` / `PermanentError` / `TransientError`) at the
boundary. That single translation layer is what lets the retry policy be
identical for the real and mock backends, and it's why the entire pipeline,
batch round-trip included, runs offline and deterministically in tests.

The lesson encoded here: **the error taxonomy that matters to an LLM pipeline
isn't the vendor's. It's "will this succeed if I try again, and should I spend
money doing so."** Three classes. Everything else is detail.
