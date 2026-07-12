# Output Consistency — exgentic traces on RITS Qwen3-VL-235B-A22B-Instruct

**Experiment:** first 10 exgentic agent traces, replayed **10× each** against the live
RITS endpoint with `disable_output_substitution: true`, server-default sampling.
Analyzed 2026-07-01. Raw data: `reports-consistency/analysis.json`.

## What the setup guarantees

With substitution off, each (trace, call-position) is fed **byte-identical input** on
all 10 runs. So every difference below is **pure serving-stack nondeterminism**
(sampling temperature + vLLM batching/numerics), not compounding input drift.

- 10 runs completed cleanly (ok=10, fail=0).
- 922 request records → **142 identical-input groups**; **125 usable** (≥2 successful
  responses). 17 groups had <2 usable responses (transient HTTP 504s clustered on a few
  early turns, which then cancelled the rest of those session chains — hence per-run
  record counts ranged 11–129). Errors are excluded from consistency math, not counted
  as "inconsistency."

## Headline: the endpoint is *not* deterministic, but is *mostly* semantically stable

| Metric | Value | Reading |
|---|---|---|
| Byte-identical across all 10 runs | **11.2%** of groups | exact reproducibility is rare |
| Distinct responses / group | median **6**, up to 9 | given the same input, you usually get 5–9 different strings out of 10 |
| Modal-response frequency | median **0.25** | the single most common exact output appears in only ~2–3 of 10 runs |
| Pairwise Levenshtein similarity | median **0.71** | outputs are ~70% character-similar — same shape, local wording differs |
| Pairwise Jaccard (token sets) | median **0.59** | ~60% word overlap |
| Output length CV | median **0.0** | length is extremely stable (see caveat) |
| finish_reason agreement | median **1.0** | how generation ends is essentially always consistent |
| **Semantic clusters / group (LLM judge)** | median **2**, mean **2.09** | on average ~2 distinct *meanings* per input |

**So:** the model reliably produces the *same kind* of answer (same structure, same
stopping behavior, near-identical length) but rarely the *same exact text*, and a
meaningful fraction of the time it produces a genuinely *different answer*.

## The load-bearing finding: text drift ≠ semantic drift, and it's input-dependent

Of the 108 text-diverged groups the judge scored:
- **31%** still mean **one thing** (1 cluster) — pure cosmetic wording variation.
- **69%** carry **≥2 distinct meanings** — substantive divergence.

And semantic divergence scales with lexical divergence, monotonically:

| Lexical similarity | mean semantic clusters |
|---|---|
| ≥ 0.90 | 1.47 |
| 0.70–0.90 | 1.97 |
| < 0.70 | 2.32 |

This is the actionable result: **pairwise Levenshtein is a usable cheap proxy** — when
outputs are ≥90% character-similar they usually (not always) mean the same thing; below
0.7 you should expect the model to have actually made different decisions across runs.

## Consistency is dominated by task type, not by chain depth

Per-trace byte-identical rates span the full range (0%–100%). The split is by *what the
turn asks for*, not how deep in the session it is:

- **Closed/lookup answers stay consistent.** `trace_e39d29fce9` (40% identical, lev
  0.82) and the short customer-service traces cluster to **1 meaning** even when wording
  drifts — e.g. "Mei Patel placed an order under user ID mei\_pa…" comes back the same
  answer 10/10 times in different words.
- **Open-ended agentic reasoning is where it diverges.** The one long coding-agent
  session `trace_4934ca45b7` (81 groups, mean lev 0.72, only 10% identical) is where the
  4–5-cluster groups live: the agent picks *different next actions* from the same
  state — "examine serializer code" vs "find serialization test" vs "inspect
  handle\_m2m\_field". These are real behavioral forks, not paraphrases.

Tool-call turns (small n=4): tool **name + arg-keys** agree ~89% of the time, but full
**argument values** agree only ~64% — the model calls the right tool but fills in
arguments inconsistently.

## Caveats

- **Length CV median 0.0 is misleading on its own.** Most turns emit short fixed-shape
  outputs (tool calls), so length barely moves — *except* 2 groups where one run ran
  away to the 4096-token cap while siblings stopped at 7–16 tokens. Length is stable in
  the common case, catastrophically unstable in rare tail cases.
- **11 trace-ids, not 10:** the trace-replay generator leaves `session_id` null on the
  metric (lib bug — `replay_graph_session_datagen.py:1388` passes the lazy stub's `None`
  instead of the extracted session id). The analyzer falls back to hashing each call's
  leading message; one dataset session with varying first-messages split into two ids.
  This does **not** affect the aggregate — groups still join on exact request hash.
- Measures the **deployed serving stack** (model + vLLM sampling/batching), not weights
  in isolation. Sampling was left at server default (real-world), so temperature>0 is
  the primary driver; a temperature=0 rerun would isolate the numerical-nondeterminism
  floor.
- The semantic judge (same 235B model, temp 0) is a soft signal; 1 group failed to
  judge (network).

## Bottom line

Given identical inputs, RITS-served Qwen3-VL-235B **rarely reproduces exact text (11%)**
but **usually preserves the answer's structure, length, and stopping behavior**. Whether
the *meaning* is stable depends on the task: closed lookups stay consistent; open-ended
agentic decisions fork into 2–5 genuinely different trajectories. For agentic workloads,
this means run-to-run reproducibility should not be assumed — the same state can yield
different next actions ~2/3 of the time it diverges lexically.
