# Output Consistency Experiment — exgentic traces on RITS

**Date:** 2026-06-30
**Status:** approved, implementing

## Goal

Measure how *consistent* a RITS-served endpoint is: given **identical inputs**, how
repeatable are its outputs? We run the same 10 exgentic agent traces, 10 times each, and
analyze the differences between the 10 copies of every call. (The reference run used
`Qwen/Qwen3-VL-235B-A22B-Instruct`.)

## Why `disable_output_substitution: true` is the linchpin

This flag makes the measurement clean by removing input drift as a confound.

- **Substitution ON (default):** turn *N*'s input is built from the *live* output the
  model produced at turn *N−1* in that same run. Repetition #3 and #7 of a trace
  diverge at turn 1, and the divergence **compounds** — by turn 5 they are answering
  different questions. You would be measuring cascade drift, not model consistency.
- **Substitution OFF:** every turn always sends the **recorded** prior context. For
  trace *T*, call-position *P*, **all 10 repetitions send byte-identical input**. Any
  difference in the 10 outputs is therefore attributable purely to model/server
  nondeterminism (sampling temperature, vLLM batching/numerical nondeterminism,
  KV-cache effects) — the exact variable we want to isolate.

So the experiment answers: *given identical inputs, how repeatable is the
RITS-served model's output?*

## Design decisions (from brainstorming)

- **Trace selection:** the first 10 exgentic sessions as-is, full session each
  (`num_sessions: 10`, deterministic slice with a fixed `base_seed`).
- **Sampling:** server default (no temperature override) → real-world consistency a
  user would actually see.
- **Repetition:** 10 **independent process invocations** of one config (NOT
  `duplicate_sessions_target`). Each run writes its own report directory. Independent
  processes eliminate any cross-copy interaction within a run.
- **Analysis depth:** both offline (lexical + structural) and semantic (LLM judge).
- **Scope:** build and execute the full experiment, then analyze.

## Architecture

> **As built:** the files described below now live under
> `experiments/consistency-replay/` (tracked, committed). The reference run used
> `Qwen/Qwen3-VL-235B-A22B-Instruct` on RITS with the API key supplied via the
> `RITS_API_KEY` environment variable (no key in the repo). See `README.md` for the
> current commands and layout.

Three artifacts:

### 1. Config — `experiments/consistency-replay/config.yml`
- `data.otel_trace_replay.hf_dataset_path: Exgentic/agent-llm-traces`
- `disable_output_substitution: true`
- `use_static_model: true`, `static_model_name: Qwen/Qwen3-VL-235B-A22B-Instruct`
- `default_max_tokens: 200` (bounds output; keeps runs comparable/cheap)
- `api.headers.RITS_API_KEY` for auth (placeholder in the file; injected at run time); `server.api_key` left unset
- `load.type: trace_session_replay`, one stage `{concurrent_sessions: 10, num_sessions: 10}`
- `load.base_seed` — fixed so all 10 process runs replay the **same** 10 traces
- `report.request_lifecycle.per_request: true` — **the data source**: writes
  `per_request_lifecycle_metrics.json` with full `request` + `response` bodies
- `storage.local_storage.path` overridden per iteration by the runner via the
  `--storage.local_storage.path` CLI flag.

### 2. Runner — `experiments/consistency-replay/run_consistency.sh`
Loops `i` in 1..N, invokes `python -m inference_perf.main --config <cfg>` once per
iteration into `reports-consistency/run_$i`, injecting the API key from `RITS_API_KEY`.
Sequential to avoid self-induced load skew and to be gentle on the flaky gateway.

### 3. Analyzer — `experiments/consistency-replay/analyze_consistency.py`
- Loads `per_request_lifecycle_metrics.json` from all `run_*` dirs.
- **Join key:** `(session_id, sha1(request_payload))`. With substitution off, identical
  inputs across runs yield identical request bodies, giving an exact join. Each group =
  the ≤10 repetitions of one identical-input call.
- Parses each `response` body → `choices[0].message.content`, `tool_calls`,
  `finish_reason`, `usage.completion_tokens`.
- Computes per-group metrics, rolls up per-trace / per-turn-depth / overall.
- Emits `consistency_analysis.json` + a printed human report.

## Metrics

Per group of identical-input outputs (cheapest → most semantic):

| Dimension | Metric | Interpretation |
|---|---|---|
| Completion | successes / 10, error category counts | gateway flakiness vs real inconsistency |
| Exact reproducibility | # distinct responses; modal frequency | is the endpoint deterministic at all? |
| Length stability | mean / stdev / CV of completion tokens | variable rambling even when content stable |
| Lexical similarity | mean pairwise normalized Levenshtein & Jaccard | graded drift when not exactly equal |
| Structural (tool turns) | same tool name? same arg keys? args JSON-equal? finish_reason agreement | agentic reliability |
| Semantic | LLM-judge "same answer?" cluster count over the group | cosmetic vs substantive divergence |

Roll-ups: **per-trace** (stable vs chaotic traces), **per-turn-depth** (does
consistency degrade deeper in a chain even with fixed inputs?), **overall** (e.g.
"X% of call positions byte-identical across all 10 runs; median pairwise similarity Y;
tool-name agreement Z%; N semantic clusters on average").

## Caveats the report must state

- RITS gateway is flaky (intermittent 502/504/400). The analyzer treats error
  responses as a separate category and reports completion rate per group, so transient
  failures do not masquerade as "inconsistency."
- This measures the **serving-stack** consistency as deployed (model + vLLM batching +
  sampling), not model weights in isolation.
- Semantic judging adds its own nondeterminism; the judge runs at temp 0 and reports
  cluster counts, used as a soft signal, not ground truth.
- exgentic traces vary in length; per-turn-depth roll-up is reported only where enough
  traces reach that depth.

## Out of scope (YAGNI)

- No temperature sweep (single real-world setting).
- No `duplicate_sessions_target` / `inject_random_session_id`.
- No retry-until-success loop in the runner; failures are reported, not papered over.
