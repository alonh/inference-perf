---
name: consistency-experiment
description: Run the end-to-end output-consistency experiment — replay the same trace(s) N times against a live server with output substitution disabled, analyze how much the repeated outputs differ (exact-match, Levenshtein/Jaccard, tool-call structure, LLM-judge semantic clustering), and build an interactive side-by-side HTML viewer. Use when the user wants to measure how consistent/deterministic a model endpoint's outputs are given identical inputs, or asks to re-run / adapt this experiment for a different model, dataset, or trace set.
user-invocable: true
---

# Output Consistency Experiment

Measures: **given byte-identical inputs, how repeatable are a model endpoint's outputs?**

The whole design hinges on `disable_output_substitution: true`. With it on, every
repetition of a given (trace, call-position) is fed the **recorded** prior context, so
all N runs send **identical input** at each call. Any difference in the outputs is then
**pure serving-stack nondeterminism** (sampling temperature + vLLM batching/numerics),
not compounding cascade drift. Without the flag you'd measure input divergence instead.

## Prerequisites

- Python venv at the repo root (`.venv/`); invoke it as `.venv/bin/python` from the repo root.
- A reachable inference endpoint. Default is RITS (header auth `RITS_API_KEY`, NOT Bearer).
  **The RITS gateway is flaky** — probe it before a long run (step 0).
- **Ask the user for the RITS API key, then `export RITS_API_KEY=<key>`** in the shell.
  The runner and the `--judge` analyzer step both read it from the environment; no key is
  stored in the repo (the config holds a placeholder that the runner overrides).
- The `disable_output_substitution` feature must be present — use branch
  `feat/disable-output-substitution` or a branch based on it.
- Run all commands from the **repo root** (dataset/trace paths resolve against CWD).

## Files this skill uses (all under `experiments/consistency-replay/`)

| File | Role |
|---|---|
| `experiments/consistency-replay/config.yml` | The experiment config (placeholder API key) |
| `experiments/consistency-replay/run_experiment.sh` | **The runner.** Model + benchmark on the command line; replays a named session set N times each, one process per (session, repetition) cell. `new` / `continue` modes; injects `$RITS_API_KEY` |
| `experiments/consistency-replay/list_sessions.py` | Names which sessions to replay (seeded shuffle or explicit ids) → the append-only `sessions.json` ledger |
| `experiments/consistency-replay/migrate_to_session_major.py` | Converts an old run-major results dir to session-major, into a new dir (source read-only) |
| `experiments/consistency-replay/run_consistency.sh` | Older fixed-config runner: runs `config.yml` N times, one report dir per run (run-major) |
| `experiments/consistency-replay/analyze_consistency.py` | Groups by (trace, identical-input) and computes metrics |
| `experiments/consistency-replay/consistency_statistics.py` | Paper-grounded metrics: Yagubyan TSS/AC + Hypothesis 1; Raj U-statistic θ+CI; run×run matrices; cross-condition MMD |
| `experiments/consistency-replay/export_viewer_data.py` | Exports per-group data (texts + pairwise matrices) for the viewer |
| `experiments/consistency-replay/viewer_template.html` | The viewer UI (data injected at build time) |
| `experiments/consistency-replay/build_viewer.py` | Export + inject → single self-contained HTML |
| `experiments/consistency-replay/README.md` | Human-facing quick start |

Results land in `reports-consistency/`. Two layouts exist:

- **session-major** (`run_experiment.sh`, current) —
  `<benchmark>/<model>/<stamp>/sessions/<session_id>/run_<i>/`, plus `sessions.json`,
  `experiment.json`, `coverage.json`.
- **run-major** (`run_consistency.sh`, and everything collected before 2026-08-12) —
  `run_<i>/` each holding all ten sessions interleaved.

**The analysis steps below only understand run-major** (`replay_parsing.find_run_dirs` globs
`<base>/run_*`). On a session-major directory `analyze_consistency.py` exits non-zero with
"No run_* directories found" — it does not silently produce an empty analysis. Until the
finder is updated, either analyze a run-major directory or convert one with
`migrate_to_session_major.py`.

## Procedure

Create a todo per step and work them in order.

> **Analysis-only (reuse existing runs).** If `reports-consistency/run_*` already exist
> and you only want (re)analysis — not fresh repetitions — **skip Steps 0–1** and go
> straight to Step 2 / Step 2c / Step 3. The analysis and build steps run **fully offline
> against the existing run dirs**: no endpoint, no `RITS_API_KEY` — *unless* you pass
> `--judge` (Step 2) or `--kernel judge` (Step 2c), which are the only network calls.

### Step 0 — Probe the endpoint (don't skip)

Ask the user for the RITS key and `export RITS_API_KEY=<key>`, then test the endpoint:
```bash
.venv/bin/python - <<'PY'
import os, urllib.request, json, time
url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen3-vl-235b-a22b-instruct/v1/chat/completions"
hdr={"RITS_API_KEY":os.environ["RITS_API_KEY"],"Content-Type":"application/json"}
body=json.dumps({"model":"Qwen/Qwen3-VL-235B-A22B-Instruct","messages":[{"role":"user","content":"Say ok."}],"max_tokens":16,"temperature":0}).encode()
t=time.time()
try:
    r=urllib.request.urlopen(urllib.request.Request(url,data=body,headers=hdr),timeout=45)
    print("UP",round(time.time()-t,1),json.loads(r.read())["choices"][0]["message"].get("content"))
except Exception as e:
    print("DOWN",round(time.time()-t,1),type(e).__name__,str(e)[:100])
PY
```

If it prints `DOWN`, do not launch the runs — either wait/retry, or switch the model
(see "Adapting" below). A single-run smoke test (`--storage.local_storage.path reports-consistency/smoke`)
before the full N is a good idea; expect a small fraction of transient 502/504s even when up.

### Step 1 — Replay each session N times (default 10 sessions × 10 repetitions)

```bash
# Cheapest end-to-end check first: one session, two repetitions.
N_SESSIONS=1 experiments/consistency-replay/run_experiment.sh new \
    Qwen/Qwen3-VL-235B-A22B-Instruct tau2_airline 2

# Then the full grid.
experiments/consistency-replay/run_experiment.sh new \
    Qwen/Qwen3-VL-235B-A22B-Instruct tau2_airline 10
```

Each **cell** = one independent process replaying **one** session
(`concurrent_sessions: 1`, `num_sessions: 1`, session pinned by the replay filter) into
`sessions/<session_id>/run_<i>/`. The session set is chosen by `list_sessions.py` — the first
`N_SESSIONS` (10) after a `SESSION_SEED` (41) shuffle, which reproduces the exact sessions the
older run-major corpora replayed — and recorded in `sessions.json`.

Env knobs: `N_SESSIONS`, `SESSION_SEED`, `SESSION_IDS` (explicit ids, overriding both),
`PARALLEL_REPS` (10 — how many repetitions of the *same* session run at once; matches the
`concurrent_sessions: 10` of the run-major config, so request concurrency at the endpoint is
unchanged. Each cell is pinned to `num_workers: 1`, so the process count tracks the cell count.
Set to 1 for a strictly sequential run), `REQUEST_TIMEOUT` (1200), `RUN_ANALYSIS` (0).

Per-cell record counts vary (a 504 on an early turn cancels the rest of that session chain) —
expected and handled by the analyzer. A failure now costs **one cell**, not all ten sessions.
Read the `coverage.json` / printed grid at the end to see exactly which cells are missing,
then fill them in:

```bash
experiments/consistency-replay/run_experiment.sh continue <OUT_BASE> 10
```

`continue` re-runs only missing or unfinished cells (wiping a partial cell dir first), leaves
complete ones untouched, and never re-queries HuggingFace. To **add a session** to a finished
experiment — appends to the ledger and runs only that session's cells, touching no existing
file:

```bash
SESSION_IDS=<session_id> experiments/consistency-replay/run_experiment.sh continue <OUT_BASE> 10
```

Long-running; run in the background and monitor the per-cell logs under
`<OUT_BASE>/logs/<session_id>/run_<i>.log`.

**Do not delete anything to "start clean."** Every `new` invocation gets a fresh timestamped
directory, so previous experiments are never overwritten.

### Step 2 — Analyze (offline metrics + LLM-judge semantic clustering)

```bash
.venv/bin/python experiments/consistency-replay/analyze_consistency.py reports-consistency \
    --judge --out reports-consistency/analysis.json
```

Drop `--judge` to skip the semantic pass (no network, faster). Prints a summary +
per-trace table; writes full per-group detail to `analysis.json`.

### Step 2c — Paper-grounded analysis (optional, offline)

A second analyzer implementing two papers on agent consistency. It reuses the same
`run_*` dirs and writes a **separate** file (`analysis_papers.json`) that Step 3 folds
into the viewer as a distinct **"Paper metrics"** tab — the existing `analysis.json`
view is untouched.

```bash
.venv/bin/python experiments/consistency-replay/consistency_statistics.py \
    --condition base=reports-consistency \
    --out reports-consistency/analysis_papers.json
```

Stdlib only, runs offline (~45s for 10 runs). Computes, per condition:
- **Yagubyan (arXiv:2605.28840)** — TSS (tool-name-sequence similarity), AC (argument
  consistency), and the **Hypothesis 1** check `E[TSS] ≫ E[AC]`.
- **Raj et al. (arXiv:2605.10516)** — U-statistic consistency **θ with a 95% CI**
  (`--kernel exact|levenshtein|judge`, default `exact`); `--kernel judge` is the only
  option that calls the network. Plus a **trajectory** U-statistic θ+CI (the same Eq. 1/3
  all-pairs construction, applied to tool-call sequences instead of text): **JS**
  (composition — *which* tools) and **GAK** (ordering — *what order*). Computed within one
  condition — no second run set, since under Assumption 1 the repetitions of a single
  setup are themselves a valid sample.
- A **run × run** matrix for every metric (each repetition compared against each other),
  plus an outlier (most-divergent) run.

**Cross-condition MMD** needs **≥2 conditions** — pass `--condition` more than once and
add permutation testing:
```bash
.venv/bin/python experiments/consistency-replay/consistency_statistics.py \
    --condition base=reports-consistency \
    --condition variant=reports-consistency-variant \
    --perm 200 --out reports-consistency/analysis_papers.json
```
This adds a two-sample MMD² + permutation p-value over trajectory distributions (JS and
GAK kernels) between each condition pair. With a single condition the MMD section is
simply omitted.

### Step 3 — Build the interactive viewer

```bash
.venv/bin/python experiments/consistency-replay/build_viewer.py reports-consistency \
    --analysis reports-consistency/analysis.json \
    --papers reports-consistency/analysis_papers.json \
    --out reports-consistency/consistency_viewer.html
open reports-consistency/consistency_viewer.html
```

Self-contained HTML (double-click to open, no server). **Takes a few minutes** — the
pairwise Levenshtein matrix is O(len²) in pure Python; the exporter caps compared
strings at 2000 chars to bound it.

The viewer is a **single merged view** (no tabs). Picking a trace in the sidebar shows a
**conversation timeline**: the **original task** (`messages[0]`) is pinned at the top, then
each model call in the agent loop is listed **in send order** (steps ordered by message
count — request → response, request → response …), one compact row per step with its
consistency pills. Steps where all runs agree carry a green **"✓ all N agree"** marker but
stay visible. Clicking a step **expands** it into the detailed comparison: the **request
that every run saw is pinned at the top** (sticky header: system head, tools available, the
message the model responded to), the N response cards render below, and clicking any two
cards (A, B) shows a word-level diff. A **checkbox row** above the cards lets you pick which
metrics appear on the A↔B comparison — content similarity, output Jaccard, exact match,
**TSS, AC, JS-kernel, GAK** — each computed **for that specific run pair at that specific
call position** (exported per group by `export_viewer_data.py`, no extra flag needed). The
back-link returns to the timeline. The **"All traces" overview** (sidebar top) keeps the
cross-trace dashboard + inconsistency-ranked list.

`--papers` is optional and feeds the **aggregate "Paper-grounded summary"** — the
U-statistic θ + trajectory θ cards, Hypothesis 1, the run×run heatmap, and cross-condition
MMD — which now lives in a collapsible section on the **"All traces" overview** (built from
the pre-generated `analysis_papers.json`; `build_viewer.py` does **not** run that analyzer
itself). Omit the flag (or if the file is absent) and the viewer still builds — the summary
section is simply absent, and the per-pair checkbox metrics still work (they come from the
export step, not `--papers`).

### Step 4 — Write findings

Summarize into `reports-consistency/FINDINGS.md`: headline consistency numbers, the
text-drift-vs-meaning-drift split, and which task types stay consistent vs fork. Use the
existing `FINDINGS.md` as the template if present.

## Interpreting the metrics

| Metric | Meaning | Good = |
|---|---|---|
| **Byte-identical %** | fraction of call positions where all runs returned the same output, **ignoring whitespace formatting** (tool args are whitespace-stripped; content collapses whitespace runs but keeps word boundaries) | high = deterministic |
| **Distinct outputs / call** | number of unique outputs out of N runs (same whitespace-insensitive signature; the viewer still displays the original text) | low |
| **Levenshtein similarity** | mean pairwise char similarity (1=identical) | high; a usable cheap proxy for semantic drift |
| **Jaccard** | pairwise word-set overlap | high |
| **finish_reason agreement** | do runs stop the same way | high |
| **tool name+argkeys / full-args agreement** | for tool turns: same tool, same args | high |
| **Semantic clusters** (judge) | distinct *meanings* per call (1 = all mean the same) | 1 |

Key relationship to report: semantic clusters scale with lexical distance — high
Levenshtein usually (not always) means same meaning; low Levenshtein means the model
likely made a genuinely different decision. Closed lookups tend to stay semantically
consistent even when wording drifts; open-ended agentic reasoning forks into multiple
real trajectories.

### Paper-grounded metrics (Step 2c → "Paper-grounded summary" in the overview; per-pair values also selectable on any A↔B comparison via the checkbox row)

| Metric | Meaning | Read as |
|---|---|---|
| **TSS** (Yagubyan) | mean pairwise similarity of the **tool-name sequences** | high = runs call the same tools in the same order |
| **AC** (Yagubyan) | mean pairwise **argument** agreement (step-aligned key/value overlap; 0 when tools differ) | high = same arguments, not just same tools |
| **Hypothesis 1** | does `E[TSS] ≫ E[AC]`? | **supported** = *structure is stable, arguments vary* — the arguments are where nondeterminism shows up |
| **θ + 95% CI** (Raj) | U-statistic: mean pairwise output agreement under the chosen kernel, with a confidence interval | higher θ = more consistent; the CI is the uncertainty from having only M instances |
| **θ_traj (JS)** (Raj, Eq. 1/3) | same U-statistic, but over tool-call **composition** (which tools each run used) | high = runs invoke the same *set* of tools |
| **θ_traj (GAK)** (Raj, Eq. 1/3) | same U-statistic, but over tool-call **ordering** (the sequence) | high = runs invoke tools in the same *order* |
| **run × run matrix** | every repetition scored against every other | spot a single divergent run (flagged as the outlier) vs. uniform drift |
| **MMD²** (Raj, ≥2 conditions) | two-sample distance between two conditions' **trajectory distributions**, with a permutation p-value | MMD²≈0 with high p ⇒ the conditions are **statistically indistinguishable** |

Typical headline for this endpoint: **TSS ≈ 1.0 but AC ≈ 0.46** — the agent almost always
picks the same tools in the same order, yet the *arguments* it passes vary run to run.
That is the Hypothesis-1 pattern (arXiv:2605.28840), and it lines up with the
text-drift-vs-meaning-drift split above: structure is deterministic, free-form content is
not. θ (arXiv:2605.10516) gives that a single number with error bars. The two **θ_traj**
values are the trajectory analogue and typically sit near **1.0** here (js≈0.999,
gak≈0.998) — the tool-call structure is essentially deterministic; they equal the
corresponding run-pair means but add a confidence interval (Eq. 1/3, computed within one
condition — no perturbation set needed, per Assumption 1's `x_mi = x_m0`).

## Adapting the experiment

- **Different model/endpoint:** pass the model as `run_experiment.sh`'s first argument and add
  its RITS slug to the `model_base_url()` table in that script — the config needs no edit.
  Update the judge endpoint in `analyze_consistency.py` (`judge_cfg`) and the probe URL in
  step 0. Confirm the RITS slug with `/v1/models` or a chat probe first (guessing slugs
  returns immediate 404).
- **Different benchmark:** `run_experiment.sh`'s second argument. One of `tau2_retail`,
  `appworld`, `swebench`, `tau2_airline`, `tau2_telecom`, `browsecompplus` — the dataset is a
  single corpus with a `benchmark` column, so this sets a filter, not a different dataset.
- **More/fewer sessions:** `N_SESSIONS`, or `SESSION_IDS` to name them exactly. Browse
  candidates with `list_sessions.py <BENCHMARK>` (prints a table; writes nothing).
- **More/fewer repetitions:** `run_experiment.sh`'s third positional argument (e.g. `20`).
- **A specific trace file instead of the HF dataset:** set `trace_files: [...]` and
  remove `hf_dataset_path` in the config's `otel_trace_replay` block.
- **Determinism floor:** to isolate numerical nondeterminism from sampling, force
  `temperature: 0` (server default is >0). The config doesn't expose a temperature knob
  today — would need adding, or set it at the server.

## Known caveats

**Recorded ids embed the replaying process's slot, so they differ between layouts.** Records
carry `session_id` / `event_id` and `replay_parsing.session_key` reads them directly (the
leading-message hash is only a fallback for old metrics that lack the field). A run-major run
of ten sessions produced `trace0_…`–`trace9_…`; a session-major cell replays one session in
slot 0 and always records `trace0_<dataset_session_id>`. **Join across layouts on the
`sessions/<session_id>` directory name** (the bare dataset id in both) or on `sessions.json`'s
`trace_id` — never on the recorded id.

**`concurrent_sessions: 1` changes what is measured.** Session-major runs put one session in
flight per process, so cross-session batching interference — present in every corpus collected
before 2026-08-12, including the numbers in `FINDINGS.md` — is no longer part of the
phenomenon. Note this when comparing new results against old.

**The harness wrote the resolved API key to disk in two places — redacted at the source since
2026-08-12.** Each run's `config.yaml` (`reportgen/base.py` `generate_config_report` →
`client/filestorage/local.py:37-39`) and the captured stdout log (`config/config.py` logs the
whole merged config at startup) both now pass through `config.redact_secrets`, which masks every
`headers` value plus any `api_key`/`token`/`secret`/`password`/`authorization`/`credential`
field. Header names survive; the live `Config` is unaffected.

**Runs from before that fix still hold plaintext keys** under git-ignored `reports-*/`. Sweep any
older corpus with
`grep -rl '<key>' reports-consistency | xargs sed -i '' 's/<key>/***REDACTED***/g'`,
and if a key ever reached those files, treat it as exposed and rotate it — scrubbing the copies
does not un-expose a key that was also committed or shared.
