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
| `experiments/consistency-replay/run_consistency.sh` | Runs the config N times, one report dir per run; injects `$RITS_API_KEY` |
| `experiments/consistency-replay/analyze_consistency.py` | Groups by (trace, identical-input) and computes metrics |
| `experiments/consistency-replay/consistency_statistics.py` | Paper-grounded metrics: Yagubyan TSS/AC + Hypothesis 1; Raj U-statistic θ+CI; run×run matrices; cross-condition MMD |
| `experiments/consistency-replay/export_viewer_data.py` | Exports per-group data (texts + pairwise matrices) for the viewer |
| `experiments/consistency-replay/viewer_template.html` | The viewer UI (data injected at build time) |
| `experiments/consistency-replay/build_viewer.py` | Export + inject → single self-contained HTML |
| `experiments/consistency-replay/README.md` | Human-facing quick start |

Results land in `reports-consistency/` (run dirs, `analysis.json`, `consistency_viewer.html`).

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

### Step 1 — Run N times (default 10)

```bash
rm -rf reports-consistency/run_* reports-consistency/logs
bash experiments/consistency-replay/run_consistency.sh 10
```

Each iteration is an **independent process** writing `reports-consistency/run_<i>/`.
The config fixes `base_seed: 42` and uses `num_sessions: 10`, so every run replays the
**same** first 10 traces. Per-run record counts vary (a 504 on an early turn cancels the
rest of that session chain) — this is expected and handled by the analyzer.

Long-running; run in the background and monitor `reports-consistency/runner.log` for
`Run i/10` / `OK (rc=0)` / `Done. ok= fail=`.

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

- **Different model/endpoint:** edit `server.base_url`, `server.model_name`, and
  `static_model_name` in the config; update the judge endpoint in
  `analyze_consistency.py` (`judge_cfg`) and the probe URL in step 0. Confirm the RITS
  slug with `/v1/models` or a chat probe first (guessing slugs returns immediate 404).
- **More/fewer traces:** change `num_sessions` in the config's load stage.
- **More/fewer repetitions:** pass a different N to `run_consistency.sh` (e.g. `20`).
- **A specific trace file instead of the HF dataset:** set `trace_files: [...]` and
  remove `hf_dataset_path` in the config's `otel_trace_replay` block.
- **Determinism floor:** to isolate numerical nondeterminism from sampling, force
  `temperature: 0` (server default is >0). The config doesn't expose a temperature knob
  today — would need adding, or set it at the server.

## Known caveat

The trace-replay generator currently leaves `session_id` **null** on per-request metrics
(in `inference_perf/datagen/replay_graph_session_datagen.py`, the built
`SessionChatCompletionAPIData` is passed the lazy stub's `None` instead of the locally
extracted session id). The analyzer works around this via `trace_key()` (hash of the
request's leading message), which can split one dataset session into two trace-ids —
harmless, since groups still join on the exact request hash.
