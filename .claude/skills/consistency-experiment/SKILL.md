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

- Python venv at `.venv/` — use `/Users/alonhal/PycharmProjects/inference-perf/.venv/bin/python`.
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
| `experiments/consistency-replay/export_viewer_data.py` | Exports per-group data (texts + pairwise matrices) for the viewer |
| `experiments/consistency-replay/viewer_template.html` | The viewer UI (data injected at build time) |
| `experiments/consistency-replay/build_viewer.py` | Export + inject → single self-contained HTML |
| `experiments/consistency-replay/README.md` | Human-facing quick start |

Results land in `reports-consistency/` (run dirs, `analysis.json`, `consistency_viewer.html`).

## Procedure

Create a todo per step and work them in order.

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

### Step 3 — Build the interactive viewer

```bash
.venv/bin/python experiments/consistency-replay/build_viewer.py reports-consistency \
    --analysis reports-consistency/analysis.json \
    --out reports-consistency/consistency_viewer.html
open reports-consistency/consistency_viewer.html
```

Self-contained HTML (double-click to open, no server). **Takes a few minutes** — the
pairwise Levenshtein matrix is O(len²) in pure Python; the exporter caps compared
strings at 2000 chars to bound it.

### Step 4 — Write findings

Summarize into `reports-consistency/FINDINGS.md`: headline consistency numbers, the
text-drift-vs-meaning-drift split, and which task types stay consistent vs fork. Use the
existing `FINDINGS.md` as the template if present.

## Interpreting the metrics

| Metric | Meaning | Good = |
|---|---|---|
| **Byte-identical %** | fraction of call positions where all runs returned the exact same output | high = deterministic |
| **Distinct outputs / call** | number of unique outputs out of N runs | low |
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
(`replay_graph_session_datagen.py:1388` passes the lazy stub's `None` instead of the
extracted session id). The analyzer works around this via `trace_key()` (hash of the
request's leading message), which can split one dataset session into two trace-ids —
harmless, since groups still join on the exact request hash.
