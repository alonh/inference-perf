# Output Consistency via Trace Replay

**Measures: given byte-identical inputs, how repeatable are a model endpoint's outputs?**

Replays the same trace(s) N times against a live server with **output substitution
disabled**, then quantifies how much the repeated outputs differ and builds an
interactive side-by-side HTML viewer.

> **Why `disable_output_substitution: true` is the linchpin.** With it on, every
> repetition of a given (trace, call-position) is fed the *recorded* prior context, so
> all N runs send **identical input** at each call. Any difference in the outputs is then
> pure serving-stack nondeterminism (sampling + vLLM batching/numerics), not compounding
> cascade drift. Without the flag you'd be measuring input divergence instead.
>
> This feature lives on branch `feat/disable-output-substitution` (and this branch is
> based on it). On a branch without it, the config field is rejected.

## Contents

| File | Role |
|---|---|
| `config.yml` | Experiment config: model, exgentic HF dataset, `disable_output_substitution: true`, `num_sessions: 10`, `base_seed: 42`, `per_request: true`. Holds a **placeholder** API key. |
| `run_consistency.sh` | Runs `config.yml` N times (independent processes) → `reports-consistency/run_<i>/`. Injects `RITS_API_KEY` from env. |
| `analyze_consistency.py` | Groups results by (trace, identical-input); computes exact-match, Levenshtein/Jaccard, tool-call structure, and `--judge` LLM-judge semantic clusters. |
| `export_viewer_data.py` | Exports per-group texts + pairwise similarity matrices → `viewer_data.json`. |
| `viewer_template.html` | The viewer UI (HTML/CSS/JS); data injected at build time. |
| `build_viewer.py` | Runs the exporter, then injects JSON into the template → self-contained `consistency_viewer.html`. |
| `FINDINGS.md` | Written-up results from the reference 10×10 run. |
| `DESIGN.md` | Original design/rationale. |

Outputs are written to `reports-consistency/` at the repo root (git-ignored via `reports-*/`).

## Prerequisites

- Python venv at `.venv/` (repo standard).
- `export RITS_API_KEY=<your key>` — used by both the runner and the `--judge` analyzer step. No key is stored in the repo.
- The RITS gateway is flaky — probe before a long run (below). Run commands from the repo root.

## Run it end to end

```bash
export RITS_API_KEY=<your key>

# 0. Probe the endpoint is up (immediate 404 = wrong slug; timeout/502 = gateway down).
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

# 1. Run the config 10 times.
bash experiments/consistency-replay/run_consistency.sh 10

# 2. Analyze (offline metrics + LLM-judge semantic clustering).
.venv/bin/python experiments/consistency-replay/analyze_consistency.py reports-consistency \
    --judge --out reports-consistency/analysis.json

# 3. Build the interactive viewer (a few minutes — pairwise Levenshtein is O(n²)).
.venv/bin/python experiments/consistency-replay/build_viewer.py reports-consistency \
    --analysis reports-consistency/analysis.json \
    --out reports-consistency/consistency_viewer.html
open reports-consistency/consistency_viewer.html
```

## The viewer

- **Dashboard** — headline consistency numbers.
- **Sidebar** — traces sorted most-inconsistent first.
- **Call list** — each identical-input call position, tagged with metric pills.
- **Detail** — the shared input, the semantic-judge verdict, and a grid of all N
  versions colored by which distinct-output cluster each fell into. Click any two
  versions for a **word-level diff** with that pair's Levenshtein/Jaccard.

## Interpreting the metrics

| Metric | Meaning | Consistent = |
|---|---|---|
| Byte-identical % | call positions where all runs returned the exact same output | high |
| Distinct outputs / call | unique outputs out of N runs | low |
| Levenshtein similarity | mean pairwise char similarity (1 = identical) | high — cheap proxy for meaning drift |
| Jaccard | pairwise word-set overlap | high |
| finish_reason agreement | do runs stop the same way | high |
| tool name+argkeys / full-args agreement | tool turns: same tool, same args | high |
| Semantic clusters (judge) | distinct *meanings* per call (1 = all mean the same) | 1 |

## Adapting

- **Model/endpoint:** edit `server.base_url`, `server.model_name`, `static_model_name` in `config.yml`; update the judge URL/model in `analyze_consistency.py` (`judge_cfg`) and the probe URL above. Confirm any new RITS slug with a chat probe first (a wrong slug 404s immediately).
- **Trace count / repetitions:** change `num_sessions` in `config.yml`; pass a different N to `run_consistency.sh`.
- **Specific trace file(s):** set `trace_files: [...]` and remove `hf_dataset_path` in `config.yml`.
- **Determinism floor:** force `temperature: 0` (server default is >0) to separate numerical nondeterminism from sampling — not exposed in the config today; set server-side or add the knob.

## Known caveat

The trace-replay generator leaves `session_id` **null** on per-request metrics
(`inference_perf/datagen/replay_graph_session_datagen.py`: the built `SessionChatCompletionAPIData`
is passed the lazy stub's `None` instead of the extracted session id). The analyzer works
around this in `trace_key()` (hashes the request's leading message), which can split one
dataset session into two trace-ids — harmless, since groups still join on the exact
request hash. **A good first task for whoever picks this up:** fix that at the source and
drop the workaround.
