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
| `config.yml` | Experiment config: model, exgentic HF dataset, `disable_output_substitution: true`, `num_sessions: 10`, `base_seed: 41`, `per_request: true`. Holds a **placeholder** API key. |
| `run_consistency.sh` | Runs `config.yml` N times (independent processes) → `reports-consistency/run_<i>/`. Injects `RITS_API_KEY` from env. Run-major; unchanged. |
| `run_experiment.sh` | **The runner.** Takes model + benchmark on the command line and replays a named set of sessions N times each, one process per (session, repetition) cell → `reports-consistency/<benchmark>/<model>/<stamp>/sessions/<session_id>/run_<i>/`. `new` starts a fresh experiment, `continue` re-runs only the missing or unfinished cells. |
| `list_sessions.py` | Names the sessions of one benchmark — either the first N after the seeded shuffle (which reproduces what the load generator would have picked) or explicit ids — and writes the **append-only** `sessions.json` ledger. Also usable standalone to browse candidate sessions. |
| `migrate_to_session_major.py` | Converts an old run-major results directory into the session-major layout, into a **new** directory (the source is opened read-only). Records are copied byte-identical; API keys in copied configs are redacted. |
| `replay_parsing.py` | The general parsing layer, stdlib-only: reads a report off disk, turns one raw record into a parsed response, derives its grouping keys. Computes no metric and imports nothing from `compare/`, so a notebook can use it alone. |
| `compare/` | The comparison library: the canonical units equality is taken over (`signatures.py`), the per-pair metrics, the trace profile, and the trajectory kernels. Single source of truth for every metric definition. See `compare/README.md`. |
| `analyze_experiment.sh` | Re-runs analysis + viewer over an existing results directory, without replaying. |
| `analyze_consistency.py` | Groups results by (trace, identical-input); computes exact-match, Levenshtein/Jaccard, tool-call structure, and `--judge` LLM-judge semantic clusters. |
| `consistency_statistics.py` | Pairwise aggregation, **within a session**: per-session U-statistic θ (instance = session, sample unit = run) with a delete-one-run jackknife CI, the equally-weighted session mean with a t-CI over sessions, run × run matrices, session-trajectory MMD, and the paper-grounded hypothesis checks. |
| `extract_to_dataframe.py` | Flattens every run's records into one tidy row-per-call table (CSV / Parquet) for ad-hoc analysis. |
| `find_metric_witnesses.py` | Per metric, finds a concrete run-pair whose difference *only* that metric captures — evidence each metric earns its place. |
| `export_viewer_data.py` | Exports per-group texts + pairwise similarity matrices → `viewer_data.json`. |
| `viewer_template.html` | The viewer UI (HTML/CSS/JS); data injected at build time. |
| `build_viewer.py` | Runs the exporter, then injects JSON into the template → self-contained `consistency_viewer.html`. |
| `build_metric_viewer.py` | Renders the witness table into a standalone per-metric example viewer. |
| `reliability_metrics.py` | CLI over `compare.reliability`: the hal-harness Consistency (C) sub-metrics per session, aggregated across sessions. |
| `reliability_analysis.ipynb` | Notebook walkthrough of the same C metrics, with the per-session breakdown. |
| `test_replay_parsing.py` | Unit tests for the general parsing layer (the metric tests live in `compare/test_compare.py`). |
| `FINDINGS.md` | Written-up results from the reference 10×10 run. |
| `DESIGN.md` | Original design/rationale. |

Outputs are written to `reports-consistency/` at the repo root (git-ignored via `reports-*/`).

## Results layout

`run_experiment.sh` writes one directory per **session**, with the repetitions inside it:

```
reports-consistency/<benchmark>/<model-slug>/<stamp>/
├── sessions.json                        which sessions were replayed (append-only ledger)
├── experiment.json                      model, endpoint, tokenizer, benchmark, n_runs, timeout
├── coverage.json                        which (session, repetition) cells completed
├── sessions/<session_id>/run_<i>/        THE data: per_request_lifecycle_metrics.json, config.yaml
└── logs/<session_id>/run_<i>.log
```

The unit of work is one **cell** = one process replaying **one** session
(`concurrent_sessions: 1`, `num_sessions: 1`, with the session pinned by the replay filter).
Three things follow, and they are the reason for the layout:

- **Which sessions ran is data**, in `sessions.json`, rather than an emergent property of
  `load.base_seed` shuffling dataset rows. You can name a session and replay just it.
- **Adding a session later is a pure append** — one new `sessions/<id>/` tree plus one ledger
  line. Nothing already on disk moves or is rewritten. A run-major tree would need
  re-sharding on every addition.
- **A gateway failure costs one cell**, not all ten sessions of a run.

The cost, which matters when comparing new numbers against `FINDINGS.md` or any corpus
collected before 2026-08-12: with `concurrent_sessions: 1`, **cross-session batching
interference is no longer part of what is measured**. That is a change in the phenomenon
under study, not just plumbing.

> **The analysis scripts are mid-migration.** They still glob for run-major `<base>/run_*`
> (`replay_parsing.find_run_dirs`), so against this layout they find nothing. `run_experiment.sh`
> therefore *skips* them by default rather than emitting an empty-but-successful analysis;
> opt in with `RUN_ANALYSIS=1` once the finder understands `<base>/sessions/<id>/run_<i>`.
> Until then `analyze_consistency.py` on a session-major directory exits non-zero with
> "No run_* directories found".

### Choosing which sessions to replay

Two modes, both via `list_sessions.py` (called for you by `run_experiment.sh`):

```bash
# The first N sessions after the seeded shuffle. This REPLICATES the load generator's own
# selection, so seed 41 + limit 10 names the exact sessions the pre-session-major
# tau2_airline corpora replayed — new runs stay comparable to the old ones.
.venv/bin/python experiments/consistency-replay/list_sessions.py tau2_airline --limit 10 --seed 41

# Exactly these sessions, in this order. No shuffle.
.venv/bin/python experiments/consistency-replay/list_sessions.py tau2_airline \
    --session-id d3d3e0d1a8f8_20b42b16 --session-id 247b0c0650d3_f6b29499
```

With no `--out`/`--append` it only prints, so it doubles as a way to browse candidates.

### Adding a session to a finished experiment

One command. It appends to the ledger (preserving every existing entry **and its `slot`**),
then runs only the new session's cells:

```bash
SESSION_IDS=45eff42acfad_60e35976 \
  experiments/consistency-replay/run_experiment.sh continue \
  reports-consistency/tau2_airline/qwen-qwen3-vl-235b-a22b-instruct/<stamp> 10
```

`SESSION_IDS` both *restricts* the loop to those ids and *appends* any the ledger lacks.
No file under an existing session's tree is touched — verified by checksum and mtime.

### Converting an older run-major directory

```bash
.venv/bin/python experiments/consistency-replay/migrate_to_session_major.py \
    reports-consistency/tau2_airline/<model-slug>/<stamp> --dry-run   # inspect the grid first
```

The source is read-only; output goes to a new `<parent>/<today>-session-major/`. Records are
copied byte-identical, so they still carry the original ten-session shuffle slot in their
`session_id` (`trace<slot>_<id>`) whereas a fresh session-major run always records slot 0.
Join on the `sessions/<session_id>` directory name, or on `sessions.json`'s `trace_id`.
Per-run aggregates (`summary_*`, `stage_0_*`) describe a whole ten-session run and cannot be
attributed to one session, so they are preserved unsplit under `source_run_reports/`.

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

# 1. Replay 10 sessions x 10 repetitions, one process per cell.
#    Start with N_SESSIONS=1 and 2 reps to confirm the endpoint and the wiring first.
N_SESSIONS=1 experiments/consistency-replay/run_experiment.sh new \
    Qwen/Qwen3-VL-235B-A22B-Instruct tau2_airline 2
experiments/consistency-replay/run_experiment.sh new \
    Qwen/Qwen3-VL-235B-A22B-Instruct tau2_airline 10

# Interrupted? Re-runs only the cells that are missing or unfinished. Never re-queries HF.
experiments/consistency-replay/run_experiment.sh continue \
    reports-consistency/tau2_airline/qwen-qwen3-vl-235b-a22b-instruct/<stamp> 10

# 2 & 3. Analysis + viewer — see the mid-migration note under "Results layout" above; the
# runner skips them by default. On a run-major directory they work as before:
.venv/bin/python experiments/consistency-replay/analyze_consistency.py <RUN_MAJOR_DIR> \
    --judge --out <RUN_MAJOR_DIR>/analysis.json
.venv/bin/python experiments/consistency-replay/build_viewer.py <RUN_MAJOR_DIR> \
    --analysis <RUN_MAJOR_DIR>/analysis.json \
    --out <RUN_MAJOR_DIR>/consistency_viewer.html
```

Env knobs for the runner: `N_SESSIONS` (10), `SESSION_SEED` (41), `SESSION_IDS`,
`PARALLEL_REPS` (10 — repetitions of one session in flight at once, matching the
`concurrent_sessions: 10` the run-major config used, so the endpoint still sees ten requests
concurrently; set to 1 for a strictly sequential run), `REQUEST_TIMEOUT` (1200),
`RUN_ANALYSIS` (0).
`run_consistency.sh` is the older fixed-config runner and still writes the run-major
`reports-consistency/run_<i>/` layout.

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

- **Model/endpoint:** with `run_experiment.sh`, pass the model as an argument and add its RITS slug to the `model_base_url()` table in that script — `config.yml` needs no edit. (For `run_consistency.sh`, edit `server.base_url`, `server.model_name`, `static_model_name` in `config.yml`.) Update the judge URL/model in `analyze_consistency.py` (`judge_cfg`) and the probe URL above. Confirm any new RITS slug with a chat probe first (a wrong slug 404s immediately).
- **Session count / repetitions:** `N_SESSIONS` and the `N_RUNS` positional arg to `run_experiment.sh`. It overrides `load.stages` per cell, so `config.yml`'s `num_sessions: 10` applies only to `run_consistency.sh`.
- **Specific trace file(s):** set `trace_files: [...]` and remove `hf_dataset_path` in `config.yml`.
- **Determinism floor:** force `temperature: 0` (server default is >0) to separate numerical nondeterminism from sampling — not exposed in the config today; set server-side or add the knob.

## Known caveats

**Recorded ids differ between layouts.** Records carry `session_id` and `event_id`, and
`replay_parsing.session_key` reads them directly (hashing the request's leading message only
as a fallback for old metrics that never populated the field). The recorded id embeds the
session's slot in the process that replayed it: a run-major run of ten sessions produced
`trace0_…` through `trace9_…`, whereas a session-major cell replays one session in slot 0 and
so always records `trace0_<dataset_session_id>`. **Do not join on the recorded id across
layouts** — join on the `sessions/<session_id>` directory name, which is the bare dataset id
in both, or on `sessions.json`'s `trace_id`.

**The harness used to write the resolved API key to disk in two places — fixed 2026-08-12.**
Both paths now redact via `config.redact_secrets`:

1. **`<run>/config.yaml`** — `reportgen/base.py`'s `generate_config_report` dumps the whole
   resolved `Config`, headers included; appended at `base.py:991` and written as YAML by
   `client/filestorage/local.py:37-39`.
2. **`logs/<...>.log`** — `config/config.py` logs the entire merged config at startup, so the
   runner's captured stdout contained the header dict too.

Every value under `headers` is masked regardless of its name (auth header names are
gateway-specific, so name-matching alone fails open), as is any field named like
`api_key` / `token` / `secret` / `password` / `authorization` / `credential`. Header **names**
survive, so the saved config still records which headers a run sent. `None` is left alone, so
`api_key: null` still reads as unset. The live `Config` is untouched — only what reaches disk
and logs is redacted. Covered by `tests/required/config/test_secret_redaction.py`, which
asserts through both real write paths rather than the helper alone.

**Corpora collected before that fix still contain plaintext keys** under git-ignored
`reports-*/`. Nothing was ever committed from there, but the files are on disk. Sweep them:

```bash
grep -rl '<the key>' reports-consistency | xargs sed -i '' 's/<the key>/***REDACTED***/g'
```

`migrate_to_session_major.py` also redacts on copy, which covers path 1 for migrated corpora.

**`choices[0]` vs all choices.** `replay_parsing.parse_response` reads `choices[0]`
(`replay_parsing.py:234`) while `inference_perf/apis/chat.py:575` concatenates every choice.
With `n=1` — every run so far — these agree; with `n>1` they would not.
