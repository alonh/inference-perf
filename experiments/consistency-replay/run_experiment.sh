#!/bin/bash

# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run the output-consistency experiment for a chosen MODEL + BENCHMARK: replay a fixed set
# of recorded sessions N times each, writing to a NEW timestamped results directory so
# previous experiments are never overwritten.
#
# The unit of work is one (session, repetition) CELL — one process replaying ONE session:
#
#   reports-consistency/<benchmark>/<model>/<stamp>/
#   ├── sessions.json                  append-only ledger of exactly which sessions ran
#   ├── experiment.json                model, endpoint, tokenizer, benchmark, n_runs, timeout
#   ├── coverage.json                  which cells completed
#   ├── sessions/<session_id>/run_<i>/ THE data: per_request_lifecycle_metrics.json, config.yaml
#   └── logs/<session_id>/run_<i>.log
#
# Why session-major rather than one process replaying all sessions at once (what this script
# used to do, and what every corpus collected before 2026-08-12 contains):
#
#   * The sessions being replayed are now DATA (sessions.json), not an emergent property of
#     load.base_seed shuffling dataset rows. You can name a session and replay it.
#   * Adding a session to a finished experiment is a pure append — one new sessions/<id>/
#     tree and one new ledger line. Nothing already on disk moves. That is the whole reason
#     for the layout; a run-major tree would need re-sharding on every addition.
#   * A gateway failure costs one cell, not all ten sessions in the run.
#
# The cost, worth remembering when comparing new numbers against older ones or FINDINGS.md:
# with concurrent_sessions=1, cross-session batching interference is no longer part of what
# is being measured. That is a change in the phenomenon under study, not just plumbing.
#
# This script runs in one of two MODES, given as the first argument:
#
#   new       Start a fresh experiment in a new timestamped results directory.
#   continue  Resume an existing results directory, re-running only the cells that are
#             missing or were left incomplete (see "Continue mode" below).
#
# Usage:
#   export RITS_API_KEY=<your key>
#   experiments/consistency-replay/run_experiment.sh new <MODEL> <BENCHMARK> [N_RUNS] [TOKENIZER_NAME]
#   experiments/consistency-replay/run_experiment.sh continue <OUT_BASE> [N_RUNS]
#
# new mode:
#   MODEL        Model id the ENDPOINT accepts (the HTTP `model` field), e.g.
#                Qwen/Qwen3-VL-235B-A22B-Instruct. Must have an entry in the
#                model_base_url lookup table below.
#   BENCHMARK    Which benchmark's traces to replay. The dataset (config.yml's
#                hf_dataset_path, Exgentic/agent-llm-traces) is a SINGLE corpus with a
#                `benchmark` column; this arg does NOT swap datasets — it selects rows
#                where x['benchmark'] == <BENCHMARK>. One of:
#                  tau2_retail  appworld  swebench  tau2_airline  tau2_telecom  browsecompplus
#   N_RUNS       Repetitions PER SESSION (default 10). Total cells = sessions x N_RUNS.
#   TOKENIZER_NAME  Optional HF model id for the tokenizer only. Use when the endpoint's
#                model name isn't resolvable on HuggingFace (e.g. an -FP8-A100 variant):
#                pass the HF-resolvable base name here. Defaults to MODEL.
#
# continue mode:
#   OUT_BASE     An existing results directory previously produced by `new` mode, e.g.
#                reports-consistency/tau2_airline/<model-slug>/<timestamp>. Model, endpoint,
#                tokenizer, benchmark and request timeout are recovered from experiment.json
#                (falling back to a completed cell's saved config.yaml), and the session set
#                from sessions.json — so none of them are re-specified, and HuggingFace is
#                NOT re-queried.
#   N_RUNS       Repetitions per session the experiment should contain (default 10). A cell
#                is COMPLETE iff sessions/<id>/run_<i>/per_request_lifecycle_metrics.json
#                exists. Two kinds of missing cells are both re-run from scratch:
#                  (1) run_<i>/ does not exist yet (never started); and
#                  (2) run_<i>/ exists but its per_request_lifecycle_metrics.json is absent
#                      (started but did not finish).
#                Any partial run_<i>/ directory is wiped before being re-executed.
#                Already-complete cells are left untouched.
#
# Env:
#   N_SESSIONS       How many sessions to replay, taken as the first N after the seeded
#                    shuffle (default 10). new mode only.
#   SESSION_SEED     Shuffle seed for that selection (default 41). This replicates the load
#                    generator's own session ordering, so the default reproduces the exact
#                    session set of the pre-session-major corpora. new mode only.
#   SESSION_IDS      Space- or comma-separated session ids to replay, overriding
#                    N_SESSIONS/SESSION_SEED. In continue mode this both RESTRICTS the loop
#                    to those ids and APPENDS any that sessions.json does not yet list — so
#                    "add session X to this experiment" is one command.
#   PARALLEL_REPS    How many repetitions of one session to run concurrently (default 10,
#                    matching the concurrent_sessions: 10 the run-major config used, so the
#                    endpoint sees the same number of requests in flight as before — ten,
#                    just ten repetitions of one session rather than ten different sessions).
#                    Sessions are always processed one at a time; only repetitions fan out.
#                    Set to 1 for a strictly sequential run (one request in flight).
#   REQUEST_TIMEOUT  Per-request timeout in seconds (default 1200). Raise for slow
#                    "thinking" models that can exceed aiohttp's 300s default; harmless for
#                    fast models. In continue mode the recovered value is used unless
#                    REQUEST_TIMEOUT is set explicitly in the environment.
#   RUN_ANALYSIS     Run the analyzer + viewer after the replay. On by default (=1); set to 0
#                    to replay only — the analysis can always be run later against the same
#                    directory with analyze_experiment.sh. The judge pass makes network calls
#                    and needs RITS_API_KEY, so RUN_ANALYSIS=0 is the offline-only path.
#
# Examples:
#   experiments/consistency-replay/run_experiment.sh new \
#       Qwen/Qwen3-VL-235B-A22B-Instruct tau2_retail 10
#   # One session, two repetitions — the cheapest end-to-end check:
#   N_SESSIONS=1 experiments/consistency-replay/run_experiment.sh new \
#       Qwen/Qwen3-VL-235B-A22B-Instruct tau2_airline 2
#   # Endpoint name differs from the HF tokenizer name:
#   experiments/consistency-replay/run_experiment.sh new \
#       Qwen/Qwen3.5-397B-A17B-FP8-A100 tau2_airline 10 Qwen/Qwen3.5-397B-A17B
#   # Resume an interrupted experiment (only incomplete cells re-run):
#   experiments/consistency-replay/run_experiment.sh continue \
#       reports-consistency/tau2_airline/qwen-qwen3-5-397b-a17b-fp8-a100/20260804-154811 10
#   # Add one more session to that finished experiment:
#   SESSION_IDS=45eff42acfad_60e35976 experiments/consistency-replay/run_experiment.sh \
#       continue reports-consistency/tau2_airline/<model-slug>/<stamp> 10

set -u

# macOS fork-safety: this repo forces multiprocessing 'fork', and mp.Manager() makes the
# parent multi-threaded before workers fork. Without this, forked children abort the moment
# they touch Network.framework (getaddrinfo) on this machine's IPv6-first resolver. This env
# var tells the ObjC runtime not to abort in that case. Harmless on Linux/other machines.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# ------------------------------------------------------------------------------
# Model -> RITS endpoint base_url lookup.
# The RITS base_url needs an endpoint slug that doesn't always follow from the model
# name, so keep an explicit table. Add a line per model you replay against.
# ------------------------------------------------------------------------------
RITS_BASE="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com"
model_base_url() {
  case "$1" in
    "Qwen/Qwen3-VL-235B-A22B-Instruct")
      echo "$RITS_BASE/qwen3-vl-235b-a22b-instruct" ;;
    "Qwen/Qwen3-VL-235B-A22B-Thinking")
      echo "$RITS_BASE/qwen3-vl-235b-a22b-thinking" ;;
    "Qwen/Qwen3.5-397B-A17B-FP8-A100")
      echo "$RITS_BASE/qwen3-5-397b-a17b-fp8-a100" ;;
    # Add more models here, e.g.:
    # "meta-llama/Llama-3.3-70B-Instruct")
    #   echo "$RITS_BASE/llama-3-3-70b-instruct" ;;
    *)
      return 1 ;;
  esac
}

# ------------------------------------------------------------------------------
# Paths (needed early so continue mode can read an existing experiment.json)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CONFIG="experiments/consistency-replay/config.yml"
PYTHON="$REPO_ROOT/.venv/bin/python"
LIST_SESSIONS="experiments/consistency-replay/list_sessions.py"

# The output file whose presence marks a CELL as COMPLETE. This is exactly the file the
# analysis layer loads per cell, so "complete" here means "analyzable".
DONE_MARKER="per_request_lifecycle_metrics.json"

# Model is slugified for the results path (it contains '/').
slugify() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed 's/-\{1,\}/-/g; s/^-//; s/-$//'; }

usage() {
  echo "Usage:" >&2
  echo "  $0 new <MODEL> <BENCHMARK> [N_RUNS] [TOKENIZER_NAME]" >&2
  echo "  $0 continue <OUT_BASE> [N_RUNS]" >&2
  exit 2
}

# A session id is interpolated into the filter lambda that the load generator eval()s
# (_compile_filter, otel_trace_replay_datagen.py:208-236). Anything outside this character
# set is rejected rather than escaped — no legitimate dataset id needs a quote or a paren.
# list_sessions.py applies the same rule; this is the second gate, for ids arriving via
# SESSION_IDS or a hand-edited sessions.json without passing through it.
validate_session_id() {
  case "$1" in
    "" ) return 1 ;;
    *[!A-Za-z0-9_.:-]* ) return 1 ;;
    * ) return 0 ;;
  esac
}

# SESSION_IDS accepts commas or whitespace; normalize to whitespace-separated.
SESSION_IDS_RAW="${SESSION_IDS:-}"
SESSION_IDS_LIST="$(echo "$SESSION_IDS_RAW" | tr ',' ' ' | tr -s ' ' | sed 's/^ //; s/ $//')"

N_SESSIONS="${N_SESSIONS:-10}"
SESSION_SEED="${SESSION_SEED:-41}"
PARALLEL_REPS="${PARALLEL_REPS:-10}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

case "$N_SESSIONS" in ''|*[!0-9]*) echo "ERROR: N_SESSIONS must be a positive integer (got '$N_SESSIONS')." >&2; exit 2 ;; esac
case "$PARALLEL_REPS" in ''|*[!0-9]*) echo "ERROR: PARALLEL_REPS must be a positive integer (got '$PARALLEL_REPS')." >&2; exit 2 ;; esac
[ "$N_SESSIONS" -ge 1 ] || { echo "ERROR: N_SESSIONS must be >= 1." >&2; exit 2; }
[ "$PARALLEL_REPS" -ge 1 ] || { echo "ERROR: PARALLEL_REPS must be >= 1." >&2; exit 2; }

for sid in $SESSION_IDS_LIST; do
  if ! validate_session_id "$sid"; then
    echo "ERROR: SESSION_IDS entry '$sid' contains characters not allowed in a replay" >&2
    echo "       filter (expected only A-Z a-z 0-9 _ . : -)." >&2
    exit 2
  fi
done

# ------------------------------------------------------------------------------
# Args — dispatch on MODE. The mode is an OPTIONAL leading keyword; when the first
# arg is neither 'new' nor 'continue' it defaults to 'new' and is treated as <MODEL>,
# so the original `... <MODEL> <BENCHMARK> ...` invocation keeps working unchanged.
# ------------------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
  usage
fi
case "$1" in
  new)      MODE="new";      shift ;;
  continue) MODE="continue"; shift ;;
  *)        MODE="new" ;;   # no explicit mode keyword -> default to a fresh run
esac

if [ "$MODE" = "new" ]; then
  if [ "$#" -lt 2 ]; then
    echo "ERROR: 'new' mode needs <MODEL> <BENCHMARK>." >&2
    usage
  fi
  MODEL="$1"
  BENCHMARK="$2"
  N_RUNS="${3:-10}"
  # Optional 4th arg: the HuggingFace model id used ONLY for the tokenizer. Some endpoints
  # expose a model under a name that isn't on HuggingFace (e.g. an -FP8-A100 quantized
  # variant), so the name the endpoint accepts and the name AutoTokenizer.from_pretrained
  # can resolve differ. When they differ, pass the HF-resolvable name here; the endpoint
  # still receives $MODEL. Defaults to $MODEL, in which case one name serves both.
  TOKENIZER_NAME="${4:-$MODEL}"

  # Per-request timeout (seconds) passed to load.request_timeout. Without it, aiohttp's
  # 300s default applies, which slow "thinking" models can exceed under concurrency. Long
  # default is harmless for fast models (they finish well under it). Override via env:
  #   REQUEST_TIMEOUT=600 experiments/consistency-replay/run_experiment.sh new ...
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"

  BASE_URL="$(model_base_url "$MODEL")"
  if [ -z "$BASE_URL" ]; then
    echo "ERROR: no endpoint URL known for model '$MODEL'." >&2
    echo "       Add it to the model_base_url() lookup table in this script." >&2
    exit 2
  fi

  # Known benchmark values in the `benchmark` column of Exgentic/agent-llm-traces.
  KNOWN_BENCHMARKS="tau2_retail appworld swebench tau2_airline tau2_telecom browsecompplus"
  case " $KNOWN_BENCHMARKS " in
    *" $BENCHMARK "*) ;;
    *)
      echo "ERROR: unknown benchmark '$BENCHMARK'." >&2
      echo "       Expected one of: $KNOWN_BENCHMARKS" >&2
      exit 2 ;;
  esac

  # Results layout:  reports-consistency/<benchmark>/<model>/<timestamp>/
  # The benchmark and model folders are created once and reused across experiments; each
  # individual experiment gets its own timestamp subdir so nothing is ever overwritten.
  MODEL_SLUG="$(slugify "$MODEL")"
  # Timestamp is captured once so every cell + the analysis + viewer share one directory.
  STAMP="$(date +%Y%m%d-%H%M%S)"
  OUT_BASE="reports-consistency/$BENCHMARK/$MODEL_SLUG/$STAMP"

elif [ "$MODE" = "continue" ]; then
  if [ "$#" -lt 1 ]; then
    echo "ERROR: 'continue' mode needs <OUT_BASE>." >&2
    usage
  fi
  OUT_BASE="${1%/}"   # strip any trailing slash so paths join cleanly
  N_RUNS="${2:-10}"
  if [ ! -d "$OUT_BASE" ]; then
    echo "ERROR: OUT_BASE '$OUT_BASE' does not exist or is not a directory." >&2
    exit 2
  fi

  # Recover model/endpoint/tokenizer/timeout. experiment.json is written by new mode and is
  # the authoritative record; a saved config.yaml from any COMPLETE cell is the fallback for
  # directories that predate it. A complete cell's config is trustworthy because
  # inference_perf writes its resolved config only after the run produced output.
  #
  # The saved config's `filter` is deliberately NOT recovered: under this layout it names a
  # single session, so reusing it would pin the whole experiment to whichever cell happened
  # to be read first. The session set comes from sessions.json instead.
  SRC_CONFIG=""
  if [ ! -f "$OUT_BASE/experiment.json" ]; then
    for cand in "$OUT_BASE"/sessions/*/run_*/config.yaml; do
      [ -f "$cand" ] || continue
      if [ -f "$(dirname "$cand")/$DONE_MARKER" ]; then
        SRC_CONFIG="$cand"
        break
      fi
    done
    if [ -z "$SRC_CONFIG" ]; then
      echo "ERROR: '$OUT_BASE' has neither experiment.json nor a completed cell with a" >&2
      echo "       config.yaml, so the experiment settings cannot be recovered." >&2
      echo "       If this is a pre-2026-08-12 run-major directory, convert it first:" >&2
      echo "         $PYTHON experiments/consistency-replay/migrate_to_session_major.py $OUT_BASE" >&2
      exit 2
    fi
    echo "No experiment.json; recovering settings from: $SRC_CONFIG"
  fi

  # Emit shell-safe KEY=VALUE lines from whichever source is available, and eval them.
  SETTINGS="$(
    "$PYTHON" - "$OUT_BASE/experiment.json" "$SRC_CONFIG" <<'PY'
import json, os, shlex, sys

exp_path, cfg_path = sys.argv[1], sys.argv[2]
vals = {"MODEL": "", "BASE_URL": "", "TOKENIZER_NAME": "", "CFG_TIMEOUT": "", "BENCHMARK": ""}

if os.path.exists(exp_path):
    with open(exp_path) as f:
        exp = json.load(f)
    vals.update({
        "MODEL":          exp.get("model") or "",
        "BASE_URL":       exp.get("base_url") or "",
        "TOKENIZER_NAME": exp.get("tokenizer") or exp.get("model") or "",
        "CFG_TIMEOUT":    exp.get("request_timeout") or "",
        "BENCHMARK":      exp.get("benchmark") or "",
    })
elif cfg_path:
    import yaml
    with open(cfg_path) as f:
        c = yaml.safe_load(f) or {}
    tok = c.get("tokenizer") if isinstance(c.get("tokenizer"), dict) else {}
    server = c.get("server") or {}
    vals.update({
        "MODEL":          server.get("model_name") or "",
        "BASE_URL":       server.get("base_url") or "",
        "TOKENIZER_NAME": (tok or {}).get("pretrained_model_name_or_path")
                          or server.get("model_name") or "",
        "CFG_TIMEOUT":    (c.get("load") or {}).get("request_timeout") or "",
    })

for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
  )"
  if [ $? -ne 0 ] || [ -z "$SETTINGS" ]; then
    echo "ERROR: failed to recover settings for '$OUT_BASE'." >&2
    exit 2
  fi
  RECOVERED_BENCHMARK=""
  eval "$SETTINGS"
  RECOVERED_BENCHMARK="$BENCHMARK"

  # Timeout: honor an explicit env override, else the recovered value, else new mode's default.
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-${CFG_TIMEOUT:-1200}}"

  if [ -z "$MODEL" ] || [ -z "$BASE_URL" ]; then
    echo "ERROR: recovered settings are incomplete." >&2
    echo "       MODEL='$MODEL' BASE_URL='$BASE_URL'" >&2
    exit 2
  fi
  # Benchmark: from experiment.json when present, else from OUT_BASE's path layout
  # (reports-consistency/<benchmark>/<model>/<stamp>).
  if [ -z "$RECOVERED_BENCHMARK" ]; then
    BENCHMARK="$(basename "$(dirname "$(dirname "$OUT_BASE")")")"
  fi

else
  echo "ERROR: unknown mode '$MODE' (expected 'new' or 'continue')." >&2
  usage
fi

if [ -z "${RITS_API_KEY:-}" ]; then
  echo "ERROR: RITS_API_KEY is not set. Run:  export RITS_API_KEY=<your key>" >&2
  exit 1
fi

case "$N_RUNS" in ''|*[!0-9]*) echo "ERROR: N_RUNS must be a positive integer (got '$N_RUNS')." >&2; exit 2 ;; esac
[ "$N_RUNS" -ge 1 ] || { echo "ERROR: N_RUNS must be >= 1." >&2; exit 2; }
case "$REQUEST_TIMEOUT" in ''|*[!0-9]*) echo "ERROR: REQUEST_TIMEOUT must be a positive integer of seconds (got '$REQUEST_TIMEOUT')." >&2; exit 2 ;; esac

LEDGER="$OUT_BASE/sessions.json"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------------------
# Step 0 — Establish the session set, and record it as data.
#
# new mode:      build the ledger from the dataset (shuffle+limit, or explicit ids).
# continue mode: read the existing ledger; never re-query HuggingFace. SESSION_IDS both
#                restricts this invocation's loop AND appends ids the ledger lacks, which
#                is what makes "add a session" a single command against a finished dir.
# ------------------------------------------------------------------------------
if [ "$MODE" = "new" ]; then
  echo "--- Selecting sessions -> $LEDGER ---"
  if [ -n "$SESSION_IDS_LIST" ]; then
    set -- "$BENCHMARK"
    for sid in $SESSION_IDS_LIST; do
      set -- "$@" --session-id "$sid"
    done
    "$PYTHON" "$LIST_SESSIONS" "$@" --out "$LEDGER" || exit 1
  else
    "$PYTHON" "$LIST_SESSIONS" "$BENCHMARK" \
      --limit "$N_SESSIONS" --seed "$SESSION_SEED" --out "$LEDGER" || exit 1
  fi
else
  if [ ! -f "$LEDGER" ]; then
    echo "ERROR: '$LEDGER' is missing, so the session set is unknown." >&2
    echo "       A session-major experiment records what it replayed; without the ledger" >&2
    echo "       there is nothing to continue. Re-run in 'new' mode." >&2
    exit 2
  fi
  if [ -n "$SESSION_IDS_LIST" ]; then
    # Append only the ids the ledger does not already have. --append preserves every
    # existing entry and its slot verbatim, and is a no-op for ids already present.
    set -- "$BENCHMARK"
    for sid in $SESSION_IDS_LIST; do
      set -- "$@" --session-id "$sid"
    done
    echo "--- Ensuring requested sessions are in $LEDGER ---"
    "$PYTHON" "$LIST_SESSIONS" "$@" --append "$LEDGER" || exit 1
  fi
fi

# The ids to loop over: the ledger's, restricted to SESSION_IDS when that is set.
SESSIONS="$(
  "$PYTHON" - "$LEDGER" "$SESSION_IDS_LIST" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    ledger = json.load(f)
ids = [str(e["session_id"]) for e in (ledger.get("sessions") or []) if e.get("session_id")]
wanted = sys.argv[2].split()
if wanted:
    known = set(ids)
    missing = [w for w in wanted if w not in known]
    if missing:
        print("MISSING:" + ",".join(missing), file=sys.stderr)
        sys.exit(3)
    ids = [i for i in ids if i in set(wanted)]
if not ids:
    print("MISSING:<ledger has no sessions>", file=sys.stderr)
    sys.exit(3)
print("\n".join(ids))
PY
)"
if [ $? -ne 0 ] || [ -z "$SESSIONS" ]; then
  echo "ERROR: could not determine the session list from $LEDGER." >&2
  exit 2
fi

N_SESSIONS_ACTUAL=$(echo "$SESSIONS" | wc -l | tr -d ' ')

# Second gate on ids read back from the ledger, which may have been hand-edited since it
# was written. They are about to be interpolated into an eval()-ed lambda.
for sid in $SESSIONS; do
  if ! validate_session_id "$sid"; then
    echo "ERROR: session id '$sid' in $LEDGER contains characters not allowed in a" >&2
    echo "       replay filter (expected only A-Z a-z 0-9 _ . : -)." >&2
    exit 2
  fi
done

# ------------------------------------------------------------------------------
# experiment.json — the settings, so `continue` never has to guess and never has to
# re-read the dataset. Written in new mode; refreshed in continue mode only if absent.
# ------------------------------------------------------------------------------
if [ "$MODE" = "new" ] || [ ! -f "$OUT_BASE/experiment.json" ]; then
  # Values arrive as argv, not interpolated into the source: a model name or URL carrying a
  # quote would otherwise be a Python syntax error rather than a string.
  "$PYTHON" - "$OUT_BASE/experiment.json" \
      "$BENCHMARK" "$MODEL" "$BASE_URL" "$TOKENIZER_NAME" "$N_RUNS" "$REQUEST_TIMEOUT" <<'PY'
import json, sys

out, benchmark, model, base_url, tokenizer, n_runs, timeout = sys.argv[1:8]
with open(out, "w") as f:
    json.dump({
        "benchmark": benchmark,
        "model": model,
        "base_url": base_url,
        "tokenizer": tokenizer,
        "n_runs": int(n_runs),
        "request_timeout": int(timeout),
        "layout": "session-major",
        "concurrent_sessions": 1,
        "note": (
            "One process per (session, repetition) cell: concurrent_sessions=1, "
            "num_sessions=1, with the session pinned by the replay filter. Session set is "
            "in sessions.json. Records carry session_id 'trace0_<dataset_session_id>' "
            "because each process replays a single session in slot 0."
        ),
    }, f, indent=2)
    f.write("\n")
PY
fi

echo "=================================================="
echo "Consistency experiment (session-major)"
echo "Mode:       $MODE"
echo "Model:      $MODEL"
echo "Endpoint:   $BASE_URL"
echo "Tokenizer:  $TOKENIZER_NAME"
echo "Benchmark:  $BENCHMARK"
echo "Sessions:   $N_SESSIONS_ACTUAL  (ledger: $LEDGER)"
echo "Reps each:  $N_RUNS            (parallel: $PARALLEL_REPS)"
echo "Cells:      $((N_SESSIONS_ACTUAL * N_RUNS))"
echo "Timeout:    ${REQUEST_TIMEOUT}s"
echo "Config:     $CONFIG"
echo "Output:     $OUT_BASE/"
echo "=================================================="

# ------------------------------------------------------------------------------
# Step 1 — sessions outer, repetitions inner. One process per cell.
#
# Per-cell overrides on top of config.yml:
#   * filter pins ONE session, so the process replays exactly that episode;
#   * stages is replaced wholesale (--load.stages is JSON-typed and deep_merge replaces
#     lists), dropping concurrent_sessions AND num_sessions to 1. Both are required:
#     inference_perf/loadgen/load_generator.py:1000 RAISES when the stages request more
#     sessions than the corpus holds, so leaving num_sessions at 10 against a one-row
#     filter is a hard failure, not a clamp (the clamp at :441 comes too late);
#   * num_workers drops to 1. load_generator.py:953 spawns num_workers processes
#     unconditionally and session replay never clamps them to the session count
#     (:429 takes num_workers as-is), so a cell would otherwise fork cpu_count() workers to
#     serve its single session. At PARALLEL_REPS=10 that is ~110 processes for 10 requests;
#     with 1 worker per cell the fan-out lives where it belongs — in the cell count;
#   * storage path is the cell's own directory.
#
# Net effect at the defaults: 10 requests in flight, the same as the run-major
# concurrent_sessions: 10 — but each is a repetition of ONE session rather than a different
# session, which is what removes cross-session batching from the measurement.
# ------------------------------------------------------------------------------
STAGES_OVERRIDE='[{"concurrent_sessions":1,"num_sessions":1}]'

ok=0
fail=0
skipped=0

# Run one cell in the foreground. Called directly when PARALLEL_REPS=1 and as a background
# job otherwise, so the command itself lives in exactly one place.
run_cell() {
  local sid="$1" cell_dir="$2" log_file="$3"
  # RITS_API_KEY is injected via CLI override (the config holds only a placeholder).
  # --api.headers takes the WHOLE headers dict as a JSON string (there is no dotted
  # --api.headers.<key> flag), so pass the full object.
  "$PYTHON" -m inference_perf.main \
    --config "$CONFIG" \
    --api.headers "{\"RITS_API_KEY\":\"$RITS_API_KEY\"}" \
    --server.base_url "$BASE_URL" \
    --server.model_name "$MODEL" \
    --tokenizer.pretrained_model_name_or_path "$TOKENIZER_NAME" \
    --load.request_timeout "$REQUEST_TIMEOUT" \
    --load.stages "$STAGES_OVERRIDE" \
    --load.num_workers 1 \
    --data.otel_trace_replay.static_model_name "$MODEL" \
    --data.otel_trace_replay.filter \
      "lambda x: x['benchmark'] == '$BENCHMARK' and x['session_id'] == '$sid'" \
    --storage.local_storage.path "$cell_dir" \
    >"$log_file" 2>&1
}

s_idx=0
for sid in $SESSIONS; do
  s_idx=$((s_idx + 1))
  session_dir="$OUT_BASE/sessions/$sid"
  session_log_dir="$LOG_DIR/$sid"
  mkdir -p "$session_log_dir"

  echo ""
  echo "=== Session $s_idx/$N_SESSIONS_ACTUAL: $sid ==="

  # Repetitions still needing work. Skipping is decided per cell for both modes: a `new`
  # run into a fresh dir has nothing to skip, and a re-invocation of `new` cannot happen
  # (the timestamp is fresh), so this collapses to the continue-mode rule.
  todo=""
  for i in $(seq 1 "$N_RUNS"); do
    cell_dir="$session_dir/run_$i"
    if [ -f "$cell_dir/$DONE_MARKER" ]; then
      echo "  run_$i  [SKIP: already complete]"
      skipped=$((skipped + 1))
      continue
    fi
    if [ -d "$cell_dir" ]; then
      # Started but never finished. Wipe it so the replay starts from a clean directory
      # rather than mixing a new run's output with a half-written one.
      echo "  run_$i  [incomplete: wiping and re-running]"
      rm -rf "$cell_dir"
    fi
    todo="$todo $i"
  done

  # Launch the outstanding repetitions in chunks of PARALLEL_REPS. bash 3.2 (macOS) has no
  # `wait -n`, so collect pids per chunk and wait on each one individually — that is what
  # preserves each cell's own exit status instead of only the last job's.
  chunk_pids=""
  chunk_cells=""
  chunk_n=0

  drain_chunk() {
    local pid cell rest_pids rest_cells rc
    rest_pids="$chunk_pids"
    rest_cells="$chunk_cells"
    for pid in $rest_pids; do
      cell="${rest_cells%% *}"
      rest_cells="${rest_cells#* }"
      wait "$pid"
      rc=$?
      if [ $rc -eq 0 ]; then
        echo "    run_$cell OK (rc=0)"
        ok=$((ok + 1))
      else
        echo "    run_$cell FAILED (rc=$rc)  see: $session_log_dir/run_$cell.log"
        fail=$((fail + 1))
      fi
    done
    chunk_pids=""
    chunk_cells=""
    chunk_n=0
  }

  for i in $todo; do
    cell_dir="$session_dir/run_$i"
    log_file="$session_log_dir/run_$i.log"
    mkdir -p "$cell_dir"

    if [ "$PARALLEL_REPS" -eq 1 ]; then
      echo "  --- run_$i -> $cell_dir"
      run_cell "$sid" "$cell_dir" "$log_file"
      rc=$?
      if [ $rc -eq 0 ]; then
        echo "    OK (rc=0)  log: $log_file"
        ok=$((ok + 1))
      else
        echo "    FAILED (rc=$rc)  see: $log_file"
        fail=$((fail + 1))
      fi
    else
      echo "  --- run_$i -> $cell_dir  [launched]"
      run_cell "$sid" "$cell_dir" "$log_file" &
      chunk_pids="$chunk_pids $!"
      chunk_cells="$chunk_cells $i"
      chunk_n=$((chunk_n + 1))
      if [ "$chunk_n" -ge "$PARALLEL_REPS" ]; then
        # Leading space from the accumulators would make the first ${x%% *} slice empty.
        chunk_pids="${chunk_pids# }"
        chunk_cells="${chunk_cells# }"
        drain_chunk
      fi
    fi
  done

  if [ "$chunk_n" -gt 0 ]; then
    chunk_pids="${chunk_pids# }"
    chunk_cells="${chunk_cells# }"
    drain_chunk
  fi
done

echo ""
echo "Cells done. ok=$ok fail=$fail skipped=$skipped"

# ------------------------------------------------------------------------------
# Step 1b — coverage grid. With failures now per cell rather than per run, raggedness is
# the thing most likely to be missed, so print it and save it as data.
# ------------------------------------------------------------------------------
echo ""
echo "--- Coverage (rows = sessions, cols = run_1..run_$N_RUNS) ---"
"$PYTHON" - "$OUT_BASE" "$N_RUNS" "$DONE_MARKER" <<'PY'
import json, os, sys

base, n_runs, marker = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ledger_path = os.path.join(base, "sessions.json")
with open(ledger_path) as f:
    ledger = json.load(f)
ids = [str(e["session_id"]) for e in (ledger.get("sessions") or []) if e.get("session_id")]

grid, complete, total = {}, 0, 0
for sid in ids:
    row = []
    for i in range(1, n_runs + 1):
        done = os.path.exists(os.path.join(base, "sessions", sid, f"run_{i}", marker))
        row.append(done)
        total += 1
        complete += 1 if done else 0
    grid[sid] = row

width = max((len(s) for s in ids), default=10)
header = "session".ljust(width) + "  " + "".join(f"{i:>4}" for i in range(1, n_runs + 1))
print(header)
for sid, row in grid.items():
    print(sid.ljust(width) + "  " + "".join(("   +" if d else "   .") for d in row))
print()
print(f"{complete}/{total} cells complete")

with open(os.path.join(base, "coverage.json"), "w") as f:
    json.dump({
        "n_runs": n_runs,
        "marker": marker,
        "cells_complete": complete,
        "cells_total": total,
        "sessions": {sid: [i + 1 for i, d in enumerate(row) if d] for sid, row in grid.items()},
        "missing": {sid: [i + 1 for i, d in enumerate(row) if not d]
                    for sid, row in grid.items() if not all(row)},
    }, f, indent=2)
    f.write("\n")

sys.exit(0 if complete else 1)
PY
coverage_rc=$?
if [ $coverage_rc -ne 0 ]; then
  echo "ERROR: no complete cells. See logs in $LOG_DIR." >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# Step 2 & 3 — analysis + viewer.
#
# The analysis scripts read this layout natively: they walk <base>/sessions/<id>/run_<i>
# one session at a time (replay_parsing.iter_sessions) and reduce in two stages —
# per-session first, across sessions second. So they run by default now; set
# RUN_ANALYSIS=0 to skip them (e.g. to replay now and analyze later, or when the judge's
# network calls are unwanted).
# ------------------------------------------------------------------------------
ANALYSIS="$OUT_BASE/analysis.json"
VIEWER="$OUT_BASE/consistency_viewer.html"

if [ "$RUN_ANALYSIS" = "1" ]; then
  echo ""
  echo "--- Analyzing -> $ANALYSIS ---"
  "$PYTHON" experiments/consistency-replay/analyze_consistency.py "$OUT_BASE" \
    --judge --out "$ANALYSIS" 2>&1 | tee "$LOG_DIR/analyze.log"

  echo ""
  echo "--- Building viewer -> $VIEWER ---"
  "$PYTHON" experiments/consistency-replay/build_viewer.py "$OUT_BASE" \
    --analysis "$ANALYSIS" \
    --out "$VIEWER" 2>&1 | tee "$LOG_DIR/build_viewer.log"
else
  echo ""
  echo "--- Analysis SKIPPED (RUN_ANALYSIS=0) ---"
  echo "The replay data is complete; only the analysis pass was skipped. Run it later with:"
  echo "  experiments/consistency-replay/analyze_experiment.sh $OUT_BASE"
  echo "or the two steps directly:"
  echo "  $PYTHON experiments/consistency-replay/analyze_consistency.py $OUT_BASE --judge --out $ANALYSIS"
  echo "  $PYTHON experiments/consistency-replay/build_viewer.py $OUT_BASE --analysis $ANALYSIS --out $VIEWER"
fi

echo ""
echo "=================================================="
echo "Done. ok=$ok fail=$fail skipped=$skipped"
echo "Results:  $OUT_BASE/"
echo "Sessions: $LEDGER"
echo "Coverage: $OUT_BASE/coverage.json"
if [ "$RUN_ANALYSIS" = "1" ]; then
  echo "Analysis: $ANALYSIS"
  echo "Viewer:   $VIEWER"
  echo "  open $VIEWER"
fi
echo "=================================================="
