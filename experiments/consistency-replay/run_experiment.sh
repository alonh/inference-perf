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

# Run the output-consistency experiment (the consistency-experiment SKILL.md pipeline)
# for a chosen MODEL + BENCHMARK, writing to a NEW timestamped results directory each
# time so previous experiments are never overwritten.
#
# Unlike run_consistency.sh (which hard-codes the model/dataset in config.yml and reuses
# the fixed reports-consistency/run_* dirs), this script:
#   * takes the model and benchmark on the command line,
#   * overrides server.base_url / server.model_name / static_model_name / hf_dataset_path
#     on top of config.yml via CLI flags (no per-model config files),
#   * writes everything under reports-consistency/<model>_<benchmark>_<timestamp>/,
#   * then runs the analyzer and builds the viewer into that same directory.
#
# This script runs in one of two MODES, given as the first argument:
#
#   new       Start a fresh experiment in a new timestamped results directory.
#   continue  Resume an existing results directory, re-running only the runs that are
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
#                `benchmark` column; this arg does NOT swap datasets — it sets a filter
#                lambda selecting rows where x['benchmark'] == <BENCHMARK>. One of:
#                  tau2_retail  appworld  swebench  tau2_airline  tau2_telecom  browsecompplus
#   N_RUNS       Number of independent replay runs (default 10).
#   TOKENIZER_NAME  Optional HF model id for the tokenizer only. Use when the endpoint's
#                model name isn't resolvable on HuggingFace (e.g. an -FP8-A100 variant):
#                pass the HF-resolvable base name here. Defaults to MODEL.
#
# continue mode:
#   OUT_BASE     An existing results directory previously produced by `new` mode, e.g.
#                reports-consistency/tau2_airline/<model-slug>/<timestamp>. The model,
#                endpoint, tokenizer, benchmark filter and request timeout are all
#                recovered from a completed run's saved config.yaml inside OUT_BASE, so
#                they do NOT need to be re-specified (and the model_base_url table is
#                not consulted).
#   N_RUNS       Total number of runs the experiment should contain (default 10). A run
#                is considered COMPLETE iff run_<i>/per_request_lifecycle_metrics.json
#                exists. Two kinds of missing runs are both re-run from scratch:
#                  (1) run_<i>/ does not exist yet (the run never started); and
#                  (2) run_<i>/ exists but its per_request_lifecycle_metrics.json is
#                      absent (the run started but did not finish).
#                Any partial run_<i>/ directory is wiped before the run is re-executed,
#                so every re-run replays all sessions from the beginning. Already-complete
#                runs are left untouched.
#
# Env:
#   REQUEST_TIMEOUT  Per-request timeout in seconds (default 1200). Raise for slow
#                    "thinking" models that can exceed aiohttp's 300s default under
#                    concurrency; harmless for fast models. In continue mode the value
#                    saved in the existing config.yaml is used unless REQUEST_TIMEOUT is
#                    explicitly set in the environment.
#
# Examples:
#   experiments/consistency-replay/run_experiment.sh new \
#       Qwen/Qwen3-VL-235B-A22B-Instruct tau2_retail 10
#   # Endpoint name differs from the HF tokenizer name:
#   experiments/consistency-replay/run_experiment.sh new \
#       Qwen/Qwen3.5-397B-A17B-FP8-A100 tau2_airline 10 Qwen/Qwen3.5-397B-A17B
#   # Resume an interrupted experiment (e.g. run_10 never finished):
#   experiments/consistency-replay/run_experiment.sh continue \
#       reports-consistency/tau2_airline/qwen-qwen3-5-397b-a17b-fp8-a100/20260804-154811 10

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
# Paths (needed early so continue mode can read an existing config.yaml)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CONFIG="experiments/consistency-replay/config.yml"
PYTHON="$REPO_ROOT/.venv/bin/python"

# The output file whose presence marks a run as COMPLETE. This is exactly the file the
# analyzer (analyze_consistency.py) loads per run, so "complete" here means "analyzable".
DONE_MARKER="per_request_lifecycle_metrics.json"

# Model is slugified for the results path (it contains '/').
slugify() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed 's/-\{1,\}/-/g; s/^-//; s/-$//'; }

usage() {
  echo "Usage:" >&2
  echo "  $0 new <MODEL> <BENCHMARK> [N_RUNS] [TOKENIZER_NAME]" >&2
  echo "  $0 continue <OUT_BASE> [N_RUNS]" >&2
  exit 2
}

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

  # Select this benchmark's rows within the dataset. The dataset itself stays fixed
  # (config.yml's hf_dataset_path); we only add a row filter on the `benchmark` column.
  FILTER="lambda x: x['benchmark'] == '$BENCHMARK'"

  # Results layout:  reports-consistency/<benchmark>/<model>/<timestamp>/
  # The benchmark and model folders are created once and reused across experiments; each
  # individual experiment gets its own timestamp subdir so nothing is ever overwritten.
  MODEL_SLUG="$(slugify "$MODEL")"
  # Timestamp is captured once so every run + the analysis + viewer share one directory.
  STAMP="$(date +%Y%m%d-%H%M%S)"
  OUT_BASE="reports-consistency/$BENCHMARK/$MODEL_SLUG/$STAMP"

elif [ "$MODE" = "continue" ]; then
  if [ "$#" -lt 1 ]; then
    echo "ERROR: 'continue' mode needs <OUT_BASE>." >&2
    usage
  fi
  OUT_BASE="${1%/}"   # strip any trailing slash so run_$i paths join cleanly
  N_RUNS="${2:-10}"
  if [ ! -d "$OUT_BASE" ]; then
    echo "ERROR: OUT_BASE '$OUT_BASE' does not exist or is not a directory." >&2
    exit 2
  fi

  # Recover model/endpoint/tokenizer/filter/timeout from a COMPLETE run's saved config.
  # A complete run is authoritative because inference_perf wrote its resolved config only
  # after the run produced output. Pick the first run_*/config.yaml that has a done marker.
  SRC_CONFIG=""
  for i in $(seq 1 "$N_RUNS"); do
    cand="$OUT_BASE/run_$i/config.yaml"
    if [ -f "$cand" ] && [ -f "$OUT_BASE/run_$i/$DONE_MARKER" ]; then
      SRC_CONFIG="$cand"
      break
    fi
  done
  if [ -z "$SRC_CONFIG" ]; then
    echo "ERROR: no completed run with a config.yaml found under '$OUT_BASE'." >&2
    echo "       Cannot recover the experiment settings to continue. Re-run in 'new' mode." >&2
    exit 2
  fi
  echo "Recovering settings from: $SRC_CONFIG"

  # Pull the settings out of the saved YAML. Emit shell-safe KEY=VALUE lines and eval them.
  SETTINGS="$(
    "$PYTHON" - "$SRC_CONFIG" <<'PY'
import sys, shlex, yaml
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
otr = (c.get("data") or {}).get("otel_trace_replay") or {}
vals = {
    "MODEL":          (c.get("server") or {}).get("model_name", ""),
    "BASE_URL":       (c.get("server") or {}).get("base_url", ""),
    "TOKENIZER_NAME": (c.get("tokenizer") or {}).get("pretrained_model_name_or_path", ""),
    "FILTER":         otr.get("filter", "") or "",
    "CFG_TIMEOUT":    (c.get("load") or {}).get("request_timeout", "") or "",
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
  )"
  if [ $? -ne 0 ] || [ -z "$SETTINGS" ]; then
    echo "ERROR: failed to parse settings from $SRC_CONFIG." >&2
    exit 2
  fi
  eval "$SETTINGS"

  # Timeout: honor an explicit env override, else use the value saved in the config,
  # else fall back to the same 1200s default as new mode.
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-${CFG_TIMEOUT:-1200}}"

  if [ -z "$MODEL" ] || [ -z "$BASE_URL" ] || [ -z "$FILTER" ]; then
    echo "ERROR: recovered settings are incomplete (MODEL/BASE_URL/FILTER)." >&2
    echo "       MODEL='$MODEL' BASE_URL='$BASE_URL' FILTER='$FILTER'" >&2
    exit 2
  fi
  # BENCHMARK is only used for display in continue mode; derive it from OUT_BASE's path.
  BENCHMARK="$(basename "$(dirname "$(dirname "$OUT_BASE")")")"

else
  echo "ERROR: unknown mode '$MODE' (expected 'new' or 'continue')." >&2
  usage
fi

if [ -z "${RITS_API_KEY:-}" ]; then
  echo "ERROR: RITS_API_KEY is not set. Run:  export RITS_API_KEY=<your key>" >&2
  exit 1
fi

LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Consistency experiment"
echo "Mode:       $MODE"
echo "Model:      $MODEL"
echo "Endpoint:   $BASE_URL"
echo "Tokenizer:  $TOKENIZER_NAME"
echo "Benchmark:  $BENCHMARK  (filter: $FILTER)"
echo "Runs:       $N_RUNS"
echo "Timeout:    ${REQUEST_TIMEOUT}s"
echo "Config:     $CONFIG"
echo "Output:     $OUT_BASE/"
echo "=================================================="

# ------------------------------------------------------------------------------
# Step 1 — N independent replay runs, each its own process + report dir.
# Model/benchmark are overridden on top of config.yml via CLI flags.
# ------------------------------------------------------------------------------
ok=0
fail=0
skipped=0
for i in $(seq 1 "$N_RUNS"); do
  run_dir="$OUT_BASE/run_$i"
  log_file="$LOG_DIR/run_$i.log"

  # Continue mode: a run is COMPLETE iff its done marker exists — leave it untouched.
  # Anything else (dir missing, or dir present but no marker = started-but-unfinished)
  # is re-run from scratch, so wipe any partial dir first to replay all sessions cleanly.
  if [ "$MODE" = "continue" ]; then
    if [ -f "$run_dir/$DONE_MARKER" ]; then
      echo ""
      echo "--- Run $i/$N_RUNS  ->  $run_dir  [SKIP: already complete] ---"
      skipped=$((skipped + 1))
      continue
    fi
    if [ -d "$run_dir" ]; then
      echo ""
      echo "--- Run $i/$N_RUNS  ->  $run_dir  [incomplete: re-running from scratch] ---"
      rm -rf "$run_dir"
    else
      echo ""
      echo "--- Run $i/$N_RUNS  ->  $run_dir  [missing: running from scratch] ---"
    fi
  else
    echo ""
    echo "--- Run $i/$N_RUNS  ->  $run_dir ---"
  fi
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
    --data.otel_trace_replay.static_model_name "$MODEL" \
    --data.otel_trace_replay.filter "$FILTER" \
    --storage.local_storage.path "$run_dir" \
    >"$log_file" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "    OK (rc=0)  log: $log_file"
    ok=$((ok + 1))
  else
    echo "    FAILED (rc=$rc)  see: $log_file"
    fail=$((fail + 1))
  fi
done

echo ""
echo "Runs done. ok=$ok fail=$fail skipped=$skipped"

# Number of runs whose done marker now exists (freshly succeeded + already-complete skips).
complete=0
for i in $(seq 1 "$N_RUNS"); do
  [ -f "$OUT_BASE/run_$i/$DONE_MARKER" ] && complete=$((complete + 1))
done
if [ "$complete" -eq 0 ]; then
  echo "ERROR: no complete runs; skipping analysis. See logs in $LOG_DIR." >&2
  exit 1
fi
echo "Complete runs available for analysis: $complete/$N_RUNS"

# ------------------------------------------------------------------------------
# Step 2 — Analyze (offline metrics + LLM-judge semantic clustering).
# ------------------------------------------------------------------------------
ANALYSIS="$OUT_BASE/analysis.json"
echo ""
echo "--- Analyzing -> $ANALYSIS ---"
"$PYTHON" experiments/consistency-replay/analyze_consistency.py "$OUT_BASE" \
  --judge --out "$ANALYSIS" 2>&1 | tee "$LOG_DIR/analyze.log"

# ------------------------------------------------------------------------------
# Step 3 — Build the self-contained interactive viewer.
# ------------------------------------------------------------------------------
VIEWER="$OUT_BASE/consistency_viewer.html"
echo ""
echo "--- Building viewer -> $VIEWER ---"
"$PYTHON" experiments/consistency-replay/build_viewer.py "$OUT_BASE" \
  --analysis "$ANALYSIS" \
  --out "$VIEWER" 2>&1 | tee "$LOG_DIR/build_viewer.log"

echo ""
echo "=================================================="
echo "Done. ok=$ok fail=$fail skipped=$skipped"
echo "Results:  $OUT_BASE/"
echo "Analysis: $ANALYSIS"
echo "Viewer:   $VIEWER"
echo "  open $VIEWER"
echo "=================================================="
