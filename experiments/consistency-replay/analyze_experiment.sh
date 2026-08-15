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

# Analyze an EXISTING consistency-experiment results directory and (re)build the viewer,
# without launching any new replay runs. This is the SKILL.md "Analysis-only (reuse
# existing runs)" path — Steps 2 / 2c / 3 against session/run dirs that already exist.
#
# Usage:
#   experiments/consistency-replay/analyze_experiment.sh <RESULTS_DIR> [options]
#
#   RESULTS_DIR   An experiment directory holding sessions/<session_id>/run_<i>/, e.g.
#                 reports-consistency/tau2_airline/qwen-qwen3-vl-235b-a22b-instruct/20260812-183412
#
#                 The OLD run-major layout (top-level run_* dirs, each holding all sessions)
#                 is NOT readable: the analysis scripts read one session at a time and there
#                 is no session axis in that layout. run_consistency.sh still writes it, so
#                 its output cannot be analyzed by this script.
#
# Options:
#   --no-judge    Skip the LLM semantic-judge pass in analyze_consistency.py (no network,
#                 faster). By default the judge runs, which needs RITS_API_KEY.
#   --papers      Also run consistency_statistics.py (Step 2c, offline, stdlib-only)
#                 and fold analysis_papers.json into the viewer's Paper-grounded summary.
#   --no-viewer   Stop after analysis; do not build the HTML viewer.
#
# Everything is written INTO <RESULTS_DIR> (analysis.json, analysis_papers.json,
# consistency_viewer.html). The existing sessions/ tree is a read-only input; nothing is deleted.
#
# The offline metrics + viewer need NO network and NO API key. Only the judge pass
# (default on; disable with --no-judge) makes network calls and reads RITS_API_KEY.

set -u

# macOS fork-safety (see run_consistency.sh). Harmless on Linux; the judge pass may fork.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
PYTHON="$REPO_ROOT/.venv/bin/python"

# ------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <RESULTS_DIR> [--no-judge] [--papers] [--no-viewer]" >&2
  echo "  e.g. $0 reports-consistency/qwen-qwen3-vl-235b-a22b-instruct_tau2-airline_20260802-105239" >&2
  exit 2
fi

RESULTS_DIR="$1"; shift
JUDGE=1
PAPERS=0
VIEWER=1
for arg in "$@"; do
  case "$arg" in
    --no-judge)  JUDGE=0 ;;
    --papers)    PAPERS=1 ;;
    --no-viewer) VIEWER=0 ;;
    *)
      echo "ERROR: unknown option '$arg'" >&2
      echo "       valid: --no-judge --papers --no-viewer" >&2
      exit 2 ;;
  esac
done

# Trim a trailing slash so path joins read cleanly.
RESULTS_DIR="${RESULTS_DIR%/}"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "ERROR: results dir not found: $RESULTS_DIR" >&2
  exit 2
fi
# Confirm it actually holds sessions/<session_id>/run_<i>/ *directories* holding the done
# marker, before doing work. (Match replay_parsing._run_dirs, which keeps only entries that
# hold per_request_lifecycle_metrics.json — a logs/<sid>/run_*.log tree must NOT pass, and
# neither must a run dir whose replay died before writing its metrics.)
n_cells=0
n_sessions=0
for sd in "$RESULTS_DIR"/sessions/*; do
  [ -d "$sd" ] || continue
  found_in_session=0
  for d in "$sd"/run_*; do
    if [ -f "$d/per_request_lifecycle_metrics.json" ]; then
      n_cells=$((n_cells + 1))
      found_in_session=1
    fi
  done
  [ "$found_in_session" -eq 1 ] && n_sessions=$((n_sessions + 1))
done
if [ "$n_cells" -eq 0 ]; then
  echo "ERROR: no sessions/<session_id>/run_<i>/per_request_lifecycle_metrics.json under $RESULTS_DIR" >&2
  echo "       Point this at a session-major results dir produced by run_experiment.sh." >&2
  if [ -d "$RESULTS_DIR/run_1" ]; then
    echo "       NOTE: $RESULTS_DIR holds top-level run_* dirs — that is the OLD run-major" >&2
    echo "       layout, which the analysis scripts no longer read." >&2
  fi
  exit 2
fi

if [ "$JUDGE" -eq 1 ] && [ -z "${RITS_API_KEY:-}" ]; then
  echo "ERROR: --judge is on (default) but RITS_API_KEY is not set." >&2
  echo "       Either 'export RITS_API_KEY=<key>' or pass --no-judge for offline analysis." >&2
  exit 1
fi

ANALYSIS="$RESULTS_DIR/analysis.json"
PAPERS_JSON="$RESULTS_DIR/analysis_papers.json"
VIEWER_HTML="$RESULTS_DIR/consistency_viewer.html"

echo "=================================================="
echo "Analyze existing experiment"
echo "Results dir: $RESULTS_DIR"
echo "Data:        $n_sessions sessions, $n_cells completed (session, run) cells"
echo "Judge:       $([ "$JUDGE" -eq 1 ] && echo 'on (network)' || echo 'off')"
echo "Papers:      $([ "$PAPERS" -eq 1 ] && echo 'yes' || echo 'no')"
echo "Viewer:      $([ "$VIEWER" -eq 1 ] && echo 'yes' || echo 'no')"
echo "=================================================="

# ------------------------------------------------------------------------------
# Step 2 — offline metrics (+ optional LLM-judge semantic clustering)
# ------------------------------------------------------------------------------
echo ""
echo "--- Analyzing -> $ANALYSIS ---"
JUDGE_FLAG=""
[ "$JUDGE" -eq 1 ] && JUDGE_FLAG="--judge"
"$PYTHON" experiments/consistency-replay/analyze_consistency.py "$RESULTS_DIR" \
  $JUDGE_FLAG --out "$ANALYSIS"
rc=$?
if [ $rc -ne 0 ]; then
  echo "ERROR: analyze_consistency.py failed (rc=$rc)." >&2
  exit $rc
fi

# ------------------------------------------------------------------------------
# Step 2c — paper-grounded metrics (optional, offline)
# ------------------------------------------------------------------------------
PAPERS_FLAG=""
if [ "$PAPERS" -eq 1 ]; then
  echo ""
  echo "--- Paper-grounded analysis -> $PAPERS_JSON ---"
  "$PYTHON" experiments/consistency-replay/consistency_statistics.py \
    --condition base="$RESULTS_DIR" \
    --out "$PAPERS_JSON"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "WARNING: consistency_statistics.py failed (rc=$rc); viewer will omit paper summary." >&2
  else
    PAPERS_FLAG="--papers $PAPERS_JSON"
  fi
fi

# ------------------------------------------------------------------------------
# Step 3 — build the self-contained viewer
# ------------------------------------------------------------------------------
if [ "$VIEWER" -eq 1 ]; then
  echo ""
  echo "--- Building viewer -> $VIEWER_HTML ---"
  "$PYTHON" experiments/consistency-replay/build_viewer.py "$RESULTS_DIR" \
    --analysis "$ANALYSIS" \
    $PAPERS_FLAG \
    --out "$VIEWER_HTML"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "ERROR: build_viewer.py failed (rc=$rc)." >&2
    exit $rc
  fi
fi

echo ""
echo "=================================================="
echo "Done."
echo "Analysis: $ANALYSIS"
[ "$PAPERS" -eq 1 ] && [ -f "$PAPERS_JSON" ] && echo "Papers:   $PAPERS_JSON"
if [ "$VIEWER" -eq 1 ]; then
  echo "Viewer:   $VIEWER_HTML"
  echo "  open $VIEWER_HTML"
fi
echo "=================================================="
