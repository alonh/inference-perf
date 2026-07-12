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
# Run the exgentic consistency experiment: replay the same 10 traces N times against
# RITS, each run as an independent process writing its own report directory.
#
# Usage:
#   export RITS_API_KEY=<your key>
#   experiments/consistency-replay/run_consistency.sh [N_RUNS]
#
# Defaults to 10 runs. Resolves the repo root from its own location, so it can be
# invoked from anywhere. Each run writes reports-consistency/run_<i>/ containing
# per_request_lifecycle_metrics.json — the input to analyze_consistency.py.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

CONFIG="experiments/consistency-replay/config.yml"
N_RUNS="${1:-10}"
PYTHON=".venv/bin/python"
OUT_BASE="reports-consistency"
LOG_DIR="$OUT_BASE/logs"

if [ -z "${RITS_API_KEY:-}" ]; then
  echo "ERROR: RITS_API_KEY is not set. Run:  export RITS_API_KEY=<your key>" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Consistency experiment: $N_RUNS runs"
echo "Config:  $CONFIG"
echo "Output:  $OUT_BASE/run_<i>"
echo "=================================================="

ok=0
fail=0
for i in $(seq 1 "$N_RUNS"); do
  run_dir="$OUT_BASE/run_$i"
  log_file="$LOG_DIR/run_$i.log"
  echo ""
  echo "--- Run $i/$N_RUNS  ->  $run_dir ---"
  # RITS_API_KEY is injected via CLI override (the config holds only a placeholder).
  # --api.headers takes the WHOLE headers dict as a JSON string (there is no dotted
  # --api.headers.<key> flag), so pass the full object.
  "$PYTHON" -m inference_perf.main \
    --config "$CONFIG" \
    --api.headers "{\"RITS_API_KEY\":\"$RITS_API_KEY\"}" \
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
echo "=================================================="
echo "Done. ok=$ok fail=$fail"
echo "Analyze with:"
echo "  $PYTHON experiments/consistency-replay/analyze_consistency.py $OUT_BASE --judge --out $OUT_BASE/analysis.json"
echo "=================================================="
