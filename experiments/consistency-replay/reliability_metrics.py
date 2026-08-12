#!/usr/bin/env python3
"""CLI for the session-level Consistency (C) metrics (hal-harness reliability_eval port).

Thin wrapper around :mod:`compare.reliability` — all metric definitions live in the
library so the notebook, viewer, and this CLI share one implementation. See that module's
docstring for the full metric mapping and the replay-vs-agent caveat.

Usage:
    python reliability_metrics.py <base_dir> [--out FILE] [--min-runs K]

where ``<base_dir>`` is a timestamp directory holding ``run_1`` .. ``run_K`` (e.g.
``reports-consistency/tau2_airline/<model>/<timestamp>``). Prints a table and writes
``reliability_metrics.json`` (per-session + aggregate) into the base dir by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

# The compare/ library holds the metric definitions; make it importable from anywhere.
_KIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _KIT_DIR not in sys.path:
    sys.path.insert(0, _KIT_DIR)

from compare import SessionMetrics, compute_consistency  # noqa: E402
from compare.reliability import (  # constants aren't re-exported in __all__  # noqa: E402
    EPSILON,
    W_OUTCOME,
    W_TRAJECTORY,
    W_RESOURCE,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
)


def _fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "   nan"
    return f"{x:6.3f}"


def _nan_to_none(x):
    return None if (isinstance(x, float) and math.isnan(x)) else x


def print_report(base: str, session_metrics: List[SessionMetrics], summary: Dict) -> None:
    print(f"\nReliability :: Consistency (C) metrics  —  {base}")
    print(f"Sessions (tasks): {len(session_metrics)}   "
          f"(hal-harness reliability_eval formulation, session-level)\n")

    header = f"{'session_id':<40} {'runs':>4} {'ok':>3}  " \
             f"{'C_out':>6} {'Ctrj_d':>6} {'Ctrj_s':>6} {'C_conf':>6} {'C_res':>6} {'R_con':>6}"
    print(header)
    print("-" * len(header))
    for m in sorted(session_metrics, key=lambda s: (s.r_con if not math.isnan(s.r_con) else 1e9)):
        print(f"{m.session_id[:40]:<40} {m.n_runs:>4} {m.n_success:>3}  "
              f"{_fmt(m.c_out)} {_fmt(m.c_traj_d)} {_fmt(m.c_traj_s)} "
              f"{_fmt(m.c_conf)} {_fmt(m.c_res)} {_fmt(m.r_con)}")
    print("-" * len(header))

    labels = {"c_out": "C_out", "c_traj_d": "C_traj_d", "c_traj_s": "C_traj_s",
              "c_conf": "C_conf", "c_res": "C_res", "r_con": "R_con"}
    na_reason = {
        "c_conf": "no self-reported confidence in replay data (always N/A)",
        "c_traj_d": "no session had ≥2 successful (clean) runs",
        "c_traj_s": "no session had ≥2 successful (clean) runs",
    }
    print("\nAggregate across sessions (mean ± SE, NaN-masked):")
    for key, label in labels.items():
        s = summary[key]
        if s["mean"] is None:
            reason = na_reason.get(key, "no defined values")
            print(f"  {label:<9}   n/a  ({reason})")
        else:
            print(f"  {label:<9} {s['mean']:6.3f} ± {s['se']:.3f}   (n={s['n']})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="Timestamp dir containing run_1..run_K")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <base>/reliability_metrics.json)")
    ap.add_argument("--min-runs", type=int, default=2,
                    help="Skip sessions appearing in fewer than this many runs (default 2)")
    args = ap.parse_args(argv)

    session_metrics, summary, skipped = compute_consistency(args.base, min_runs=args.min_runs)
    if skipped:
        print(f"[note] skipped {skipped} session(s) with < {args.min_runs} runs", file=sys.stderr)
    if not session_metrics:
        raise SystemExit("No sessions with enough runs to compute consistency.")

    print_report(args.base, session_metrics, summary)

    out_path = args.out or os.path.join(args.base, "reliability_metrics.json")
    payload = {
        "base": args.base,
        "formulation": "hal-harness reliability_eval Consistency (C), session-level replay port",
        "n_sessions": len(session_metrics),
        "constants": {"epsilon": EPSILON, "weights": [W_OUTCOME, W_TRAJECTORY, W_RESOURCE],
                      "input_cost_per_token": INPUT_COST_PER_TOKEN,
                      "output_cost_per_token": OUTPUT_COST_PER_TOKEN},
        "summary": summary,
        "per_session": [
            {"session_id": m.session_id, "n_runs": m.n_runs, "n_success": m.n_success,
             "C_out": _nan_to_none(m.c_out), "C_traj_d": _nan_to_none(m.c_traj_d),
             "C_traj_s": _nan_to_none(m.c_traj_s), "C_conf": _nan_to_none(m.c_conf),
             "C_res": _nan_to_none(m.c_res), "R_con": _nan_to_none(m.r_con)}
            for m in session_metrics
        ],
    }
    # allow_nan=False guarantees strict JSON (viewer/JS JSON.parse safe); values are already
    # nan-scrubbed above, so this just enforces the invariant.
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
