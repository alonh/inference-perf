#!/usr/bin/env python3
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

"""Per-session counts of events that produced a non-empty response, model by model.

"Non-empty" is `replay_parsing.parse_response`'s `has_output`: the response parsed into a
well-formed choice AND carries content or tool calls. That distinction matters here rather
than being pedantry — a reasoning model can return `ok` with an empty message when it spends
its whole token budget on `reasoning` and stops at `finish_reason=length`. Counting `ok` would
score those as answers; counting `has_output` does not, and the EMPTY-cause breakdown says how
often it happened.

Each session was replayed N times, so a (session, event) has N responses and the count of
non-empty ones can differ per repetition. The table reports the per-run count: a single number
when every repetition agreed, `mean [min-max]` when they did not.

Usage:
  summarize_events.py <BENCHMARK_DIR> [--csv OUT.csv]

  # every model under one benchmark
  summarize_events.py reports-consistency/appworld

  # or explicit experiment directories, when a benchmark holds several stamps per model
  summarize_events.py reports-consistency/appworld/<model-slug>/<stamp> ...
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_parsing import (  # noqa: E402
    SESSIONS_DIR,
    Session,
    analyze,
    parse_response,
)


def experiment_dirs(target: str) -> List[str]:
    """The experiment directories under `target`, or `[target]` if it is one itself.

    This is the one layer ABOVE the shared contract: `replay_parsing` reads one experiment, and
    comparing models means several. Everything below this function works on a single experiment.
    """
    if os.path.isdir(os.path.join(target, SESSIONS_DIR)):
        return [target.rstrip("/")]
    # <benchmark>/<model-slug>/<stamp>/sessions
    found = sorted(
        d[: -len(SESSIONS_DIR) - 1]
        for d in glob.glob(os.path.join(target, "*", "*", SESSIONS_DIR))
    )
    return found


def model_label(exp_dir: str) -> str:
    """Prefer the model the experiment recorded; fall back to the model-slug directory."""
    meta = os.path.join(exp_dir, "experiment.json")
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                model = (json.load(f) or {}).get("model")
            if model:
                return str(model)
        except (OSError, json.JSONDecodeError):
            pass
    return os.path.basename(os.path.dirname(exp_dir.rstrip("/")))


def run_counts(parsed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One (session, repetition): how many events had output, and why the rest didn't.

    `parsed` is one run's column of the session's (event, run) table, so the events are already
    distinct — the table is keyed by event, which is the identity the comparison layer groups on,
    and a duplicated record cannot inflate one repetition's count.
    """
    out = {"events": len(parsed), "non_empty": 0, "errored": 0, "reasoning_only": 0, "empty_other": 0}
    for p in parsed:
        if p["has_output"]:
            out["non_empty"] += 1
        elif not p["ok"]:
            out["errored"] += 1
        elif p["reasoning"]:
            out["reasoning_only"] += 1
        else:
            out["empty_other"] += 1
    return out


def session_counts(session: Session) -> Tuple[str, List[Dict[str, Any]]]:
    """Stage 1: one entry per repetition of ONE session, in run order.

    Note what the run-major axis buys here: a repetition that stopped early has a shorter column,
    so its `events` is genuinely smaller and `spread()` shows the raggedness instead of hiding it
    behind the union.
    """
    return session.session_id, [run_counts(list(events.values()))
                                for events in session.by_run().values()]


def scan_experiment(exp_dir: str) -> "OrderedDict[str, List[Dict[str, Any]]]":
    """Per session, one entry per repetition, in run order."""
    try:
        # Stage 2 is just "the sessions side by side": OrderedDict over the (id, cells) pairs,
        # in session order. Nothing is pooled — the cross-model comparison happens in main(),
        # one column per experiment.
        result = analyze(exp_dir, parse_response, session_counts, OrderedDict)
    except (FileNotFoundError, ValueError) as e:
        print(f"warning: skipping {exp_dir}: {e}", file=sys.stderr)
        return OrderedDict()
    return result.combined


def spread(values: Sequence[int]) -> str:
    """A single value when every repetition agreed, else `mean [min-max]`."""
    if not values:
        return "-"
    lo, hi = min(values), max(values)
    if lo == hi:
        return str(lo)
    return f"{sum(values) / len(values):.1f} [{lo}-{hi}]"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="A benchmark dir, or specific experiment dirs")
    ap.add_argument("--csv", default=None, metavar="OUT", help="Also write the per-(model, session) rows as CSV")
    args = ap.parse_args(argv)

    experiments: List[Tuple[str, str]] = []
    for target in args.targets:
        dirs = experiment_dirs(target)
        if not dirs:
            print(f"warning: no session-major experiment under {target}", file=sys.stderr)
        experiments.extend((model_label(d), d) for d in dirs)
    if not experiments:
        print("error: nothing to summarize", file=sys.stderr)
        return 2

    scanned = [(label, exp_dir, scan_experiment(exp_dir)) for label, exp_dir in experiments]

    # Union of sessions, first-seen order, so a session missing from one model still gets a row.
    sessions: List[str] = []
    for _, _, per_session in scanned:
        for sid in per_session:
            if sid not in sessions:
                sessions.append(sid)

    width = max([len(s) for s in sessions] + [7])
    col = max(max((len(label) for label, _, _ in scanned), default=10), 13)

    print("Events with a non-empty response (content or tool calls), per session per run")
    print()
    print("session".ljust(width) + "  " + "  ".join(label.rjust(col) for label, _, _ in scanned) + "   events")
    print("-" * (width + 2 + (col + 2) * len(scanned) + 9))

    rows: List[Dict[str, Any]] = []
    for sid in sessions:
        cells_by_model = [per_session.get(sid) or [] for _, _, per_session in scanned]
        event_counts = {c["events"] for cells in cells_by_model for c in cells}
        line = sid.ljust(width) + "  "
        for (label, exp_dir, _), cells in zip(scanned, cells_by_model):
            line += spread([c["non_empty"] for c in cells]).rjust(col) + "  "
            rows.append(
                {
                    "model": label,
                    "experiment": exp_dir,
                    "session_id": sid,
                    "runs": len(cells),
                    "events": cells[0]["events"] if cells else 0,
                    "non_empty_min": min((c["non_empty"] for c in cells), default=0),
                    "non_empty_max": max((c["non_empty"] for c in cells), default=0),
                    "non_empty_mean": round(sum(c["non_empty"] for c in cells) / len(cells), 2) if cells else 0,
                    "errored": sum(c["errored"] for c in cells),
                    "reasoning_only": sum(c["reasoning_only"] for c in cells),
                    "empty_other": sum(c["empty_other"] for c in cells),
                }
            )
        # The trailing column is the event count itself, so a cell short of it is visible as
        # "produced fewer answers than the session has calls" rather than looking complete.
        print(line + spread(sorted(event_counts)).rjust(8))

    print()
    for label, exp_dir, per_session in scanned:
        cells = [c for cs in per_session.values() for c in cs]
        events = sum(c["events"] for c in cells)
        non_empty = sum(c["non_empty"] for c in cells)
        pct = (100.0 * non_empty / events) if events else 0.0
        print(
            f"{label}: {non_empty}/{events} event-runs non-empty ({pct:.1f}%)  "
            f"[empty: {sum(c['errored'] for c in cells)} errored, "
            f"{sum(c['reasoning_only'] for c in cells)} reasoning-only, "
            f"{sum(c['empty_other'] for c in cells)} other]  "
            f"{len(per_session)} sessions x {max((len(cs) for cs in per_session.values()), default=0)} runs"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
