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

"""List the replayable sessions of one benchmark, and record which ones an experiment ran.

One row of the Exgentic corpus is one session (one recorded agent episode). The replay
runner needs to know *which* sessions it is replaying — as data, not as a side effect of a
shuffle seed buried in the load generator. This script produces that list.

Two selection modes, mutually exclusive:

  --limit N [--seed S]   The first N sessions after the seeded shuffle. This REPLICATES the
                         load generator's own selection, so `--limit 10 --seed 41` names the
                         exact sessions a pre-session-major run of this benchmark replayed.
  --session-id ID ...    Exactly those sessions, in the order given. No shuffle.

The output ledger (`sessions.json`) is **append-only**: `--append` merges new sessions into
an existing ledger, preserving every existing entry and its `slot` verbatim. That is what
makes adding a session to a finished experiment cheap — nothing already on disk moves, and
the runner just fills in the new session's cells. `slot` is a stable label from the moment
it is assigned, NOT a position in the list.

Usage:
  list_sessions.py <BENCHMARK> [--limit N] [--seed 41] [--out sessions.json] [--ids]
  list_sessions.py <BENCHMARK> --session-id A --session-id B [--out sessions.json]
  list_sessions.py <BENCHMARK> --session-id C --append sessions.json

With no --out/--append the selection is only printed, so this doubles as a way to browse
candidate sessions before naming ids.
"""

import argparse
import datetime
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DATASET = "Exgentic/agent-llm-traces"
DEFAULT_SPLIT = "train"

# The load generator's own default is 42 (otel_trace_replay_datagen.py:359); config.yml pins
# 41 and every corpus already in reports-consistency/ was selected with 41, so 41 is the
# default here too — it keeps new runs comparable with the ones already collected.
DEFAULT_SEED = 41

# Columns kept for the ledger. `spans` is deliberately absent: it holds the whole recorded
# episode and runs to tens of GB across the corpus, so it must never be materialized here.
META_COLUMNS = (
    "session_id",
    "benchmark",
    "harness",
    "models",
    "max_tokens",
    "total_tokens",
    "collected_at",
)

# A session id is interpolated into the filter lambda that the load generator eval()s
# (_compile_filter, otel_trace_replay_datagen.py:208-236). An id carrying a quote, a paren
# or whitespace is rejected outright rather than escaped — nothing legitimate needs them.
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


# ----------------------------------------------------------------------------- selection

def shuffled_order(n: int, seed: int) -> List[int]:
    """Slot -> row index, replicating the load generator's session ordering.

    These three lines deliberately duplicate otel_trace_replay_datagen.py:437-441:

        order = list(range(num_rows))
        random.seed(self.base_seed)
        random.shuffle(order)

    They cannot be imported — they are inline in the generator's __init__, which loads the
    entire dataset as a side effect. Duplicating them is what lets this script name the
    sessions *without* running a replay; the anchor check in the README (seed 41 over
    tau2_airline reproduces an already-collected corpus) is what catches drift between the
    two copies.

    The generator shuffles indices of the *filtered* dataset, so `n` must be the
    per-benchmark row count, never the corpus size.
    """
    order = list(range(n))
    random.seed(seed)
    random.shuffle(order)
    return order


def select_shuffled(
    rows: Sequence[Dict[str, Any]], seed: int, limit: Optional[int]
) -> List[Tuple[int, Dict[str, Any]]]:
    """The first `limit` sessions in shuffled slot order, as (slot, row) pairs."""
    order = shuffled_order(len(rows), seed)
    if limit is not None:
        order = order[:limit]
    return [(slot, rows[row]) for slot, row in enumerate(order)]


def select_by_ids(
    rows: Sequence[Dict[str, Any]], session_ids: Sequence[str]
) -> List[Tuple[int, Dict[str, Any]]]:
    """Exactly the named sessions, in the order given. Raises on an unknown or repeated id.

    Slots are assigned by position. They are labels only — under the session-major layout
    the storage path is keyed by session_id, so a slot never decides where data lands.
    """
    repeated = sorted(s for s, c in Counter(session_ids).items() if c > 1)
    if repeated:
        raise LookupError("--session-id repeated: " + ", ".join(repeated))

    by_id = {r["session_id"]: r for r in rows}
    missing = [s for s in session_ids if s not in by_id]
    if missing:
        raise LookupError(
            "not in this benchmark: "
            + ", ".join(missing)
            + f" ({len(rows)} sessions available; run without --session-id to list them)"
        )
    return [(slot, by_id[sid]) for slot, sid in enumerate(session_ids)]


def validate_ids(session_ids: Sequence[str]) -> None:
    """Reject any id that could not be safely interpolated into the eval'd filter lambda."""
    bad = [s for s in session_ids if not SAFE_ID.match(s)]
    if bad:
        raise ValueError(
            "session id contains characters that are not allowed in a replay filter "
            f"(expected {SAFE_ID.pattern}): " + ", ".join(repr(s) for s in bad)
        )


# --------------------------------------------------------------------------------- ledger

def session_entry(slot: int, row: Dict[str, Any], added_at: str) -> Dict[str, Any]:
    """One ledger entry: the id, its stable slot, and the metadata worth keeping.

    harness / models are carried so downstream tables can report the recorded harness and
    the model that originally produced the trace without re-reading the dataset.
    """
    entry: Dict[str, Any] = {"slot": slot, "session_id": row["session_id"]}
    for col in ("harness", "models", "max_tokens", "total_tokens", "collected_at"):
        if col in row:
            entry[col] = row[col]
    entry["added_at"] = added_at
    return entry


def build_ledger(
    benchmark: str,
    selected: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    dataset: str,
    split: str,
    mode: str,
    seed: Optional[int],
    limit: Optional[int],
    n_available: int,
    now: str,
) -> Dict[str, Any]:
    """A fresh ledger for `selected`."""
    return {
        "dataset": dataset,
        "split": split,
        "benchmark": benchmark,
        "selection": {"mode": mode, "seed": seed, "limit": limit},
        "n_available": n_available,
        "updated_at": now,
        "sessions": [session_entry(slot, row, now) for slot, row in selected],
    }


def append_to_ledger(
    ledger: Dict[str, Any],
    selected: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    now: str,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Merge `selected` into an existing ledger without disturbing what is already there.

    Existing entries — including their `slot` — are carried over untouched: the runner has
    already written data under those session ids and re-numbering them would orphan it. New
    slots continue past the highest slot ever assigned, so a slot is never reused even if an
    entry were removed by hand.

    Returns (merged ledger, ids added, ids skipped as already present).
    """
    existing = list(ledger.get("sessions") or [])
    known = {e["session_id"] for e in existing}
    next_slot = max((int(e.get("slot", -1)) for e in existing), default=-1) + 1

    added: List[str] = []
    skipped: List[str] = []
    for _, row in selected:
        sid = row["session_id"]
        if sid in known:
            skipped.append(sid)
            continue
        existing.append(session_entry(next_slot, row, now))
        known.add(sid)
        next_slot += 1
        added.append(sid)

    merged = dict(ledger)
    merged["sessions"] = existing
    if added:
        # Only stamp the ledger when something actually changed, so a no-op append leaves
        # the file byte-identical.
        merged["updated_at"] = now
        selection = dict(merged.get("selection") or {})
        selection["mode"] = "appended"
        merged["selection"] = selection
    return merged, added, skipped


def read_ledger(path: str) -> Dict[str, Any]:
    with open(path) as f:
        ledger = json.load(f)
    if not isinstance(ledger, dict) or not isinstance(ledger.get("sessions"), list):
        raise ValueError(f"{path} is not a sessions ledger (no 'sessions' list)")
    return ledger


def write_ledger(path: str, ledger: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")


# ------------------------------------------------------------------------------- dataset

def load_index(dataset: str, split: str) -> List[Dict[str, Any]]:
    """Every row's metadata, in dataset order, with `spans` never materialized.

    Column projection happens before any row is read, so this stays a few MB regardless of
    how large the corpus's recorded episodes are. Row order matches what the load generator
    sees: it filters the full dataset, and a projection followed by the same filter leaves
    the surviving rows in the same order.
    """
    from datasets import load_dataset  # imported lazily: argument errors shouldn't pay for it

    ds = load_dataset(dataset, split=split)
    cols = list(ds.column_names)
    for required in ("session_id", "benchmark"):
        if required not in cols:
            raise ValueError(
                f"dataset {dataset!r} has no {required!r} column (found: {', '.join(cols)})"
            )
    keep = [c for c in META_COLUMNS if c in cols]
    return list(ds.select_columns(keep))


def filter_benchmark(rows: Sequence[Dict[str, Any]], benchmark: str) -> List[Dict[str, Any]]:
    """Rows of one benchmark, in dataset order — the same subset the replay filter selects."""
    return [r for r in rows if r.get("benchmark") == benchmark]


# -------------------------------------------------------------------------------- output

def print_table(selected: Sequence[Tuple[int, Dict[str, Any]]], out=sys.stdout) -> None:
    """A scannable table, so you can pick ids to pass back in via --session-id."""
    def cell(value: Any, width: int) -> str:
        text = "" if value is None else str(value)
        if len(text) > width:
            text = text[: width - 1] + "…"
        return text.ljust(width)

    print(f"{'SLOT':>4}  {cell('SESSION_ID', 34)}  {cell('HARNESS', 12)}  "
          f"{cell('MODELS', 34)}  {'TOTAL_TOKENS':>12}", file=out)
    for slot, row in selected:
        print(
            f"{slot:>4}  {cell(row.get('session_id'), 34)}  {cell(row.get('harness'), 12)}  "
            f"{cell(row.get('models'), 34)}  {str(row.get('total_tokens') or ''):>12}",
            file=out,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="List a benchmark's replayable sessions; record a run's session set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    ap.add_argument("benchmark", help="Value of the dataset's `benchmark` column, e.g. tau2_airline")
    ap.add_argument("--limit", type=int, default=None,
                    help="Keep only the first N sessions after the seeded shuffle (default: all)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Shuffle seed, matching load.base_seed (default {DEFAULT_SEED})")
    ap.add_argument("--session-id", action="append", default=None, metavar="ID",
                    help="Select this session explicitly (repeatable). Disables the shuffle.")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="Write the ledger here (default: print only)")
    ap.add_argument("--append", default=None, metavar="PATH",
                    help="Merge the selection into this existing ledger, preserving its "
                         "entries and slots. Writes back to PATH unless --out is given.")
    ap.add_argument("--ids", action="store_true",
                    help="Print bare session ids, one per line, instead of the table")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"HuggingFace dataset path (default {DEFAULT_DATASET})")
    ap.add_argument("--split", default=DEFAULT_SPLIT, help=f"Split (default {DEFAULT_SPLIT})")
    args = ap.parse_args(argv)

    if args.session_id and (args.limit is not None):
        ap.error("--session-id and --limit are mutually exclusive: explicit ids are not shuffled")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")

    if args.session_id:
        try:
            validate_ids(args.session_id)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    try:
        rows = load_index(args.dataset, args.split)
    except Exception as e:  # dataset problems are environment problems, not usage errors
        print(f"error: could not load {args.dataset!r} ({args.split}): {e}", file=sys.stderr)
        return 1

    pool = filter_benchmark(rows, args.benchmark)
    if not pool:
        known = sorted({str(r.get("benchmark")) for r in rows if r.get("benchmark")})
        print(f"error: no sessions with benchmark == {args.benchmark!r}", file=sys.stderr)
        print(f"       known benchmarks: {', '.join(known)}", file=sys.stderr)
        return 2

    try:
        if args.session_id:
            selected = select_by_ids(pool, args.session_id)
            mode, seed, limit = "explicit", None, None
        else:
            selected = select_shuffled(pool, args.seed, args.limit)
            mode, seed, limit = "shuffle", args.seed, args.limit
    except LookupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not selected:
        print(f"error: selection is empty ({len(pool)} sessions available)", file=sys.stderr)
        return 2

    # Ids from the shuffle path reach the runner's filter too, so they get the same check.
    try:
        validate_ids([row["session_id"] for _, row in selected])
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Resolve the append target BEFORE printing anything: an incompatible ledger should
    # fail outright, not after a table that suggests the selection was accepted.
    ledger: Optional[Dict[str, Any]] = None
    if args.append:
        try:
            ledger = read_ledger(args.append)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"error: cannot append to {args.append}: {e}", file=sys.stderr)
            return 2
        if ledger.get("benchmark") not in (None, args.benchmark):
            print(f"error: {args.append} is a ledger for benchmark "
                  f"{ledger['benchmark']!r}, not {args.benchmark!r}", file=sys.stderr)
            return 2

    if args.ids:
        for _, row in selected:
            print(row["session_id"])
    else:
        print_table(selected)

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    if ledger is not None:
        merged, added, skipped = append_to_ledger(ledger, selected, now=now)
        merged["n_available"] = len(pool)
        dest = args.out or args.append
        write_ledger(dest, merged)
        for sid in skipped:
            print(f"already present, left untouched: {sid}", file=sys.stderr)
        print(f"{len(added)} session(s) appended, {len(merged['sessions'])} total -> {dest}",
              file=sys.stderr)
        return 0

    if args.out:
        ledger = build_ledger(
            args.benchmark, selected,
            dataset=args.dataset, split=args.split, mode=mode, seed=seed, limit=limit,
            n_available=len(pool), now=now,
        )
        write_ledger(args.out, ledger)
        print(f"{len(selected)} of {len(pool)} session(s) -> {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
