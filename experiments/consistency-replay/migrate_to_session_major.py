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

"""Convert a run-major results directory into the session-major layout.

The old runner wrote one directory per repetition, each holding all ten sessions
interleaved:

    <src>/run_<i>/per_request_lifecycle_metrics.json     # 10 sessions, ~89 records

The new layout keys on the session, so adding a session later is a pure append instead of a
reorganization:

    <dest>/sessions/<session_id>/run_<i>/per_request_lifecycle_metrics.json

This script splits the former into the latter. **The source is opened read-only and never
modified** — everything is written to a new directory.

Two things it deliberately does not do:

* **It does not rewrite record contents.** Each record is copied through byte-identical,
  including its `session_id` and `event_id`. Records in old runs carry the slot from the
  ten-session shuffle (`trace3_<dataset_id>`), whereas a fresh session-major run replays one
  session per process and therefore always records slot 0 (`trace0_<dataset_id>`). The
  directory name — the bare dataset session id — is the stable key across both, and the
  ledger records each session's observed `trace_id` so the mapping is explicit rather than
  something a reader has to rediscover by parsing prefixes.
* **It does not copy the per-run aggregate reports** (`summary_*`, `stage_0_*`) into the
  session directories. Those describe a whole ten-session run; filed under one session they
  would silently claim to describe that session alone. They are preserved unsplit under
  `<dest>/source_run_reports/run_<i>/` instead.

API keys found in a copied `config.yaml` are redacted: the harness writes resolved headers
to disk, so the source configs contain the live key in plaintext.

Usage:
  migrate_to_session_major.py <SRC_BASE> [--out DEST] [--dry-run] [--no-enrich]

With no --out the destination is `<parent-of-src>/<today>-session-major`, i.e. a new dated
directory beside the source.
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_parsing import load_records  # noqa: E402

METRICS_FILE = "per_request_lifecycle_metrics.json"

# Aggregates that describe a whole run and cannot be attributed to one session.
RUN_LEVEL_REPORTS = (
    "summary_lifecycle_metrics.json",
    "summary_session_lifecycle_metrics.json",
)

REDACTED = "REDACTED_BY_MIGRATION"

# `RITS_API_KEY: <value>` and friends, as the harness serializes resolved headers.
SECRET_LINE = re.compile(r"^(\s*)([A-Za-z0-9_]*(?:API_KEY|api_key|TOKEN|token))(\s*:\s*)(.+)$")

TRACE_ID = re.compile(r"^trace(\d+)_(.+)$")


def run_index(run_dir: str) -> Optional[int]:
    """The repetition number from a run_<i> directory name."""
    m = re.match(r"^run_(\d+)$", os.path.basename(run_dir.rstrip("/")))
    return int(m.group(1)) if m else None


def source_runs(src: str) -> List[Tuple[int, str]]:
    """(index, path) for every run_<i> under src, ordered numerically.

    Numerically, not lexicographically: a plain sort puts run_10 before run_2, and the
    destination must preserve each repetition's own number so run_10 stays run_10.
    """
    out = []
    for d in glob.glob(os.path.join(src, "run_*")):
        if not os.path.isdir(d):
            continue
        idx = run_index(d)
        if idx is not None:
            out.append((idx, d))
    return sorted(out)


def dataset_session_id(recorded: str) -> str:
    """`trace3_d3d3..._20b4...` -> `d3d3..._20b4...`; unprefixed ids pass through."""
    m = TRACE_ID.match(recorded)
    return m.group(2) if m else recorded


def split_by_session(records: Sequence[dict]) -> "OrderedDict[str, List[dict]]":
    """Group a run's records by recorded session_id, preserving within-session order.

    Order matters: the analysis layer derives a call's position in the episode from the
    order of records as much as from the event_id, so a session's records must stay in the
    sequence the run emitted them.
    """
    groups: "OrderedDict[str, List[dict]]" = OrderedDict()
    for rec in records:
        sid = rec.get("session_id") or (rec.get("info") or {}).get("session_id")
        if not sid:
            raise ValueError(
                "record has no session_id, so it cannot be filed under a session; "
                f"event_id={rec.get('event_id')!r}"
            )
        groups.setdefault(str(sid), []).append(rec)
    return groups


def redact_config(text: str) -> Tuple[str, int]:
    """Blank out any api-key/token value in a saved config. Returns (text, n_redacted)."""
    out, n = [], 0
    for line in text.splitlines(keepends=True):
        m = SECRET_LINE.match(line.rstrip("\n"))
        if m and m.group(4).strip() not in ("null", "", "SET_VIA_ENV"):
            out.append(f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}\n")
            n += 1
        else:
            out.append(line)
    return "".join(out), n


def read_source_config(src_run: str) -> Dict[str, Any]:
    """The saved config of one source run, as a dict (best effort)."""
    path = os.path.join(src_run, "config.yaml")
    if not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def infer_benchmark(cfg: Dict[str, Any], src: str) -> Optional[str]:
    """Benchmark name from the saved filter lambda, falling back to the path layout."""
    filt = (((cfg.get("data") or {}).get("otel_trace_replay") or {}).get("filter")) or ""
    m = re.search(r"""\[\s*['"]benchmark['"]\s*\]\s*==\s*['"]([^'"]+)['"]""", str(filt))
    if m:
        return m.group(1)
    # reports-consistency/<benchmark>/<model>/<stamp>
    parts = os.path.abspath(src).split(os.sep)
    return parts[-3] if len(parts) >= 3 else None


def enrich(benchmark: str, ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Dataset metadata per session id, or {} if the dataset can't be read."""
    try:
        from list_sessions import filter_benchmark, load_index

        rows = filter_benchmark(load_index("Exgentic/agent-llm-traces", "train"), benchmark)
        by_id = {r["session_id"]: r for r in rows}
        return {i: by_id[i] for i in ids if i in by_id}
    except Exception as e:
        print(f"warning: no dataset metadata ({type(e).__name__}: {e}); "
              f"ledger will hold ids only", file=sys.stderr)
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a run-major results dir into the session-major layout.",
    )
    ap.add_argument("src", help="Existing results dir containing run_* subdirectories")
    ap.add_argument("--out", default=None, metavar="DEST",
                    help="Destination (default: <parent-of-src>/<today>-session-major)")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be written")
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip the dataset lookup for harness/models metadata")
    ap.add_argument("--force", action="store_true",
                    help="Write into DEST even if it already exists")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.src.rstrip("/"))
    if not os.path.isdir(src):
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 2

    runs = source_runs(src)
    if not runs:
        print(f"error: no run_* subdirectories under {src}", file=sys.stderr)
        return 2

    today = datetime.date.today().strftime("%Y%m%d")
    dest = os.path.abspath(args.out) if args.out else os.path.join(
        os.path.dirname(src), f"{today}-session-major"
    )
    if os.path.realpath(dest) == os.path.realpath(src):
        print("error: destination must differ from the source — the source is read-only",
              file=sys.stderr)
        return 2
    if os.path.exists(dest) and not (args.force or args.dry_run):
        print(f"error: {dest} already exists (use --force to write into it)", file=sys.stderr)
        return 2

    cfg = read_source_config(runs[0][1])
    benchmark = infer_benchmark(cfg, src)
    server = cfg.get("server") or {}
    otel = (cfg.get("data") or {}).get("otel_trace_replay") or {}

    # Pass 1: read every run, split it, and collect the coverage table before writing
    # anything, so a corrupt run aborts the migration instead of half-populating DEST.
    per_run: List[Tuple[int, "OrderedDict[str, List[dict]]"]] = []
    for idx, rd in runs:
        path = os.path.join(rd, METRICS_FILE)
        if not os.path.exists(path):
            print(f"warning: run_{idx} has no {METRICS_FILE} — skipping", file=sys.stderr)
            continue
        try:
            groups = split_by_session(load_records(path))
        except ValueError as e:
            print(f"error: run_{idx}: {e}", file=sys.stderr)
            return 1
        per_run.append((idx, groups))

    if not per_run:
        print(f"error: no readable runs under {src}", file=sys.stderr)
        return 1

    # Session order: first appearance, ordered by the slot the original run assigned, so the
    # ledger reads in the same order the old shuffle produced.
    recorded_ids: List[str] = []
    for _, groups in per_run:
        for rid in groups:
            if rid not in recorded_ids:
                recorded_ids.append(rid)
    recorded_ids.sort(key=lambda r: (int(m.group(1)) if (m := TRACE_ID.match(r)) else 1 << 30, r))
    ds_ids = [dataset_session_id(r) for r in recorded_ids]

    print(f"source: {src}")
    print(f"dest:   {dest}{' (dry run)' if args.dry_run else ''}")
    print(f"benchmark: {benchmark}   runs: {[i for i, _ in per_run]}   "
          f"sessions: {len(recorded_ids)}")
    print()
    header = "session".ljust(24) + "trace_id".ljust(10) + "".join(
        f"r{i}".rjust(5) for i, _ in per_run) + "total".rjust(8)
    print(header)
    total_records = 0
    for rid, dsid in zip(recorded_ids, ds_ids):
        counts = [len((groups.get(rid) or [])) for _, groups in per_run]
        total_records += sum(counts)
        slot = TRACE_ID.match(rid).group(1) if TRACE_ID.match(rid) else "-"
        print(dsid[:23].ljust(24) + f"slot {slot}".ljust(10)
              + "".join((str(c) if c else "·").rjust(5) for c in counts)
              + str(sum(counts)).rjust(8))
    print(f"\n{total_records} records across {len(per_run)} run(s)")

    ragged = [dsid for rid, dsid in zip(recorded_ids, ds_ids)
              if any(not groups.get(rid) for _, groups in per_run)]
    if ragged:
        print(f"note: {len(ragged)} session(s) are missing from at least one run: "
              + ", ".join(ragged), file=sys.stderr)

    if args.dry_run:
        return 0

    # Pass 2: write.
    n_redacted = 0
    for idx, groups in per_run:
        for rid, recs in groups.items():
            run_dest = os.path.join(dest, "sessions", dataset_session_id(rid), f"run_{idx}")
            os.makedirs(run_dest, exist_ok=True)
            with open(os.path.join(run_dest, METRICS_FILE), "w") as f:
                json.dump(recs, f)
            src_cfg = os.path.join(src, f"run_{idx}", "config.yaml")
            if os.path.exists(src_cfg):
                with open(src_cfg) as f:
                    text = f.read()
                redacted, n = redact_config(text)
                n_redacted += n
                with open(os.path.join(run_dest, "config.yaml"), "w") as f:
                    f.write(redacted)

        # Run-level aggregates, kept unsplit so they can't be mistaken for per-session data.
        keep_dest = os.path.join(dest, "source_run_reports", f"run_{idx}")
        os.makedirs(keep_dest, exist_ok=True)
        for name in RUN_LEVEL_REPORTS + ("stage_0_session_lifecycle_metrics.json",):
            p = os.path.join(src, f"run_{idx}", name)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(keep_dest, name))

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    meta = enrich(benchmark, ds_ids) if not args.no_enrich else {}

    sessions = []
    for slot, (rid, dsid) in enumerate(zip(recorded_ids, ds_ids)):
        entry: Dict[str, Any] = {"slot": slot, "session_id": dsid, "trace_id": rid}
        row = meta.get(dsid) or {}
        for col in ("harness", "models", "max_tokens", "total_tokens", "collected_at"):
            if col in row:
                entry[col] = row[col]
        entry["added_at"] = now
        sessions.append(entry)

    ledger = {
        "dataset": otel.get("hf_dataset_path") or "Exgentic/agent-llm-traces",
        "split": "train",
        "benchmark": benchmark,
        "selection": {"mode": "migrated", "seed": (cfg.get("load") or {}).get("base_seed"),
                      "limit": len(sessions)},
        "n_available": None,
        "updated_at": now,
        "sessions": sessions,
    }
    with open(os.path.join(dest, "sessions.json"), "w") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")

    experiment = {
        "benchmark": benchmark,
        "model": server.get("model_name") or otel.get("static_model_name"),
        "base_url": server.get("base_url"),
        "tokenizer": (cfg.get("tokenizer") or {}).get("pretrained_model_name_or_path")
        if isinstance(cfg.get("tokenizer"), dict) else None,
        "n_runs": max(i for i, _ in per_run),
        "run_indices": [i for i, _ in per_run],
        "layout": "session-major",
        "migrated_from": src,
        "migrated_at": now,
        "note": (
            "Converted from a run-major directory. Records are byte-identical copies and "
            "still carry the original ten-session shuffle slot in session_id/event_id "
            "(trace<slot>_<id>); a fresh session-major run records slot 0 instead. Join on "
            "the sessions/<session_id> directory name, or on sessions.json's trace_id. "
            "Run-level aggregates that cannot be split per session are under "
            "source_run_reports/. API keys in copied configs are redacted."
        ),
    }
    with open(os.path.join(dest, "experiment.json"), "w") as f:
        json.dump(experiment, f, indent=2)
        f.write("\n")

    print(f"\nwrote {dest}")
    print(f"  sessions.json ({len(sessions)} sessions), experiment.json")
    if n_redacted:
        print(f"  redacted {n_redacted} secret value(s) in copied config.yaml files")
    if not meta and not args.no_enrich:
        print("  (ledger has ids only — dataset metadata was unavailable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
