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

"""Extract per_request_lifecycle_metrics.json into a flat pandas DataFrame.

Each row is one LLM call (one record from one run). The script handles the two
structural session types present in the tau2_airline benchmark:

  user_simulator — has a system message with a <scenario> block that encodes
                   the customer task.  The agent speaks via plain text in the
                   user role; the model (simulator) replies in the assistant role
                   with plain text.  No tool calls appear in these sessions.

  agent          — no system message. The policy is embedded in the first user
                   message. Every model response is a tool call; the agent
                   "speaks" by calling mcp__environment__message whose `content`
                   argument is the text sent to the customer.

Usage
-----
  # Minimal — flat CSV
  python extract_to_dataframe.py <run_base_dir> [--out results.csv]

  # Include full response text columns
  python extract_to_dataframe.py <run_base_dir> --include-text --out results.csv

  # Parquet (recommended for large datasets)
  python extract_to_dataframe.py <run_base_dir> --format parquet --out results.parquet

<run_base_dir> is scanned for run_* subdirectories, exactly like analyze_consistency.py.

Output columns
--------------
Identity — the same two keys analyze_consistency.py groups on, so a row joins straight
onto an analyzer group:

  run_id              : str  — "run_1" … "run_N". The only run-resolved column; the analyzer
                               is deliberately run-anonymous, so this is what lets you see a
                               single run's behaviour instead of the across-run roll-up.
  session_id          : str  — one source trace (compare.session_key). Every call position in
                               that trace shares it. Falls back to a hash of the session's
                               anchor message only for old metrics files that never
                               populated it.
  event_id            : str  — ONE identical-input call position (compare.event_key), of the
                               form "<session_id>:event_<NNN>_<hash>". Because the experiment
                               runs with disable_output_substitution=true, all N runs feed
                               byte-identical input to a given event_id — so grouping the
                               frame by event_id reproduces exactly the groups that
                               analyze_consistency.py reports on.
  session_type        : str  — "user_simulator" | "agent"
  task                : str  — the customer's scenario: the text inside the <scenario> block
                               of the system message, which states the domain, the reason for
                               the call, the known info (name / user id / confirmation
                               number) and the task instructions. It is the goal the simulated
                               customer is pursuing, and it is what makes two sessions
                               different tasks rather than two samples of one. Present only on
                               user_simulator sessions — agent sessions have no system message
                               and get None (the policy is embedded in their first user
                               message instead).

Timing
  start_time          : float — wall-clock seconds (Unix epoch)
  end_time            : float — wall-clock seconds
  request_latency_sec : float — end_time - start_time

Token counts  (from server response usage block)
  prompt_tokens       : int | None
  completion_tokens   : int | None
  total_tokens        : int | None

Request — derived (no raw bodies stored by default)
  n_tools_available   : int   — number of tool definitions in the request
  last_message_role   : str   — role of messages[-1] ("user" | "tool")
  last_tool_name      : str | None — for role=tool: name of the tool whose result is in
                                     messages[-1] (i.e. what the prior turn called)

Response — structural
  finish_reason       : str | None — "stop" | "tool_calls" | "length" | None
  is_tool_turn        : bool  — True when the response contains tool_calls
  n_tool_calls        : int   — number of tool calls in the response
  tool_names          : str   — comma-separated tool names (empty string if none)
  message_to_user     : str | None — text sent to the customer via
                                     mcp__environment__message (agent sessions only)
  content_len         : int   — len(response content text)
  reasoning_len       : int   — len(chain-of-thought: message `reasoning`, or the
                                `reasoning_content` alias) — 0 for non-reasoning models
  has_error           : bool  — the response did not parse into a well-formed choice/message
                                (i.e. NOT replay_parsing.parse_response's `ok`)
  has_output          : bool  — stricter: ok AND the response actually carried content or
                                tool calls. An ok turn that spent its whole budget on
                                reasoning and emitted nothing is has_error=False,
                                has_output=False. Two such empty turns compare as vacuously
                                identical, so aggregations should either exclude them or
                                score them 0 — see compare/README.md "Error / empty Handling".
  error_type          : str | None — "empty_response" | "non_json_response" | "no_choices",
                                or the record-level error_type when the request itself failed

Metric inputs — everything compare/ needs to recompute its metrics from this frame alone.
Each is produced by the library itself (not re-derived here), so a value here is the same
value analyze_consistency.py / consistency_statistics.py compute:

  content            : str   — the response text, exactly as replay_parsing.parse_response returns
                                it (stripped, not otherwise altered). Always present, not
                                gated behind --include-text, because it is a metric input:
                                content_jaccard and analyze_consistency's
                                lexical.mean_pairwise_levenshtein both consume it raw.
  content_normalized  : str   — compare.collapse_ws(content): whitespace collapsed to single
                                spaces. This is the exact string compare_responses'
                                content_levenshtein compares.

                                Both are kept because the library's own callers disagree here:
                                compare_responses collapses whitespace before taking the edit
                                distance, while analyze_consistency passes raw content
                                straight to normalized_levenshtein. On the tau2_airline data
                                the two differ on 13 of 89 event groups (small deltas, ~0.005,
                                where responses differ only in line breaks). content_jaccard
                                is indifferent — it splits on whitespace, so raw and collapsed
                                have the same word set.
  tool_signature      : str   — JSON of compare.tool_signature(tool_calls): the ordered
                                [(name, sorted arg KEYS)] sequence. Names and argument
                                STRUCTURE, not values. Equality drives the analyzer's
                                name_and_argkeys_agreement.
  tool_args_signature : str   — JSON of compare.tool_args_signature(tool_calls): the ordered
                                [(name, canonical arg VALUES)] sequence, JSON re-serialized
                                with sorted keys so formatting cannot fork it. Equality drives
                                tool_calls_exact and full_args_agreement; the per-session
                                concatenation of this column is the profile's
                                tool_call_sequence.
  tool_kv_set         : str   — JSON of sorted(compare.tool_kv_set(tool_calls)): the
                                {("tool.argname", canonical-value)} pairs as a list. Stored as
                                a set rather than a signature because its metric
                                (argument_consistency / tool_args_consistency, Yagubyan
                                Def. 4) is a Jaccard over members, which a hash could not
                                support. json.loads it back into a set to use it.
  response_sig_sha1   : str | None — SHA1 of compare.response_signature(response), the
                                exact-match unit (normalized content PLUS canonical tool
                                args). None when the response errored, matching the library,
                                which leaves the signature undefined there. Hashed because
                                the library's signature embeds a NUL separator that is not
                                CSV-safe; every caller uses it for equality only, so the hash
                                is equivalent. The two halves are kept verbatim in
                                content_normalized and tool_args_signature if you need to
                                read one.

Trace-level (Level 2) features are not stored — they are aggregations of the above over a
(run, session): the profile's tool_sequence is the concatenation of tool_names, its
tool_call_sequence the concatenation of tool_args_signature, and the trajectory kernels
(js_kernel, global_alignment_kernel) consume the tool-name sequence.

Bulk text columns (only with --include-text) — not metric inputs, just context
  reasoning_content   : str   — full chain-of-thought text
  last_message_content: str   — full text of the last message in the request (the largest
                                column by far; this is what the flag is really gating)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional heavy dependency — fail informatively at runtime, not at import
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas", file=sys.stderr)
    sys.exit(1)

# All record reading, parsing and grouping comes from replay_parsing, so this script's
# session_id / event_id columns and its response fields are derived by exactly the same code
# as analyze_consistency.py and consistency_statistics.py. The signature columns come from
# compare/, which is the single definition of what counts as "the same response".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_parsing import (  # noqa: E402
    collapse_ws,
    event_key,
    extract_tool_names,
    find_run_dirs,
    load_run,
    parse_response,
    session_key,
)
from compare import (  # noqa: E402
    response_signature,
    tool_args_signature,
    tool_kv_set,
    tool_signature,
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_SCENARIO_RE = re.compile(r"<scenario>(.*?)</scenario>", re.DOTALL)


def _extract_scenario(system_content: str) -> Optional[str]:
    """Return the text inside <scenario>...</scenario>, stripped, or None."""
    m = _SCENARIO_RE.search(system_content)
    return m.group(1).strip() if m else None


def _metric_inputs(resp: Dict[str, Any]) -> Dict[str, Any]:
    """The compare-library values a metric is computed from, made frame-storable.

    Every value here comes out of compare/ itself rather than being re-derived, so a column
    holds the same value analyze_consistency.py and consistency_statistics.py compute. Only
    the container changes: tuples/frozensets become canonical JSON, and the exact-match
    signature becomes a SHA1 because the library's version embeds a NUL separator that CSV
    cannot carry (every caller compares signatures for equality only, so this is lossless in
    use — and its two halves are kept verbatim in content_normalized / tool_args_signature).
    """
    calls = resp["tool_calls"]
    sig = response_signature(resp)
    return {
        # Raw and normalized are BOTH kept because the library's callers disagree about which
        # they feed to Levenshtein — see the module docstring's note on content_normalized.
        "content": resp["content"],
        "content_normalized": collapse_ws(resp["content"]),
        "tool_signature": json.dumps(tool_signature(calls), ensure_ascii=False),
        "tool_args_signature": json.dumps(tool_args_signature(calls), ensure_ascii=False),
        # Sorted so the JSON is canonical; json.loads back into a set to take a Jaccard.
        "tool_kv_set": json.dumps(sorted(tool_kv_set(calls)), ensure_ascii=False),
        # None (not "") when the response errored — the library leaves it undefined there.
        "response_sig_sha1": (
            hashlib.sha1(sig.encode("utf-8")).hexdigest() if sig is not None else None
        ),
    }


def _parse_request(raw: Any) -> Dict[str, Any]:
    """Parse the request body (JSON string or dict) into a normalised dict."""
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


_MESSAGE_TOOL = "mcp__environment__message"


def _message_to_user(tool_calls: List[dict]) -> Optional[str]:
    """Extract the customer-facing text from mcp__environment__message args."""
    for tc in tool_calls:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        if fn.get("name") == _MESSAGE_TOOL:
            args_raw = fn.get("arguments")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                content = (args or {}).get("content")
                if content:
                    return str(content)
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def _last_tool_name(messages: List[dict]) -> Optional[str]:
    """When messages[-1] is a tool result, return the name of the tool that
    produced it — found on the preceding assistant turn's tool_calls."""
    if not messages or messages[-1].get("role") != "tool":
        return None
    tool_call_id = messages[-1].get("tool_call_id")
    # Walk backwards to find the assistant turn that issued this tool_call_id.
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                    return (tc.get("function") or {}).get("name")
            # Found the assistant turn but couldn't match id — fall back to
            # first tool name on that turn.
            tcs = m.get("tool_calls") or []
            if tcs:
                return (tcs[0].get("function") or {}).get("name")
            break
    return None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_records(run_dirs: List[str], include_text: bool) -> List[Dict[str, Any]]:
    """Load all run dirs and return a list of flat row dicts."""

    all_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        records = load_run(run_dir)
        if not records:
            continue

        # Group records by session (session_key) within this run.
        session_records: Dict[str, List[dict]] = defaultdict(list)
        for rec in records:
            sid = session_key(rec)
            session_records[sid].append(rec)

        # ---- build one row per record ----------------------------------------
        for sid, recs in session_records.items():
            # Chronological within the session, so recs_sorted[0] is the call that carries the
            # system message (session_type / task are read off it below).
            recs_sorted = sorted(recs, key=lambda r: r.get("start_time", 0.0))

            # Determine session_type and task from the first record's request.
            first_req = _parse_request(recs_sorted[0].get("request"))
            first_msgs: List[dict] = first_req.get("messages") or []
            sys_msg = next((m for m in first_msgs if m.get("role") == "system"), None)
            if sys_msg:
                session_type = "user_simulator"
                sys_content = sys_msg.get("content") or ""
                task = _extract_scenario(sys_content)
            else:
                session_type = "agent"
                task = None

            for rec in recs_sorted:
                eid = event_key(rec)
                req_obj = _parse_request(rec.get("request"))
                messages: List[dict] = req_obj.get("messages") or []
                tools: List[dict] = req_obj.get("tools") or []

                # parse_response takes the whole record: it reads the record-level error
                # block itself (short-circuiting before the body) and the record-level
                # timing, so there is nothing left to override here.
                resp = parse_response(rec)

                tool_calls: List[dict] = resp["tool_calls"]
                names = extract_tool_names(tool_calls)

                last_msg = messages[-1] if messages else {}
                last_role = last_msg.get("role")
                last_content = last_msg.get("content") or ""
                if not isinstance(last_content, str):
                    last_content = json.dumps(last_content, ensure_ascii=False)

                row: Dict[str, Any] = {
                    # Identity
                    "run_id": run_id,
                    "session_id": sid,
                    "event_id": eid,
                    "session_type": session_type,
                    "task": task,
                    # Timing (parse_response lifts these off the record, errors included)
                    "start_time": resp["start_time"],
                    "end_time": resp["end_time"],
                    "request_latency_sec": (
                        resp["end_time"] - resp["start_time"]
                        if resp["start_time"] is not None and resp["end_time"] is not None
                        else None
                    ),
                    # Tokens
                    "prompt_tokens": resp["prompt_tokens"],
                    "completion_tokens": resp["completion_tokens"],
                    "total_tokens": resp["total_tokens"],
                    # Request — derived
                    "n_tools_available": len(tools),
                    "last_message_role": last_role,
                    "last_tool_name": _last_tool_name(messages),
                    # Response — structural
                    "finish_reason": resp["finish_reason"],
                    "is_tool_turn": bool(tool_calls),
                    "n_tool_calls": len(tool_calls),
                    "tool_names": ", ".join(names),
                    "message_to_user": _message_to_user(tool_calls) if session_type == "agent" else None,
                    "content_len": len(resp["content"]),
                    "reasoning_len": len(resp["reasoning"]),
                    "has_error": not resp["ok"],
                    "has_output": resp["has_output"],
                    "error_type": resp["error"],
                    # Metric inputs, straight out of compare/.
                    **_metric_inputs(resp),
                }

                if include_text:
                    row["reasoning_content"] = resp["reasoning"]
                    row["last_message_content"] = last_content

                all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("base_dir", help="Directory containing run_* subdirectories")
    ap.add_argument("--out", default=None, help="Output file path (default: <base_dir>/records.csv)")
    ap.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )
    ap.add_argument(
        "--include-text",
        action="store_true",
        help="Include the bulk context columns: reasoning_content, last_message_content. "
             "(Response `content` is always included — it is a metric input.)",
    )
    args = ap.parse_args()

    run_dirs = find_run_dirs(args.base_dir)
    if not run_dirs:
        print(f"No run_* directories found under {args.base_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(run_dirs)} run dirs: {[os.path.basename(d) for d in run_dirs]}")

    rows = extract_records(run_dirs, include_text=args.include_text)
    if not rows:
        print("No records extracted — check that per_request_lifecycle_metrics.json exists in each run dir.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)

    # Consistent column order: identity → timing → tokens → request → response
    col_order = [
        "run_id", "session_id", "event_id", "session_type", "task",
        "start_time", "end_time", "request_latency_sec",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "n_tools_available", "last_message_role", "last_tool_name",
        "finish_reason", "is_tool_turn", "n_tool_calls", "tool_names",
        "message_to_user", "content_len", "reasoning_len",
        "has_error", "has_output", "error_type",
        # Metric inputs (compare library)
        "content", "content_normalized",
        "tool_signature", "tool_args_signature", "tool_kv_set",
        "response_sig_sha1",
    ]
    if args.include_text:
        col_order += ["reasoning_content", "last_message_content"]
    # Preserve any extra columns not listed above (future-proofing)
    extra = [c for c in df.columns if c not in col_order]
    df = df[col_order + extra]

    out_path = args.out or os.path.join(args.base_dir, f"records.{args.format}")

    if args.format == "parquet":
        try:
            df.to_parquet(out_path, index=False)
        except ImportError:
            print("pyarrow or fastparquet is required for parquet output: pip install pyarrow", file=sys.stderr)
            return 1
    else:
        df.to_csv(out_path, index=False)

    # Summary
    print(f"\nWrote {len(df):,} rows × {len(df.columns)} columns → {out_path}")
    print(f"\nShape breakdown:")
    print(f"  Runs:            {df['run_id'].nunique()}")
    print(f"  Sessions/run:    {df.groupby('run_id')['session_id'].nunique().mean():.1f} (mean)")
    # One event_id = one identical-input call position; this is the grain analyze_consistency.py
    # reports on, so the count should match its "identical-input groups".
    print(f"  Event groups:    {df['event_id'].nunique()} "
          f"({df.groupby('event_id').size().mean():.1f} runs each, mean)")
    print(f"  Session types:   {dict(df['session_type'].value_counts())}")
    print(f"  Calls/session:   {df.groupby(['run_id','session_id']).size().mean():.1f} (mean)")
    print(f"  Error rate:      {df['has_error'].mean()*100:.1f}%")
    print(f"  Tool turns:      {df['is_tool_turn'].mean()*100:.1f}%")
    if "task" in df.columns:
        tasks = df.dropna(subset=["task"])["task"].str[:60].unique()
        if len(tasks):
            print(f"\n  Tasks found ({len(tasks)}):")
            for t in sorted(tasks):
                print(f"    • {t}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())
