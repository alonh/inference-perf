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

"""Analyze output consistency across repeated replay runs.

Reads one experiment directory (`<base>/sessions/<session_id>/run_<i>/`) one session at a
time, and within a session groups the requests by event. An event is one call position in the
session, identified by a key stable across runs (of the form "<session_id>:event_<NNN>_<hash>"
where the records carry it). Because the experiment runs with
disable_output_substitution=true, an event is fed byte-identical input on every run, so an
event's group holds the repeated outputs for ONE identical input. We then quantify how much
those outputs differ.

The reduction is the shared two-stage one (`replay_parsing.analyze`): `session_groups` is
stage 1 and combines the RUNS of one session into that session's groups and summary — the LLM
judge runs inside it, per event — and `combine_sessions` is stage 2 and is the only place that
sees more than one session. So a session's numbers do not depend on which other sessions were
on disk.

Metrics per group (identical-input outputs):
  - completion:   how many runs returned a usable (non-error) response
  - exact:        number of distinct normalized responses; modal frequency
  - length:       mean / stdev / CV of completion tokens
  - lexical:      mean pairwise normalized-Levenshtein similarity + Jaccard(token sets)
  - structural:   for tool-call turns — tool-name agreement, arg-key agreement,
                  args-JSON-equality, finish_reason agreement
  - semantic:     optional LLM-judge clustering ("how many distinct answers?")

Usage:
  analyze_consistency.py <reports_base_dir> [--judge] [--out consistency_analysis.json]

<reports_base_dir> is an experiment directory holding sessions/<session_id>/run_<i>/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

# Parsing and the per-pair metric primitives are each imported from their single source of
# truth: replay_parsing owns everything that reads a file or a raw record (load_run,
# parse_response, the event/session/request keys), and compare/ owns the canonical units and
# the comparators (the exact-match signature, the similarity metrics). This module owns only
# the aggregation layer (modal fractions, per-trace roll-up, the LLM judge).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_parsing import (  # noqa: E402
    Session,
    analyze,
    parse_response,
    parse_records,
    request_key,
)
from compare import (  # noqa: E402
    # Canonical units.
    response_signature,
    tool_signature,
    tool_args_signature,
    # Metrics.
    normalized_levenshtein,
    jaccard,
)


# ---------------------------------------------------------------------- metrics
# Every parsing helper and per-pair metric comes from compare/ (imported above).
# mean_pairwise / cv / agreement_fraction below are aggregation helpers unique to this
# run-anonymous roll-up.


def mean_pairwise(strings: List[str], fn) -> Optional[float]:
    pairs = list(combinations(strings, 2))
    if not pairs:
        return None
    return sum(fn(x, y) for x, y in pairs) / len(pairs)


def cv(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    m = statistics.mean(vals)
    if m == 0:
        return 0.0
    return statistics.pstdev(vals) / m


def agreement_fraction(items: List[Any]) -> Optional[float]:
    """Fraction of items equal to the modal item (1.0 = perfect agreement)."""
    if not items:
        return None
    counts = Counter(items)
    return counts.most_common(1)[0][1] / len(items)


# ------------------------------------------------------------------ group stats


def analyze_group(records: List[dict]) -> Dict[str, Any]:
    parsed = parse_records(records)
    n_total = len(parsed)
    ok = [p for p in parsed if p["ok"]]
    errors = Counter(p["error"] for p in parsed if not p["ok"])

    result: Dict[str, Any] = {
        "n_runs": n_total,
        "n_ok": len(ok),
        "errors": dict(errors),
    }
    if len(ok) < 2:
        result["note"] = "fewer than 2 usable responses; consistency undefined"
        result["contents"] = [p["content"] for p in ok]
        return result

    contents = [p["content"] for p in ok]
    is_tool_turn = any(p["tool_calls"] for p in ok)

    # Exact reproducibility. The "response" is content PLUS any tool calls (canonical
    # args), so a tool turn with identical empty text but different args is correctly
    # counted as non-identical rather than byte-identical. response_signature is the shared
    # definition of that unit (compare/signatures.py) — every caller counts the same thing.
    signatures = [response_signature(p) for p in ok]
    distinct = Counter(signatures)
    result["exact"] = {
        "distinct_responses": len(distinct),
        "modal_frequency": distinct.most_common(1)[0][1] / len(signatures),
        "all_identical": len(distinct) == 1,
    }

    # Length stability.
    tok = [p["completion_tokens"] for p in ok if p["completion_tokens"] is not None]
    if not tok:  # fall back to char length if server omitted usage
        tok = [len(p["content"]) for p in ok]
        length_unit = "chars"
    else:
        length_unit = "completion_tokens"
    result["length"] = {
        "unit": length_unit,
        "mean": statistics.mean(tok),
        "stdev": statistics.pstdev(tok) if len(tok) > 1 else 0.0,
        "cv": cv(tok),
        "min": min(tok),
        "max": max(tok),
    }

    # Lexical similarity (graded drift).
    result["lexical"] = {
        "mean_pairwise_levenshtein": mean_pairwise(contents, normalized_levenshtein),
        "mean_pairwise_jaccard": mean_pairwise(contents, jaccard),
    }

    # finish_reason agreement.
    result["finish_reason_agreement"] = agreement_fraction([p["finish_reason"] for p in ok])

    # Structural (tool turns only).
    if is_tool_turn:
        name_key_sigs = [tool_signature(p["tool_calls"]) for p in ok]
        full_sigs = [tool_args_signature(p["tool_calls"]) for p in ok]
        called_tool = [bool(p["tool_calls"]) for p in ok]
        result["structural"] = {
            "is_tool_turn": True,
            "tool_call_presence_agreement": agreement_fraction(called_tool),
            "name_and_argkeys_agreement": agreement_fraction(name_key_sigs),
            "full_args_agreement": agreement_fraction(full_sigs),
            "distinct_tool_signatures": len(set(name_key_sigs)),
        }
    else:
        result["structural"] = {"is_tool_turn": False}

    return result


# --------------------------------------------------------------- semantic judge


def judge_group(contents: List[str], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ask an LLM judge how many semantically-distinct answers are in the group.

    Uses the same RITS endpoint at temperature 0. Best-effort; failures are reported,
    not fatal. Requires `requests`.
    """
    import requests  # local import so offline analysis never needs it

    numbered = "\n".join(f"[{i+1}] {c[:1500]}" for i, c in enumerate(contents))
    prompt = (
        "You are comparing model responses that were all generated from the SAME input. "
        "Group them by MEANING: two responses share a cluster if a user would consider "
        "them the same answer (ignore wording, formatting, ordering). "
        "Return ONLY a JSON object: {\"clusters\": <int>, \"rationale\": \"<one sentence>\"}.\n\n"
        f"Responses:\n{numbered}"
    )
    payload = {
        "model": model_cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
    }
    try:
        resp = requests.post(
            model_cfg["url"],
            headers=model_cfg["headers"],
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            return {"error": f"judge HTTP {resp.status_code}"}
        body = resp.json()
        text = body["choices"][0]["message"].get("content") or ""
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            return {"clusters": obj.get("clusters"), "rationale": obj.get("rationale")}
        return {"error": "judge returned no JSON", "raw": text[:200]}
    except Exception as e:  # network / parse — best effort
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------- roll-up


def summarize(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [g for g in groups if "exact" in g]
    n = len(usable)
    if n == 0:
        return {"n_groups": len(groups), "n_usable_groups": 0}

    def collect(path):
        vals = []
        for g in usable:
            cur = g
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur and cur[k] is not None:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, (int, float)):
                vals.append(cur)
        return vals

    def stat(vals):
        if not vals:
            return None
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    all_identical = [g for g in usable if g["exact"]["all_identical"]]
    tool_groups = [g for g in usable if g.get("structural", {}).get("is_tool_turn")]

    summary = {
        "n_groups": len(groups),
        "n_usable_groups": n,
        "pct_groups_byte_identical": 100.0 * len(all_identical) / n,
        "distinct_responses": stat(collect(["exact", "distinct_responses"])),
        "modal_frequency": stat(collect(["exact", "modal_frequency"])),
        "length_cv": stat(collect(["length", "cv"])),
        "lexical_levenshtein": stat(collect(["lexical", "mean_pairwise_levenshtein"])),
        "lexical_jaccard": stat(collect(["lexical", "mean_pairwise_jaccard"])),
        "finish_reason_agreement": stat(collect(["finish_reason_agreement"])),
        "n_tool_turn_groups": len(tool_groups),
    }
    if tool_groups:
        summary["tool_name_argkeys_agreement"] = stat(
            [g["structural"]["name_and_argkeys_agreement"] for g in tool_groups]
        )
        summary["tool_full_args_agreement"] = stat(
            [g["structural"]["full_args_agreement"] for g in tool_groups]
        )
    judged = collect(["semantic", "clusters"])
    if judged:
        summary["semantic_clusters"] = stat(judged)
    return summary


def print_report(summary: Dict[str, Any], per_trace: Dict[str, Any]) -> None:
    def fmt(s):
        if s is None:
            return "n/a"
        return f"mean={s['mean']:.3f} median={s['median']:.3f} range=[{s['min']:.3f},{s['max']:.3f}]"

    print("\n" + "=" * 70)
    print("OUTPUT CONSISTENCY — SUMMARY")
    print("=" * 70)
    print(f"Groups (identical-input call positions): {summary['n_groups']}")
    print(f"  usable (>=2 responses):                {summary['n_usable_groups']}")
    if summary["n_usable_groups"] == 0:
        print("  No usable groups — check that runs produced responses.")
        return
    print(f"\nByte-identical across all runs: {summary['pct_groups_byte_identical']:.1f}% of groups")
    print(f"Distinct responses / group:     {fmt(summary['distinct_responses'])}")
    print(f"Modal frequency:                {fmt(summary['modal_frequency'])}")
    print(f"Length CV:                      {fmt(summary['length_cv'])}")
    print(f"Pairwise Levenshtein sim:       {fmt(summary['lexical_levenshtein'])}")
    print(f"Pairwise Jaccard sim:           {fmt(summary['lexical_jaccard'])}")
    print(f"finish_reason agreement:        {fmt(summary['finish_reason_agreement'])}")
    print(f"\nTool-call turn groups:          {summary['n_tool_turn_groups']}")
    if summary.get("tool_name_argkeys_agreement"):
        print(f"  tool name+argkeys agreement:  {fmt(summary['tool_name_argkeys_agreement'])}")
        print(f"  full args agreement:          {fmt(summary['tool_full_args_agreement'])}")
    if summary.get("semantic_clusters"):
        print(f"\nSemantic clusters / group:      {fmt(summary['semantic_clusters'])}  (1.0 = all mean the same)")

    print("\n" + "-" * 70)
    print("PER-TRACE (session)  —  identical=%groups byte-identical, lev=mean pairwise")
    print("-" * 70)
    for sid, s in sorted(per_trace.items()):
        if s["n_usable_groups"] == 0:
            print(f"  {sid:<40} no usable groups")
            continue
        lev = s.get("lexical_levenshtein")
        lev_s = f"{lev['mean']:.3f}" if lev else "n/a"
        print(
            f"  {sid:<40} groups={s['n_usable_groups']:>3}  "
            f"identical={s['pct_groups_byte_identical']:>5.1f}%  lev={lev_s}"
        )


# ------------------------------------------------------------- the two-stage reduction


def session_groups(
    session: Session, judge_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Stage 1: combine the RUNS of one session into its identical-input groups + summary.

    One group per event: the outputs that session's runs produced for that one call position.
    `session.by_event()` is exactly that grouping, and because the event axis is the UNION over
    runs, a run that stopped early contributes to the groups it reached and is simply absent
    from the later ones — the group's `n_runs` then says so.

    The LLM judge runs HERE, per event, so the whole conclusion for a session is reachable
    from the session alone. `judge_cfg` is closed over by the caller rather than passed down
    from a corpus scan: it is configuration, not data.
    """
    analyzed: List[Dict[str, Any]] = []
    n_split_input = 0
    for eid, by_run in session.by_event().items():
        recs = list(by_run.values())  # in run order
        # Sanity check: an event should be fed byte-identical input on every run. If not,
        # output substitution was not fully disabled and this group is not an identical-input
        # group at all.
        if len({request_key(r) for r in recs}) > 1:
            n_split_input += 1
        g = analyze_group(recs)
        g["session_id"] = session.session_id
        g["event_id"] = eid
        g["request_hash"] = request_key(recs[0])[:12]
        if judge_cfg is not None and "exact" in g and not g["exact"]["all_identical"]:
            contents = [parse_response(r)["content"] for r in recs if parse_response(r)["ok"]]
            # Only judge groups with real text divergence. Pure tool-call turns have
            # empty content — their consistency is captured by the structural metrics,
            # and judging empty strings just wastes calls and returns noise.
            non_empty = [c for c in contents if c.strip()]
            if len(non_empty) >= 2 and len(set(non_empty)) > 1:
                g["semantic"] = judge_group(non_empty, judge_cfg)
        analyzed.append(g)

    return {
        "session_id": session.session_id,
        "n_runs": len(session.runs),
        "n_events": len(session.events),
        "n_missing_event_runs": len(session.missing()),
        "n_split_input_groups": n_split_input,
        "summary": summarize(analyzed),
        "groups": analyzed,
    }


def combine_sessions(per_session: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stage 2: roll the per-session conclusions up across sessions.

    `summary` is over every group of every session, which is an event-weighted pool and
    therefore lets a long session outweigh a short one. That is the pre-existing meaning of
    this script's headline and is kept; `per_trace` is the per-session breakdown to read it
    against, and consistency_statistics.py is where the session-equal-weighted estimate with
    its interval lives.
    """
    all_groups = [g for s in per_session for g in s["groups"]]
    split = sum(s["n_split_input_groups"] for s in per_session)
    if split:
        print(
            f"warning: {split} event group(s) contain differing request payloads — "
            "output substitution may not be fully disabled",
            file=sys.stderr,
        )
    return {
        "n_sessions": len(per_session),
        "n_groups": len(all_groups),
        "n_split_input_groups": split,
        "summary": summarize(all_groups),
        "per_trace": {s["session_id"]: s["summary"] for s in per_session},
        "groups": all_groups,
    }


# --------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", help="Experiment directory holding sessions/<session_id>/run_<i>/")
    ap.add_argument("--judge", action="store_true", help="Run LLM semantic judge (network calls)")
    ap.add_argument("--out", default=None, help="Write full JSON analysis here")
    args = ap.parse_args()

    # Judge config. The API key is read from the RITS_API_KEY environment variable so
    # no secret lives in the repo. Only needed when --judge is passed.
    judge_cfg = None
    if args.judge:
        judge_key = os.environ.get("RITS_API_KEY", "")
        if not judge_key:
            print("warning: --judge set but RITS_API_KEY env var is empty; judge calls will fail",
                  file=sys.stderr)
        judge_cfg = {
            "url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/"
            "qwen3-vl-235b-a22b-instruct/v1/chat/completions",
            "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "headers": {"RITS_API_KEY": judge_key, "Content-Type": "application/json"},
        }

    try:
        # transform=None: analyze_group and request_key read the RAW records.
        result = analyze(
            args.base_dir,
            None,
            lambda s: session_groups(s, judge_cfg),
            combine_sessions,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    combined = result.combined
    print(f"Analyzed {combined['n_sessions']} sessions into "
          f"{combined['n_groups']} identical-input groups.")
    print_report(combined["summary"], combined["per_trace"])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "summary": combined["summary"],
                    "per_trace": combined["per_trace"],
                    "per_session": [
                        {k: v for k, v in s.items() if k != "groups"} for s in result.per_session
                    ],
                    "groups": combined["groups"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nFull analysis written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
