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

"""Find, per metric, a concrete run-pair whose difference THAT metric uniquely captures.

Motivation
----------
Every metric in compare/ measures a different axis of output drift. This script proves that
empirically: for each metric it searches the actual replay data for a "witness" — a pair of
repeated runs on the SAME identical input where that one metric diverges while the metrics
that would otherwise substitute for it all agree. If a clean witness exists, the metric is
capturing something no cheaper metric would have caught.

For each metric we define a COMPETITOR SET: the other metrics that, if they sufficed, would
make this one redundant. A strict witness is a pair where every competitor scores ~1.0
(agreement) but the target metric drops below threshold. Two metrics are, by construction,
strengthenings of a cheaper metric rather than orthogonal axes (TSS is a harsher reordering
penalty than LCS; finish_reason rarely moves alone) — for those we report the pair that
MAXIMIZES the target's margin over its nearest competitor instead, and say so.

All per-pair metrics come from compare/ (the single source of truth), so this script stays
in lockstep with the analyzers.

Usage
-----
  find_metric_witnesses.py <run_base_dir> [--full] [--threshold 0.999] [--json OUT]

  <run_base_dir>   experiment directory holding sessions/<session_id>/run_<i>/, each run dir
                   with per_request_lifecycle_metrics.json (the layout the analyzers consume).
                   Pairs are built per session (`session_pairs`); the witness search then runs
                   over the pooled candidates, which is a corpus-level question by nature.
  --full           print full (untruncated) content / arguments for each witness.
  --threshold T    agreement cutoff for competitors (default 0.999).
  --json OUT       also write the machine-readable witness table to OUT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

# Record reading comes from replay_parsing, the per-pair metric primitives from compare/
# (single source of truth for each).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_parsing import (  # noqa: E402
    Session,
    analyze,
    parse_response,
    collapse_ws,
    extract_tool_names,
)
from compare import (  # noqa: E402
    # Canonical units.
    tool_kv_set,
    response_signature,
    # Metrics.
    jaccard,
    normalized_levenshtein,
    fast_content_ratio,
    tool_sequence_lcs,
    tss,
    argument_consistency,
    compare_tool_calls,
    compare_tool_calls_ordered_dedup,
    compare_tool_set_overlap,
    action_histogram,
    js_kernel,
    global_alignment_kernel,
)


# --------------------------------------------------------------------------- metrics

# Each metric: name -> (human description, competitor set it must beat to be "unique").
# A competitor is a metric that would MISS this difference if it were the only one we ran.
METRIC_INFO: Dict[str, Tuple[str, List[str]]] = {
    "exact_match": (
        "byte-identical (content + canonical tool args); the strictest signal",
        [],  # exact_match is the union of everything; no single competitor subsumes it
    ),
    "content_levenshtein": (
        "graded character-level prose drift (edit distance)",
        ["tool_calls_exact", "tool_seq_lcs", "tss_editdist", "tool_set_overlap",
         "tool_args_consistency", "finish_reason_agreement", "js_kernel", "gak_ordering"],
    ),
    "content_jaccard": (
        "word-set overlap of prose (bag-of-words, order-free)",
        ["tool_calls_exact", "tool_seq_lcs", "tss_editdist", "tool_set_overlap",
         "tool_args_consistency", "finish_reason_agreement", "js_kernel", "gak_ordering"],
    ),
    "tool_seq_lcs": (
        "tool-name ORDER via longest common subsequence",
        ["tool_set_overlap", "js_kernel", "content_jaccard"],
    ),
    "tss_editdist": (
        "tool-name ORDER via token edit-distance (paper TSS; harsher on reorders than LCS)",
        ["tool_set_overlap", "js_kernel"],  # strengthening of tool_seq_lcs — see note
    ),
    "tool_set_overlap": (
        "WHICH tools were used (Jaccard of the tool set, ignores order/count)",
        ["content_jaccard"],
    ),
    "tool_args_consistency": (
        "tool ARGUMENT values (Jaccard over {(tool.key, value)})",
        ["tool_seq_lcs", "tss_editdist", "tool_set_overlap", "js_kernel", "gak_ordering"],
    ),
    "tool_calls_exact": (
        "exact tool calls incl. arg values (names + canonical args identical)",
        ["tool_seq_lcs", "tss_editdist", "tool_set_overlap", "js_kernel", "gak_ordering"],
    ),
    "tool_calls_ordered_dedup": (
        "tool calls as an arg-aware ORDERED sequence, forgiving only back-to-back repeats",
        # Misses this if they sufficed: name-only order (LCS/TSS), the tool set, order-free
        # args (AC), and composition/ordering kernels — none of which is order+arg aware.
        ["tool_seq_lcs", "tss_editdist", "tool_set_overlap", "tool_args_consistency",
         "js_kernel", "gak_ordering"],
    ),
    "js_kernel": (
        "tool COMPOSITION (action-histogram; catches count/multiplicity shifts)",
        ["tool_set_overlap"],
    ),
    "gak_ordering": (
        "tool ORDERING as a PD kernel (Global Alignment Kernel)",
        ["tool_set_overlap", "js_kernel"],
    ),
    "finish_reason_agreement": (
        "termination reason (stop / tool_calls / length) agreement",
        ["content_levenshtein", "content_jaccard", "tool_calls_exact", "tool_seq_lcs",
         "tss_editdist", "tool_set_overlap", "tool_args_consistency", "js_kernel",
         "gak_ordering"],
    ),
}

# Metrics that are deliberate strengthenings of a cheaper metric rather than a new axis.
# For these a strict "all competitors agree" witness may not exist; we fall back to the pair
# that maximizes the target's margin over the named baseline and flag it.
STRENGTHENINGS = {
    "tss_editdist": "tool_seq_lcs",   # same reorder, harsher penalty
    "finish_reason_agreement": None,  # rarely moves independently of everything else
}

ORDER = list(METRIC_INFO.keys())


def metric_vector(pa: dict, pb: dict) -> Dict[str, float]:
    """Full per-pair metric vector for two usable (ok) parsed responses."""
    ca, cb = pa["content"], pb["content"]
    na, nb = extract_tool_names(pa["tool_calls"]), extract_tool_names(pb["tool_calls"])
    kva, kvb = tool_kv_set(pa["tool_calls"]), tool_kv_set(pb["tool_calls"])
    ac = argument_consistency(kva, kvb)
    return {
        # response_signature (compare/signatures.py) is the shared exact-match unit.
        "exact_match": 1.0 if response_signature(pa) == response_signature(pb) else 0.0,
        "content_levenshtein": normalized_levenshtein(collapse_ws(ca), collapse_ws(cb)),
        "content_jaccard": jaccard(ca, cb),
        "tool_calls_exact": compare_tool_calls(pa["tool_calls"], pb["tool_calls"]),
        "tool_calls_ordered_dedup": compare_tool_calls_ordered_dedup(
            pa["tool_calls"], pb["tool_calls"]),
        "tool_seq_lcs": tool_sequence_lcs(na, nb),
        "tss_editdist": tss(na, nb),
        "tool_set_overlap": compare_tool_set_overlap(set(na), set(nb)),
        # None (neither turn has args) folds to 1.0 here — fixed-width vector, matches the
        # sibling structural metrics, and the viewer tags such pairs "trivial". This is the
        # documented fold policy; see compare.argument_consistency's docstring.
        "tool_args_consistency": ac if ac is not None else 1.0,
        "finish_reason_agreement": 1.0 if pa["finish_reason"] == pb["finish_reason"] else 0.0,
        "js_kernel": js_kernel(action_histogram(na), action_histogram(nb)),
        "gak_ordering": global_alignment_kernel(na, nb),
    }


# ------------------------------------------------------------------------------ IO

def session_pairs(session: Session) -> Dict[str, Any]:
    """Stage 1: ONE session's usable run-pairs as metric vectors.

    `session.by_event()` is event -> {run: parsed response}, so a pair is two runs' outputs for
    the SAME identical input — the only comparison a witness may be built from. Runs come in
    numeric order, so the pair printed as "A vs B" reads in replay order.

    Returns both the rows and the table they were built from: the witness search reports a pair,
    and rendering it needs the two responses back.
    """
    groups = session.by_event()
    rows = [
        (eid, a, b, metric_vector(by_run[a], by_run[b]))
        for eid, by_run in groups.items()
        for a, b in combinations(by_run, 2)
        if by_run[a]["ok"] and by_run[b]["ok"]
    ]
    return {"session_id": session.session_id, "groups": groups, "rows": rows}


def combine_pairs(per_session: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stage 2: pool every session's pairs into one corpus-wide candidate set.

    The witness search IS a corpus-level question — "somewhere in this data, is there a pair
    that isolates this metric?" — so pooling here is the point, not a leak. Each pair is still
    within one session; pooling only widens the search, it never compares across sessions.
    Event keys embed the session, so the merged `groups` lookup cannot collide.
    """
    groups: Dict[str, Dict[str, dict]] = {}
    rows: List[Tuple[str, str, str, Dict[str, float]]] = []
    for s in per_session:
        groups.update(s["groups"])
        rows.extend(s["rows"])
    return {"n_sessions": len(per_session), "groups": groups, "rows": rows}


# ------------------------------------------------------------------------- witnesses

def strict_witness(rows, target, competitors, thr):
    """Pair where all competitors >= thr but target is minimal (and < thr)."""
    best = None
    for eid, a, b, v in rows:
        if v[target] >= thr:
            continue
        if any(v[c] < thr for c in competitors):
            continue
        if best is None or v[target] < best[3][target]:
            best = (eid, a, b, v)
    return best


def margin_witness(rows, target, baseline, thr):
    """Fallback for strengthenings: maximize (baseline - target), i.e. where the target
    penalizes something the baseline is most lenient about. Requires target < thr."""
    best = None
    best_margin = -1.0
    for eid, a, b, v in rows:
        if v[target] >= thr:
            continue
        margin = (v[baseline] - v[target]) if baseline else (1.0 - v[target])
        if margin > best_margin:
            best_margin, best = margin, (eid, a, b, v)
    return best, best_margin


# ------------------------------------------------------------------------- rendering

def short(s: str, n: int, full: bool) -> str:
    s = (s or "").replace("\n", " ")
    if full or len(s) <= n:
        return s
    return s[:n] + f" ...(+{len(s)-n} chars)"


def render_pair(groups, eid, a, b, full):
    lines = []
    for tag in (a, b):
        p = groups[eid][tag]
        names = extract_tool_names(p["tool_calls"])
        lines.append(f"      [{tag}] finish={p['finish_reason']}  tools={names}")
        content = (p["content"] or "").strip()
        if content:
            lines.append(f"          content: {short(content, 200, full)}")
        for tc in p["tool_calls"]:
            fn = tc.get("function") or {}
            args = short(fn.get("arguments") or "", 200, full)
            lines.append(f"          call: {fn.get('name')}({args})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_base_dir",
                    help="Experiment directory holding sessions/<session_id>/run_<i>/")
    ap.add_argument("--full", action="store_true", help="Print untruncated content/args")
    ap.add_argument("--threshold", type=float, default=0.999,
                    help="Agreement cutoff for competitors (default 0.999)")
    ap.add_argument("--json", dest="json_out", help="Write witness table to this JSON file")
    args = ap.parse_args()

    try:
        result = analyze(args.run_base_dir, parse_response, session_pairs, combine_pairs)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"error: {e}")

    groups = result.combined["groups"]
    rows = result.combined["rows"]
    print(f"Source: {args.run_base_dir}")
    print(f"Sessions: {result.combined['n_sessions']}")
    print(f"Identical-input groups: {len(groups)}   Usable run-pairs: {len(rows)}")
    print(f"Agreement threshold: {args.threshold}\n")

    table = []
    for m in ORDER:
        desc, competitors = METRIC_INFO[m]
        header = f"── {m}  —  {desc}"
        print(header)

        if not competitors:
            # exact_match: no single competitor subsumes it. Just show its most-divergent
            # pair among those still lexically closest, to illustrate the strictest signal.
            cand = [r for r in rows if r[3][m] < args.threshold]
            if not cand:
                print("    (every usable pair is byte-identical — perfect consistency)\n")
                continue
            eid, a, b, v = max(cand, key=lambda r: r[3]["content_levenshtein"])
            print(f"    exact_match=0 while content is {v['content_levenshtein']:.1%} similar "
                  f"(near-identical but not byte-identical)")
            print(f"    group={eid.split(':')[-1]}  {a} vs {b}")
            print(render_pair(groups, eid, a, b, args.full), "\n")
            table.append({"metric": m, "kind": "strictest", "event_id": eid,
                          "run_a": a, "run_b": b, "value": v[m], "vector": v})
            continue

        w = strict_witness(rows, m, competitors, args.threshold)
        if w is not None:
            eid, a, b, v = w
            comp_min = min(v[c] for c in competitors)
            print(f"    UNIQUE: {m}={v[m]:.3f} while all competitors ≥ {comp_min:.3f} "
                  f"({', '.join(competitors)})")
            print(f"    group={eid.split(':')[-1]}  {a} vs {b}")
            print(render_pair(groups, eid, a, b, args.full), "\n")
            table.append({"metric": m, "kind": "strict_witness", "event_id": eid,
                          "run_a": a, "run_b": b, "value": v[m],
                          "competitor_min": comp_min, "vector": v})
        elif m in STRENGTHENINGS:
            base = STRENGTHENINGS[m]
            w2, margin = margin_witness(rows, m, base, args.threshold)
            if w2 is None:
                print(f"    no divergent pair (metric never drops below {args.threshold})\n")
                continue
            eid, a, b, v = w2
            note = (f"strengthening of {base}: here {m}={v[m]:.3f} vs {base}={v[base]:.3f} "
                    f"(+{margin:.3f} harsher)") if base else \
                   (f"rarely moves alone; shown maximizing its own divergence "
                    f"({m}={v[m]:.3f})")
            print(f"    NOT strictly unique — {note}")
            print(f"    group={eid.split(':')[-1]}  {a} vs {b}")
            print(render_pair(groups, eid, a, b, args.full), "\n")
            table.append({"metric": m, "kind": "strengthening", "event_id": eid,
                          "run_a": a, "run_b": b, "value": v[m], "baseline": base,
                          "vector": v})
        else:
            print(f"    no clean witness at threshold {args.threshold} "
                  f"(competitors: {', '.join(competitors)})\n")
            table.append({"metric": m, "kind": "none"})

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"source": args.run_base_dir,
                       "n_sessions": result.combined["n_sessions"],
                       "n_groups": len(groups),
                       "n_pairs": len(rows), "witnesses": table}, f, indent=2)
        print(f"Wrote witness table to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
