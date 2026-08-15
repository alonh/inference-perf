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

"""Export per-group data for the consistency viewer web app.

Reuses analyze_consistency's parsing to build, for each identical-input call position:
  - the input prompt (last user/tool message + a short head of the system prompt)
  - each run's output version (text, tool_calls, tokens, finish_reason, error)
  - the full pairwise Levenshtein & Jaccard matrices over the successful versions
  - group-level metrics (distinct count, modal freq, length CV, tool agreement)
  - semantic clusters/rationale if present in analysis.json

Emits one JSON blob that the single-file HTML app embeds.

Usage:
  export_viewer_data.py <reports_base_dir> [--analysis analysis.json] [--out viewer_data.json]

<reports_base_dir> is an experiment directory holding sessions/<session_id>/run_<i>/. Groups are
built one session at a time, so `per_trace` is keyed by the session directory name — the same key
analyze_consistency.py writes, which is what lets --analysis join on event_id.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Parsing and grouping come from replay_parsing, metric primitives from compare/ — the same
# definitions the analyzers use, so the viewer's numbers match theirs exactly. Only `cv`
# below is analyze_consistency's own aggregation helper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_parsing import (  # noqa: E402
    SESSIONS_DIR,
    iter_sessions,
    parse_response,
    request_key,
    collapse_ws,
    strip_ws,
)
from compare import (  # noqa: E402
    response_signature,
    normalized_levenshtein,
    jaccard,
    # Paper-metric kernels (Yagubyan TSS/AC, Raj JS/GAK), taken from compare/ directly rather
    # than re-exported through consistency_statistics: they are the same functions, and going
    # to the source keeps the viewer independent of that module's scope refactors.
    tss,
    argument_consistency,
    js_kernel,
    global_alignment_kernel,
)
from analyze_consistency import cv  # noqa: E402

# Feature is the parse-once cache, not a metric, so it still comes from the analysis module.
# The diff panel needs these metrics per run-pair per GROUP (one call position), which is a
# different granularity from the session-scope numbers consistency_statistics reports.
from consistency_statistics import Feature  # noqa: E402


def input_preview(record: dict) -> Dict[str, Any]:
    """Human-readable summary of the request that produced this group."""
    req = record.get("request")
    try:
        obj = json.loads(req) if isinstance(req, str) else req
        msgs = obj.get("messages") or []
    except (json.JSONDecodeError, TypeError):
        return {"n_messages": 0, "system_head": "", "last_message": str(req)[:400], "tools": []}

    def text_of(m):
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(part.get("text", "") for part in c if isinstance(part, dict))
        return json.dumps(c, ensure_ascii=False) if c is not None else ""

    system_head = ""
    for m in msgs:
        if m.get("role") == "system":
            system_head = text_of(m)[:600]
            break
    first = msgs[0] if msgs else {}
    last = msgs[-1] if msgs else {}
    tools = obj.get("tools") or []
    tool_names = [
        (t.get("function") or {}).get("name") for t in tools if isinstance(t, dict)
    ]
    return {
        "n_messages": len(msgs),
        "system_head": system_head,
        # messages[0] = the original task. The viewer's conversation timeline shows this
        # once at the top of a trace; later steps' last_message is a tool result, not the task.
        "first_role": first.get("role"),
        "first_message": text_of(first)[:1600],
        "last_role": last.get("role"),
        "last_message": text_of(last)[:1200],
        "tools": [t for t in tool_names if t],
        "max_tokens": obj.get("max_tokens"),
    }


def version_from_record(rec: dict, run_name: str) -> Dict[str, Any]:
    p = parse_response(rec)
    tool_summary = []
    for tc in p["tool_calls"]:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        tool_summary.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    return {
        "run": run_name,
        "ok": p["ok"],
        "error": p["error"],
        "content": p["content"],
        "tool_calls": tool_summary,
        "finish_reason": p["finish_reason"],
        "completion_tokens": p["completion_tokens"],
    }


def detect_model_endpoint(base_dir: str, records: List[dict]) -> tuple:
    """Read the model + endpoint actually used, from the run data — not hardcoded.

    Source of truth is the request body ("model" field = what was actually sent), so
    `records` is a sample of records the caller has already loaded — the corpus is read
    once, by the caller, and this function does no file walk of its own beyond the config.
    Falls back to any run's config.yaml (server.model_name / base_url) if the requests lack
    a model. The endpoint label is derived from the base URL host so the viewer reflects
    whichever model/server this particular experiment used.

    This is experiment-level provenance, not per-session data: every run of every session
    in one experiment directory hit the same endpoint, so the first config.yaml that
    answers is authoritative.
    """
    model_name = None
    base_url = None

    # 1) model from an actual request body (authoritative — this is what was sent).
    for rec in records:
        req = rec.get("request")
        try:
            obj = json.loads(req) if isinstance(req, str) else req
            m = (obj or {}).get("model")
        except (json.JSONDecodeError, TypeError):
            m = None
        if m:
            model_name = m
            break

    # 2) fall back to (and read base_url from) a per-run config.yaml.
    for cfg_path in sorted(
        glob.glob(os.path.join(base_dir, SESSIONS_DIR, "*", "run_*", "config.yaml"))
    ):
        for line in open(cfg_path):
            s = line.strip()
            if model_name is None and s.startswith("model_name:"):
                model_name = s.split(":", 1)[1].strip().strip("'\"")
            elif base_url is None and s.startswith("base_url:"):
                base_url = s.split(":", 1)[1].strip().strip("'\"")
        if base_url is not None and model_name is not None:
            break

    # 3) endpoint label from the host (best-effort, purely cosmetic).
    endpoint = "unknown"
    if base_url:
        host = base_url.split("://", 1)[-1].split("/", 1)[0]
        if "rits" in host or "fmaas" in host:
            endpoint = "RITS"
        else:
            endpoint = host

    return model_name or "unknown", endpoint, base_url


def build() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("--analysis", default=None, help="analysis.json to pull semantic clusters from")
    ap.add_argument("--out", default="viewer_data.json")
    args = ap.parse_args()

    # One session at a time (the shared loader), and within a session one group per event.
    # Nothing here compares across sessions: a group is the runs of ONE session at ONE call
    # position, so the export is a concatenation of independent per-session exports.
    try:
        sessions = list(iter_sessions(args.base_dir))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # Map event_id -> semantic info from analysis.json (optional). The analyzer keys its
    # groups on event_id, so we do too.
    semantic: Dict[str, dict] = {}
    if args.analysis and os.path.exists(args.analysis):
        adata = json.load(open(args.analysis))
        for g in adata.get("groups", []):
            if g.get("semantic") and g.get("event_id"):
                semantic[g["event_id"]] = g["semantic"]

    # event_id -> [(run_id, record)], plus which session each group came from. Event keys
    # embed the session, so they stay unique across sessions and the flat dict is safe.
    groups: Dict[str, List[tuple]] = {}
    group_session: Dict[str, str] = {}
    run_ids: List[str] = []  # union of run ids, in run order — a label for the viewer's meta
    for session in sessions:
        for run_id in session.runs:
            if run_id not in run_ids:
                run_ids.append(run_id)
        for ekey, by_run in session.by_event().items():
            groups[ekey] = list(by_run.items())  # (run_id, record), in run order
            group_session[ekey] = session.session_id

    # A few records are enough to read the model off a request body.
    sample = [rec for items in list(groups.values())[:5] for _, rec in items]
    model_name, endpoint, base_url = detect_model_endpoint(args.base_dir, sample)

    out_groups: List[Dict[str, Any]] = []
    for ekey, items in groups.items():
        tkey = group_session[ekey]  # the session DIRECTORY name (bare dataset id)
        rkey = request_key(items[0][1])  # request-payload hash (for display/validation)
        versions = [version_from_record(rec, run) for run, rec in items]
        ok_versions = [v for v in versions if v["ok"]]

        # Comparable signature per ok version = content + canonical tool args.
        # Whitespace-collapsed so pure formatting differences (e.g. `{"content":` vs
        # `{ "content":`, or pretty-printed vs single-line JSON) don't count as distinct —
        # this drives the "N distinct" count, cluster coloring, and exact-match pill.
        # response_signature (compare/signatures.py) is the shared definition; a version dict
        # carries ok / content / tool_calls, and its flattened {"name", "arguments"} calls are
        # read directly by the tool-call extractor.
        sig = response_signature

        # Display string for lexical comparison (content, or tool call if no content).
        # Whitespace-normalized so the graded metrics (Content similarity, Output Jaccard)
        # ignore pure formatting — matching the exact-match rule: tool ARGS are whitespace-
        # stripped (pretty-printed vs compact JSON compare equal), content collapses
        # whitespace runs (word boundaries kept). The viewer still DISPLAYS the raw text and
        # diffs it raw; only the similarity numbers use this normalized form.
        def disp(v):
            if v["content"].strip():
                return collapse_ws(v["content"])
            if v["tool_calls"]:
                return "\n".join(f'{t["name"]}({strip_ws(t["arguments"] or "")})' for t in v["tool_calls"])
            return ""

        n = len(ok_versions)
        lev_matrix = [[None] * n for _ in range(n)]
        jac_matrix = [[None] * n for _ in range(n)]
        disps_full = [disp(v) for v in ok_versions]
        # Levenshtein is O(len^2); cap the strings fed to the pairwise matrix so a
        # handful of 4096-token outputs don't blow up export time. 2000 chars keeps
        # the similarity signal intact (differences show up well before then).
        LEV_CAP = 2000
        disps = [s[:LEV_CAP] for s in disps_full]
        for i in range(n):
            for j in range(n):
                if i == j:
                    lev_matrix[i][j] = 1.0
                    jac_matrix[i][j] = 1.0
                elif i < j:
                    lv = round(normalized_levenshtein(disps[i], disps[j]), 4)
                    jc = round(jaccard(disps[i], disps[j]), 4)
                    lev_matrix[i][j] = lev_matrix[j][i] = lv
                    jac_matrix[i][j] = jac_matrix[j][i] = jc

        # Per-run-pair paper metrics, aligned to ok_versions (same indexing as lev_matrix).
        # These let the viewer show TSS/AC/JS/GAK etc. for a specific A-vs-B pair at THIS
        # call position; the analyzer only keeps run-level averages. Each Feature parses one
        # record's tool names/args/histogram once.
        #
        # IMPORTANT: output_levenshtein / output_jaccard compare the DISPLAY string
        # `disps[i]` (= content, or `name(args)` for tool turns) — the exact text the diff
        # panel renders — NOT Feature.content. On a tool turn Feature.content is empty, so
        # comparing content alone reports 1.0 for every pair even when the tool ARGUMENTS
        # differ visibly; that made every pill read 100%. Using the display string makes the
        # pill match what the user sees diffed, and matches lev_matrix exactly.
        PAIR_METRICS = [
            "output_exact_match",
            "output_levenshtein",
            "output_jaccard",
            "tss",
            "ac",
            "js_kernel",
            "gak",
        ]
        feats = [Feature(rec) for run, rec in items if parse_response(rec)["ok"]]
        pair_metrics: Dict[str, List[List[Any]]] = {
            mk: [[None] * n for _ in range(n)] for mk in PAIR_METRICS
        }
        for mk in PAIR_METRICS:
            for i in range(n):
                # Diagonal: a version vs itself. AC is undefined for non-tool turns even on
                # the diagonal, so mirror argument_consistency's None there too.
                if mk == "ac":
                    pair_metrics[mk][i][i] = argument_consistency(feats[i].kv, feats[i].kv)
                else:
                    pair_metrics[mk][i][i] = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                fa, fb = feats[i], feats[j]
                vals = {
                    # Exact match = full content+tool-args signature (matches cluster ids).
                    "output_exact_match": 1.0 if fa.signature == fb.signature else 0.0,
                    # Text similarity over the DISPLAYED string (so tool-arg drift shows up):
                    # reuse the already-computed lev_matrix/jac_matrix (same disps, same
                    # normalized-Levenshtein / jaccard) rather than paying the O(len^2) DP twice.
                    "output_levenshtein": lev_matrix[i][j],
                    "output_jaccard": jac_matrix[i][j],
                    # Tool-structure kernels (names / ordering / composition) — legitimately
                    # 1.0 when the same tool is called the same way; argument drift is AC's job.
                    "tss": round(tss(fa.names, fb.names), 4),
                    "js_kernel": round(js_kernel(fa.hist, fb.hist), 4),
                    "gak": round(global_alignment_kernel(fa.names, fb.names), 4),
                }
                ac = argument_consistency(fa.kv, fb.kv)
                vals["ac"] = round(ac, 4) if ac is not None else None
                for mk in PAIR_METRICS:
                    pair_metrics[mk][i][j] = pair_metrics[mk][j][i] = vals[mk]

        sigs = [sig(v) for v in ok_versions]
        distinct = len(set(sigs)) if sigs else 0
        modal = 0.0
        if sigs:
            from collections import Counter

            modal = Counter(sigs).most_common(1)[0][1] / len(sigs)
        toks = [v["completion_tokens"] for v in ok_versions if v["completion_tokens"] is not None]
        is_tool = any(v["tool_calls"] for v in ok_versions)

        # Cluster id per ok version (which distinct signature it belongs to) for coloring.
        sig_to_id: Dict[str, int] = {}
        cluster_ids = []
        for s in sigs:
            if s not in sig_to_id:
                sig_to_id[s] = len(sig_to_id)
            cluster_ids.append(sig_to_id[s])

        pairwise_mean_lev = None
        if n >= 2:
            vals = [lev_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
            pairwise_mean_lev = round(statistics.mean(vals), 4) if vals else None

        out_groups.append(
            {
                "trace": tkey,
                "event_id": ekey,
                "request_hash": rkey[:12],
                "input": input_preview(items[0][1]),
                "versions": versions,          # all, including errors (greyed in UI)
                "ok_index_map": [i for i, v in enumerate(versions) if v["ok"]],
                "cluster_ids": cluster_ids,     # aligned to ok versions
                "lev_matrix": lev_matrix,
                "jac_matrix": jac_matrix,
                "pair_metrics": pair_metrics,  # per run-pair paper kernels, aligned to ok versions
                "metrics": {
                    "n_runs": len(versions),
                    "n_ok": n,
                    "distinct": distinct,
                    "modal_frequency": round(modal, 4),
                    "all_identical": distinct == 1 and n >= 2,
                    "mean_pairwise_lev": pairwise_mean_lev,
                    "length_cv": round(cv(toks), 4) if toks else None,
                    "length_min": min(toks) if toks else None,
                    "length_max": max(toks) if toks else None,
                    "is_tool_turn": is_tool,
                },
                "semantic": semantic.get(ekey),
            }
        )

    # Sort: most inconsistent first (more distinct, lower lev), tool turns tie-break.
    def inconsistency(g):
        m = g["metrics"]
        lev = m["mean_pairwise_lev"] if m["mean_pairwise_lev"] is not None else 1.0
        return (-(m["distinct"] or 0), lev)

    out_groups.sort(key=inconsistency)

    # Overall + per-trace summaries.
    usable = [g for g in out_groups if g["metrics"]["n_ok"] >= 2]

    def summ(gs):
        u = [g for g in gs if g["metrics"]["n_ok"] >= 2]
        if not u:
            return {"n_groups": len(gs), "n_usable": 0}
        levs = [g["metrics"]["mean_pairwise_lev"] for g in u if g["metrics"]["mean_pairwise_lev"] is not None]
        return {
            "n_groups": len(gs),
            "n_usable": len(u),
            "pct_identical": round(100 * sum(g["metrics"]["all_identical"] for g in u) / len(u), 1),
            "median_distinct": statistics.median([g["metrics"]["distinct"] for g in u]),
            "median_lev": round(statistics.median(levs), 3) if levs else None,
        }

    per_trace = defaultdict(list)
    for g in out_groups:
        per_trace[g["trace"]].append(g)

    payload = {
        "meta": {
            "model": model_name,
            "endpoint": endpoint,
            "base_url": base_url,
            "n_sessions": len(sessions),
            "n_runs": len(run_ids),
            "runs": run_ids,
            "substitution_disabled": True,
        },
        "summary": summ(out_groups),
        "per_trace": {t: summ(gs) for t, gs in per_trace.items()},
        "groups": out_groups,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    size = os.path.getsize(args.out)
    print(
        f"Wrote {args.out}: {len(out_groups)} groups ({len(usable)} usable) "
        f"from {len(sessions)} sessions x {len(run_ids)} runs, {size/1024:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(build())
