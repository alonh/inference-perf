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
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Import the analyzer's primitives so metrics match exactly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_consistency import (  # noqa: E402
    find_run_dirs,
    load_run,
    parse_response,
    request_key,
    session_key,
    event_key,
    normalized_levenshtein,
    jaccard,
    tool_args_signature,
    collapse_ws,
    strip_ws,
    cv,
)

# Paper-metric kernels (Yagubyan TSS/AC, Raj JS/GAK). Computed per run-pair per group so the
# viewer can show them on a specific A-vs-B comparison at a specific call position — the
# analyzer only stores run-level averages, which is the wrong granularity for the diff panel.
from consistency_statistics import (  # noqa: E402
    Feature,
    tss,
    argument_consistency,
    js_kernel,
    global_alignment_kernel,
)


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


def detect_model_endpoint(run_dirs: List[str]) -> tuple:
    """Read the model + endpoint actually used, from the run data — not hardcoded.

    Source of truth is the request body ("model" field = what was actually sent). Falls
    back to the run's config.yaml (server.model_name / base_url) if a request lacks it.
    The endpoint label is derived from the base URL host so the viewer reflects whichever
    model/server this particular run used (run_experiment.sh can select the model).
    """
    model_name = None
    base_url = None

    # 1) model from an actual request body (authoritative — this is what was sent).
    for rd in run_dirs:
        for rec in load_run(rd):
            req = rec.get("request")
            try:
                obj = json.loads(req) if isinstance(req, str) else req
                m = (obj or {}).get("model")
            except (json.JSONDecodeError, TypeError):
                m = None
            if m:
                model_name = m
                break
        if model_name:
            break

    # 2) fall back to (and read base_url from) the per-run config.yaml.
    for rd in run_dirs:
        cfg_path = os.path.join(rd, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
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

    run_dirs = find_run_dirs(args.base_dir)
    if not run_dirs:
        print(f"No run_* dirs under {args.base_dir}", file=sys.stderr)
        return 1

    model_name, endpoint, base_url = detect_model_endpoint(run_dirs)

    # Map event_id -> semantic info from analysis.json (optional). The analyzer keys its
    # groups on event_id, so we do too.
    semantic: Dict[str, dict] = {}
    if args.analysis and os.path.exists(args.analysis):
        adata = json.load(open(args.analysis))
        for g in adata.get("groups", []):
            if g.get("semantic") and g.get("event_id"):
                semantic[g["event_id"]] = g["semantic"]

    # Group records by event_id across runs (one group = one identical-input call
    # position). session_id is derived per group for the per-trace roll-up.
    groups: Dict[str, List[tuple]] = defaultdict(list)  # event_id -> list of (run_name, record)
    for rd in run_dirs:
        run_name = os.path.basename(rd)
        for rec in load_run(rd):
            groups[event_key(rec)].append((run_name, rec))

    out_groups: List[Dict[str, Any]] = []
    for ekey, items in groups.items():
        tkey = session_key(items[0][1])  # session_id (identical across the group)
        rkey = request_key(items[0][1])  # request-payload hash (for display/validation)
        versions = [version_from_record(rec, run) for run, rec in items]
        ok_versions = [v for v in versions if v["ok"]]

        # Comparable signature per ok version = content + canonical tool args.
        # Whitespace-collapsed so pure formatting differences (e.g. `{"content":` vs
        # `{ "content":`, or pretty-printed vs single-line JSON) don't count as distinct —
        # this drives the "N distinct" count, cluster coloring, and exact-match pill.
        def sig(v):
            toolsig = json.dumps(
                tool_args_signature(
                    [{"function": {"name": t["name"], "arguments": t["arguments"]}} for t in v["tool_calls"]]
                ),
                ensure_ascii=False,
            )
            return collapse_ws(v["content"]) + "\x00" + toolsig

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
            "n_runs": len(run_dirs),
            "runs": [os.path.basename(d) for d in run_dirs],
            "substitution_disabled": True,
        },
        "summary": summ(out_groups),
        "per_trace": {t: summ(gs) for t, gs in per_trace.items()},
        "groups": out_groups,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    size = os.path.getsize(args.out)
    print(f"Wrote {args.out}: {len(out_groups)} groups ({len(usable)} usable), {size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(build())
