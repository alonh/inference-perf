#!/usr/bin/env python3
"""Analyze output consistency across repeated replay runs.

Given N run directories each containing per_request_lifecycle_metrics.json, group the
requests by (session_id, request-payload hash). Because the experiment runs with
disable_output_substitution=true, a given (trace, call-position) is fed byte-identical
input on every run, so each group holds the repeated outputs for ONE identical input.
We then quantify how much those outputs differ.

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

<reports_base_dir> is scanned for run_* subdirectories.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------- IO


def find_run_dirs(base: str) -> List[str]:
    dirs = sorted(glob.glob(os.path.join(base, "run_*")))
    return [d for d in dirs if os.path.isdir(d)]


def load_run(run_dir: str) -> List[dict]:
    """Load per_request_lifecycle_metrics.json from a run dir (list of records)."""
    path = os.path.join(run_dir, "per_request_lifecycle_metrics.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    # The report may wrap the list; accept either a bare list or {"contents": [...]}.
    if isinstance(data, dict):
        data = data.get("contents", data.get("records", []))
    return data if isinstance(data, list) else []


# ------------------------------------------------------------------- extraction


def request_key(record: dict) -> str:
    """Stable hash of the request payload. Identical inputs -> identical key.

    The request body carries volatile fields (nothing timestamped by us here, but we
    still normalize by parsing + canonical re-dump so key ordering can't split a group).
    """
    req = record.get("request")
    if req is None:
        return "no-request"
    try:
        obj = json.loads(req) if isinstance(req, str) else req
        # messages + tools + generation params define the input; drop nothing —
        # canonicalize with sorted keys so formatting can't fork the group.
        canon = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        canon = req if isinstance(req, str) else str(req)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def trace_key(record: dict) -> str:
    """Stable per-trace identifier.

    Prefer the recorded session_id, but the trace-replay generator currently leaves
    session_id None on the metric (it passes the lazy-stub's None instead of the
    extracted event session id). So we fall back to a signature of the request's
    LEADING message content, which — with disable_output_substitution — is stable and
    distinct per source trace. Two calls of the same trace at different depths share
    the same leading system/context message, so this groups a whole session together.
    """
    sid = (record.get("info") or {}).get("session_id") or record.get("session_id")
    if sid:
        return str(sid)
    req = record.get("request")
    try:
        obj = json.loads(req) if isinstance(req, str) else req
        msgs = obj.get("messages") or []
        # First message content is the per-trace anchor (system/context prompt).
        anchor = (msgs[0].get("content") or "") if msgs else ""
        anchor = anchor if isinstance(anchor, str) else json.dumps(anchor, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, AttributeError):
        anchor = str(req)[:512]
    return "trace_" + hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:10]


def parse_response(record: dict) -> Dict[str, Any]:
    """Extract the fields we compare from a per-request record.

    Returns dict with: ok(bool), content(str), reasoning(str), tool_calls(list),
    finish_reason(str|None), completion_tokens(int|None), error(str|None).
    """
    out: Dict[str, Any] = {
        "ok": False,
        "content": "",
        "reasoning": "",
        "tool_calls": [],
        "finish_reason": None,
        "completion_tokens": None,
        "error": None,
    }
    if record.get("error"):
        err = record["error"]
        out["error"] = err.get("error_type") if isinstance(err, dict) else str(err)
        return out

    resp = record.get("response")
    if not resp:
        out["error"] = "empty_response"
        return out
    try:
        body = json.loads(resp) if isinstance(resp, str) else resp
    except (json.JSONDecodeError, TypeError):
        # Non-JSON body (e.g. gateway HTML error). Treat as error but keep text.
        out["error"] = "non_json_response"
        out["content"] = resp if isinstance(resp, str) else str(resp)
        return out

    choices = body.get("choices") or []
    if not choices:
        out["error"] = "no_choices"
        return out
    msg = choices[0].get("message", {}) or {}
    out["content"] = (msg.get("content") or "").strip()
    out["reasoning"] = (msg.get("reasoning_content") or "").strip()
    out["tool_calls"] = msg.get("tool_calls") or []
    out["finish_reason"] = choices[0].get("finish_reason")
    usage = body.get("usage") or {}
    out["completion_tokens"] = usage.get("completion_tokens")
    out["ok"] = True
    return out


# ---------------------------------------------------------------------- metrics


def normalized_levenshtein(a: str, b: str) -> float:
    """Similarity in [0,1] = 1 - edit_distance / max_len."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # Space-optimized DP.
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


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


def tool_signature(tool_calls: List[dict]) -> Tuple:
    """A comparable signature of a turn's tool calls: sequence of (name, sorted arg keys)."""
    sig = []
    for tc in tool_calls:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name")
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            keys = tuple(sorted(args.keys())) if isinstance(args, dict) else ("<non-dict-args>",)
        except (json.JSONDecodeError, TypeError):
            keys = ("<unparseable-args>",)
        sig.append((name, keys))
    return tuple(sig)


def tool_args_signature(tool_calls: List[dict]) -> Tuple:
    """Full signature including canonical arg VALUES (stricter than tool_signature)."""
    sig = []
    for tc in tool_calls:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name")
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            canon = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            canon = str(args_raw)
        sig.append((name, canon))
    return tuple(sig)


def agreement_fraction(items: List[Any]) -> Optional[float]:
    """Fraction of items equal to the modal item (1.0 = perfect agreement)."""
    if not items:
        return None
    counts = Counter(items)
    return counts.most_common(1)[0][1] / len(items)


# ------------------------------------------------------------------ group stats


def analyze_group(records: List[dict]) -> Dict[str, Any]:
    parsed = [parse_response(r) for r in records]
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
    # counted as non-identical rather than byte-identical.
    signatures = [
        c + "\x00" + json.dumps(tool_args_signature(p["tool_calls"]), ensure_ascii=False)
        for c, p in zip(contents, ok)
    ]
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


# --------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", help="Directory containing run_* subdirectories")
    ap.add_argument("--judge", action="store_true", help="Run LLM semantic judge (network calls)")
    ap.add_argument("--out", default=None, help="Write full JSON analysis here")
    args = ap.parse_args()

    run_dirs = find_run_dirs(args.base_dir)
    if not run_dirs:
        print(f"No run_* directories found under {args.base_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(run_dirs)} run dirs: {[os.path.basename(d) for d in run_dirs]}")

    # Group records by (session_id, request-hash) across all runs.
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    total_records = 0
    for rd in run_dirs:
        recs = load_run(rd)
        total_records += len(recs)
        for rec in recs:
            sid = trace_key(rec)
            groups[(sid, request_key(rec))].append(rec)
    print(f"Loaded {total_records} records into {len(groups)} identical-input groups.")

    # Judge config. The API key is read from the RITS_API_KEY environment variable so
    # no secret lives in the repo. Only needed when --judge is passed.
    judge_key = os.environ.get("RITS_API_KEY", "")
    if args.judge and not judge_key:
        print("warning: --judge set but RITS_API_KEY env var is empty; judge calls will fail", file=sys.stderr)
    judge_cfg = {
        "url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/"
        "qwen3-vl-235b-a22b-instruct/v1/chat/completions",
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "headers": {"RITS_API_KEY": judge_key, "Content-Type": "application/json"},
    }

    analyzed: List[Dict[str, Any]] = []
    per_trace_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (sid, rkey), recs in groups.items():
        g = analyze_group(recs)
        g["session_id"] = sid
        g["request_hash"] = rkey[:12]
        if args.judge and "exact" in g and not g["exact"]["all_identical"]:
            contents = [parse_response(r)["content"] for r in recs if parse_response(r)["ok"]]
            # Only judge groups with real text divergence. Pure tool-call turns have
            # empty content — their consistency is captured by the structural metrics,
            # and judging empty strings just wastes calls and returns noise.
            non_empty = [c for c in contents if c.strip()]
            if len(non_empty) >= 2 and len(set(non_empty)) > 1:
                g["semantic"] = judge_group(non_empty, judge_cfg)
        analyzed.append(g)
        per_trace_groups[sid].append(g)

    summary = summarize(analyzed)
    per_trace = {sid: summarize(gs) for sid, gs in per_trace_groups.items()}
    print_report(summary, per_trace)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"summary": summary, "per_trace": per_trace, "groups": analyzed},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nFull analysis written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
