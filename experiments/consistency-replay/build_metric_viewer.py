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

"""Build the METRIC-DISAGREEMENT viewer: one tab per model, one metric per sidebar row.

This is a NEW, standalone viewer — it does not touch the existing consistency viewer
(build_viewer.py / viewer_template.html / export_viewer_data.py) or any analyzer. It answers
a different question: for each metric, where does it DISAGREE with the others?

For every model and every metric M it surfaces up to three examples in each direction:

  (1) M says the two runs are SIMILAR while other metrics say they DIFFER
      (M >= --hi, and at least one other metric <= --lo)
  (2) M says the two runs DIFFER while other metrics say they are SIMILAR
      (M <= --lo, and at least one other metric >= --hi)

Each example embeds both runs' content / tool-calls and the FULL metric vector, so the
viewer can show exactly which metrics landed on the opposite side. The per-pair metric math
is imported from find_metric_witnesses.py (which itself uses compare/), so this viewer stays
in lockstep with the analyzers — no metric is redefined here.

Usage:
  build_metric_viewer.py <run_dir> [<run_dir> ...] [--hi 0.85] [--lo 0.60]
      [--per-cell 3] [--max-chars 1800] [--out metric_viewer.html]

Each <run_dir> is a timestamped directory containing run_* subdirs (exactly what the
analyzers consume). The model label is read from run_1/config.yaml (static_model_name).

Example (all three tau2_airline models):
  build_metric_viewer.py \
      reports-consistency/tau2_airline/qwen-qwen3-vl-235b-a22b-instruct/20260804-144419 \
      reports-consistency/tau2_airline/qwen-qwen3-vl-235b-a22b-thinking/20260804-153833 \
      reports-consistency/tau2_airline/qwen-qwen3-5-397b-a17b-fp8-a100/20260804-154811 \
      --out metric_viewer.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the per-pair metric machinery (which imports compare/). Single source of truth.
import find_metric_witnesses as fw  # noqa: E402
from compare import extract_tool_names  # noqa: E402

TEMPLATE = os.path.join(HERE, "metric_viewer_template.html")


def model_label(run_dir: str) -> str:
    """Human label from run_1/config.yaml (static_model_name), else the dir's model slug."""
    cfg = os.path.join(run_dir, "run_1", "config.yaml")
    if os.path.exists(cfg):
        for line in open(cfg):
            line = line.strip()
            if line.startswith("static_model_name:"):
                name = line.split(":", 1)[1].strip().strip('"\'')
                if name and name.lower() != "null":
                    return name
    # Fallback: the model slug is the parent dir of the timestamp dir.
    return os.path.basename(os.path.dirname(os.path.abspath(run_dir))) or run_dir


def short(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n…(+{len(s) - n} chars truncated)"


def side_payload(parsed: dict, max_chars: int) -> dict:
    """Compact render-ready view of one run's response."""
    calls = []
    for tc in parsed.get("tool_calls") or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        calls.append({
            "name": fn.get("name"),
            "arguments": short(fn.get("arguments") or "", max_chars),
        })
    return {
        "content": short(parsed.get("content") or "", max_chars),
        "tool_calls": calls,
        "finish_reason": parsed.get("finish_reason"),
        "tools": list(extract_tool_names(parsed.get("tool_calls") or [])),
    }


# Metrics whose "similar" verdict is vacuous when BOTH sides are empty of the thing they
# measure. jaccard("","")=1 and tss((),())=1 are true-by-emptiness, not a real agreement, so
# such examples are deprioritized (ranked last) and tagged "trivial" for the viewer.
CONTENT_METRICS = {"content_levenshtein", "content_jaccard"}
TOOL_METRICS = {"tool_seq_lcs", "tss_editdist", "tool_set_overlap", "tool_args_consistency",
                "tool_calls_exact", "tool_calls_ordered_dedup", "js_kernel", "gak_ordering"}


def is_trivial_similarity(target, pa, pb) -> bool:
    """True when target's high score is an empty-vs-empty artifact rather than real agreement."""
    if target in CONTENT_METRICS:
        return not (pa.get("content") or "").strip() and not (pb.get("content") or "").strip()
    if target in TOOL_METRICS:
        return not (pa.get("tool_calls") or []) and not (pb.get("tool_calls") or [])
    return False


def find_examples(rows, groups, target, hi, lo, per_cell, max_chars):
    """Return {'similar_but_differ': [...], 'differ_but_similar': [...]} for one metric.

    similar_but_differ: target >= hi while >=1 OTHER metric <= lo (this metric alone says OK).
    differ_but_similar: target <= lo while >=1 OTHER metric >= hi (this metric alone flags it).
    Ranked by (informative-first, #opposing metrics desc, gap desc); deduped by event group.
    """
    others = [m for m in fw.ORDER if m != target]

    def collect(is_similar_dir):
        cands = []
        for eid, a, b, v in rows:
            tv = v[target]
            if is_similar_dir:
                if tv < hi:
                    continue
                opposing = [m for m in others if v[m] <= lo]
                if not opposing:
                    continue
                gap = tv - min(v[m] for m in opposing)
                # Empty-vs-empty inflates the "similar" side only; irrelevant when differing.
                trivial = is_trivial_similarity(target, groups[eid][a], groups[eid][b])
            else:
                if tv > lo:
                    continue
                opposing = [m for m in others if v[m] >= hi]
                if not opposing:
                    continue
                gap = max(v[m] for m in opposing) - tv
                trivial = False
            # informative (non-trivial) first, then most opposition, then largest gap.
            cands.append((0 if trivial else 1, len(opposing), gap, eid, a, b, v, opposing, trivial))
        cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
        out, seen = [], set()
        for _info, n_opp, gap, eid, a, b, v, opposing, trivial in cands:
            if eid in seen:
                continue  # one example per group → more varied evidence
            seen.add(eid)
            pa, pb = groups[eid][a], groups[eid][b]
            out.append({
                "event_id": eid,
                "event_short": eid.split(":")[-1],
                "run_a": a, "run_b": b,
                "target_value": round(v[target], 4),
                "opposing": opposing,
                "trivial": trivial,
                "vector": {m: round(v[m], 4) for m in fw.ORDER},
                "a": side_payload(pa, max_chars),
                "b": side_payload(pb, max_chars),
            })
            if len(out) >= per_cell:
                break
        return out

    return {
        "similar_but_differ": collect(True),
        "differ_but_similar": collect(False),
    }


def build_model(run_dir: str, hi, lo, per_cell, max_chars) -> dict:
    groups = fw.load_groups(run_dir)
    rows = fw.build_rows(groups)
    examples = {
        m: find_examples(rows, groups, m, hi, lo, per_cell, max_chars)
        for m in fw.ORDER
    }
    return {
        "label": model_label(run_dir),
        "source": run_dir,
        "n_groups": len(groups),
        "n_pairs": len(rows),
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", help="Timestamped run dirs (one per model tab)")
    ap.add_argument("--hi", type=float, default=0.85, help="'similar' cutoff (default 0.85)")
    ap.add_argument("--lo", type=float, default=0.60, help="'differ' cutoff (default 0.60)")
    ap.add_argument("--per-cell", type=int, default=3, help="Max examples per metric/direction")
    ap.add_argument("--max-chars", type=int, default=1800,
                    help="Truncate embedded content/args to this many chars")
    ap.add_argument("--out", default=os.path.join(HERE, "metric_viewer.html"),
                    help="Output HTML path")
    ap.add_argument("--data", default=None, help="Also write the intermediate JSON here")
    args = ap.parse_args()

    if args.lo > args.hi:
        ap.error("--lo must be <= --hi")

    models = []
    for rd in args.run_dirs:
        print(f"→ {rd}")
        m = build_model(rd, args.hi, args.lo, args.per_cell, args.max_chars)
        print(f"    {m['label']}: {m['n_groups']} groups, {m['n_pairs']} pairs")
        models.append(m)

    payload = {
        "thresholds": {"hi": args.hi, "lo": args.lo},
        "per_cell": args.per_cell,
        "metrics": [
            {"key": k, "label": METRIC_LABEL[k], "gloss": fw.METRIC_INFO[k][0]}
            for k in fw.ORDER
        ],
        "models": models,
    }
    data = json.dumps(payload, ensure_ascii=False)

    if args.data:
        with open(args.data, "w") as f:
            f.write(data)
        print(f"✓ wrote data {args.data}")

    if not os.path.exists(TEMPLATE):
        print(f"template not found: {TEMPLATE}", file=sys.stderr)
        return 1
    tpl = open(TEMPLATE).read()
    if "__DATA__" not in tpl:
        print("template missing __DATA__ placeholder", file=sys.stderr)
        return 1
    # Escape "</" so embedded JSON can't close the <script> early (lossless for JSON.parse).
    html = tpl.replace("__DATA__", data.replace("</", "<\\/"))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"✓ wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")
    print(f"  open it: open {args.out}")
    return 0


# Short labels for the sidebar (the analyzer keys are verbose). Gloss comes from
# find_metric_witnesses.METRIC_INFO so the description stays in one place.
METRIC_LABEL = {
    "exact_match": "Exact match",
    "content_levenshtein": "Content Levenshtein",
    "content_jaccard": "Content Jaccard",
    "tool_seq_lcs": "Tool seq (LCS)",
    "tss_editdist": "TSS (edit-dist)",
    "tool_set_overlap": "Tool set overlap",
    "tool_args_consistency": "Tool args (AC)",
    "tool_calls_exact": "Tool calls exact",
    "tool_calls_ordered_dedup": "Tool calls (ordered, dedup)",
    "js_kernel": "JS kernel",
    "gak_ordering": "GAK ordering",
    "finish_reason_agreement": "Finish reason",
}


if __name__ == "__main__":
    sys.exit(main())
