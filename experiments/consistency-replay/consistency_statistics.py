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

"""Paper-grounded consistency analysis of repeated agent replays.

This is a companion to analyze_consistency.py. Where that script pools ALL outputs of an
identical input into one group (run-anonymous) and reports point estimates, this script
KEEPS RUN IDENTITY so it can compare pairs of repeated runs, and it reports the metrics
proposed in two 2026 papers, with their statistical machinery:

  Yagubyan, "How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in
  Multi-Step Tool-Calling Pipelines" (arXiv:2605.28840):
    - TSS (Tool Sequence Similarity, Def. 3): pairwise normalized-Levenshtein similarity
      over tool-NAME sequences.
    - AC  (Argument Consistency, Def. 4): Jaccard over step-aligned flattened {(k,v)}
      sets; 0 when two traces call different tools at a step.
    - Unique Sequences, Divergence Point, Output Agreement.
    - Hypothesis 1: E[TSS] >> E[AC] (structure stable, arguments vary).

  Raj et al., "Consistency as a Testable Property: Statistical Methods to Evaluate AI
  Agent Reliability" (arXiv:2605.10516):
    - Consistency as a U-statistic: theta = E[k(Y,Y')], estimated by
      U_n = C(n,2)^-1 * sum_{i<j} k(y_i, y_j), aggregated across M instances into a mean
      Ubar with a t-based confidence interval Ubar +/- t_{M-1,1-a/2} * sigma_hat/sqrt(M).
    - Trajectory-level MMD two-sample test between conditions, with a Jensen-Shannon
      kernel over per-trajectory action histograms (composition) and a Global Alignment
      Kernel over ordered tool sequences (ordering).

Levels of output:
  1. Run x Run matrices  — per condition, every pair of repeated runs compared over the
                           (trace, call-position) keys they share.
  2. U-statistic theta   — off-diagonal pairwise kernel values ARE the U-statistic
     + confidence interval  summands; reported per condition with a t-based CI.
  3. Cross-condition MMD — when >=2 conditions are supplied, a two-sample trajectory test
                           per condition pair (optional permutation p-value).
  4. Per-trace pool      — the familiar analyze_consistency.py roll-up, retained for
                           continuity, plus the H1 (TSS vs AC) check.

Usage:
  consistency_statistics.py --condition NAME=DIR [--condition NAME2=DIR2 ...] \
      [--kernel exact|levenshtein|judge] [--alpha 0.05] [--perm 0] [--seed 41] \
      [--judge] [--out analysis_papers.json]

A single bare positional DIR is accepted as `--condition default=DIR`. Each DIR is
scanned for run_* subdirectories, exactly like analyze_consistency.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Per-pair metric primitives come from compare/ (the single source of truth); this module
# owns only the pairwise aggregation (Run x Run matrices, U-statistic theta/CI, MMD). The
# IO / grouping / judge helpers still live in analyze_consistency.py. Add our own dir to
# sys.path so both imports resolve regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import (  # noqa: E402
    collapse_ws,
    jaccard,
    parse_response,
    tool_args_signature,
    fast_content_ratio,
    tss,
    argument_consistency,
    action_histogram,
    js_kernel,
    js_divergence,
    global_alignment_kernel,
)
from analyze_consistency import (  # noqa: E402
    find_run_dirs,
    judge_group,
    event_key,
    load_run,
)


# ------------------------------------------------------------------ trajectories

# A "run" is one full replay, stored as a run_* dir. For a given (trace, call-position)
# key, a run holds exactly one response record. We line up the SAME key across two runs
# to compare them — that is what "compare two repeated runs" means.
#
# parse_response + json.loads are expensive and every comparison touches the same records
# many times (once per run-pair). So we parse each record EXACTLY ONCE at load time into a
# lightweight Feature struct and every downstream comparison reads cached fields.


class Feature:
    """Precomputed, comparison-ready view of one response record.

    ok:         usable (non-error) response?
    has_output: ok AND carries actual content or tool calls — distinguishes a real answer
                from an empty completion (e.g. a length-truncated reasoning turn). Used to
                stop empty-vs-empty pairs from scoring 1.0 by vacuous emptiness.
    content:    final natural-language text.
    signature:  content + canonical tool-args — the exact-match unit (None if not ok).
    names:      ordered tuple of tool NAMES in this turn (Yagubyan structural layer).
    kv:         frozenset of (tool.key, canonical-value) — the AC unit (Def. 4).
    hist:       action histogram over names — the JS-kernel unit.
    """

    __slots__ = ("ok", "has_output", "content", "signature", "names", "kv", "hist")

    def __init__(self, record: dict):
        p = parse_response(record)
        self.ok = p["ok"]
        self.has_output = p["has_output"]
        self.content = p["content"]
        names: List[str] = []
        kv: set = set()
        for tc in p["tool_calls"]:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = fn.get("name") or "<unnamed>"
            names.append(name)
            args_raw = fn.get("arguments")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            if isinstance(args, dict):
                for k, v in args.items():
                    # Namespace keys by tool so identical arg names on different tools
                    # don't overlap; canonicalize the value so ordering can't fork it.
                    kv.add((f"{name}.{k}", json.dumps(v, sort_keys=True, ensure_ascii=False)))
        self.names: Tuple[str, ...] = tuple(names)
        self.kv = frozenset(kv)
        self.hist = action_histogram(self.names)
        self.signature: Optional[str] = None
        if self.ok:
            # Whitespace-collapsed content + canonical tool args, so exact-match ignores
            # pure formatting differences (matches export_viewer_data's sig()).
            self.signature = collapse_ws(self.content) + "\x00" + json.dumps(
                tool_args_signature(p["tool_calls"]), ensure_ascii=False
            )


RunMap = Dict[str, Feature]  # event_id -> features


def load_condition(base_dir: str) -> Dict[str, RunMap]:
    """Load one condition: {run_id: {event_id: Feature}}.

    run_id is the run_* directory name. event_id is stable across runs for a given
    (trace, call-position), so the same key in two runs was fed byte-identical input.
    """
    runs: Dict[str, RunMap] = {}
    for rd in find_run_dirs(base_dir):
        run_id = os.path.basename(rd)
        rm: RunMap = {}
        for rec in load_run(rd):
            key = event_key(rec)
            # A run should hit each identical input once; if a session re-issues the exact
            # same payload we keep the first (deterministic, order-stable).
            if key not in rm:
                rm[key] = Feature(rec)
        if rm:
            runs[run_id] = rm
    return runs


# The kernels (js_divergence, action_histogram, js_kernel, global_alignment_kernel) and the
# paper primitives (tss, argument_consistency, fast_content_ratio) are imported from compare/
# above. Feature.kv already holds the flattened {(k,v)} AC set, so a step-pair's AC is just
# argument_consistency(fa.kv, fb.kv); a pair's TSS is tss(fa.names, fb.names).


# --------------------------------------------------------- run-vs-run comparison


def compare_runs(run_a: RunMap, run_b: RunMap) -> Dict[str, Any]:
    """Compare two repeated runs over the (trace, call-position) keys they share.

    Returns per-metric means over the shared, usable keys plus the shared-key count.

    Aggregation is CHANNEL-AWARE: each metric is averaged only over the pairs that actually
    exercise the channel it measures, so a metric never has its mean padded by pairs where
    the thing it scores was absent on BOTH sides:
      - content metrics (levenshtein, jaccard) — only pairs where at least one side produced
        prose. Most empty-content pairs are tool turns (output went to the tool channel, not
        prose); scoring ('','') as 1.0 content-agreement there would inflate the headline.
      - tool metrics (tss, gak, js_kernel) — only pairs where at least one side called a tool.
      - ac — only pairs where at least one side had tool arguments (already, via None).
    This mirrors how `ac` has always been aggregated (`n_ac_steps`); the coverage counts
    n_content_steps / n_tool_steps report how many pairs each family was actually meaned over.

    Additionally, a `has_output` GUARD: when exactly one side produced output and the other
    was empty (a length-truncated / stalled turn), the two sides genuinely DIVERGED. Left
    alone their empty-vs-nonempty content would still score via the metrics, but the
    empty-vs-empty *content* sub-case would read as 1.0 — vacuous agreement masking a real
    behavioral difference. So such mixed pairs are forced to 0.0 on every similarity metric.
    """
    shared = sorted(set(run_a) & set(run_b))
    exact_hits = 0
    lev_vals: List[float] = []
    jac_vals: List[float] = []
    tss_vals: List[float] = []
    ac_vals: List[float] = []
    js_vals: List[float] = []
    gak_vals: List[float] = []
    n_usable = 0

    for key in shared:
        fa, fb = run_a[key], run_b[key]
        if not fa.ok or not fb.ok:
            continue  # an errored record on either side — skip the pair
        n_usable += 1

        exact_hits += 1 if fa.signature == fb.signature else 0

        # has_output guard: one side acted, the other emitted nothing — a divergence, not
        # agreement. Force every similarity metric to 0 so the empty side can't score 1.0
        # by vacuous emptiness on any channel.
        mixed_output = fa.has_output != fb.has_output

        # Content channel: at least one side produced prose (else there's no text to compare).
        if fa.content.strip() or fb.content.strip():
            if mixed_output:
                lev_vals.append(0.0)
                jac_vals.append(0.0)
            else:
                lev_vals.append(fast_content_ratio(fa.content, fb.content))
                jac_vals.append(jaccard(fa.content, fb.content))

        # Tool channel: at least one side called a tool (else there's no sequence to compare).
        if fa.names or fb.names:
            if mixed_output:
                tss_vals.append(0.0)
                js_vals.append(0.0)
                gak_vals.append(0.0)
            else:
                tss_vals.append(tss(fa.names, fb.names))
                js_vals.append(js_kernel(fa.hist, fb.hist))
                gak_vals.append(global_alignment_kernel(fa.names, fb.names))

        # Argument consistency: undefined (None) when neither side had args; excluded then.
        ac = argument_consistency(fa.kv, fb.kv)
        if ac is not None:
            ac_vals.append(0.0 if mixed_output else ac)

    def mean(vals: List[float]) -> Optional[float]:
        return statistics.mean(vals) if vals else None

    return {
        "n_shared": len(shared),
        "n_usable": n_usable,
        "output_exact_match": (exact_hits / n_usable) if n_usable else None,
        "output_levenshtein": mean(lev_vals),
        "output_jaccard": mean(jac_vals),
        "n_content_steps": len(lev_vals),
        "tss": mean(tss_vals),
        "js_kernel": mean(js_vals),
        "gak": mean(gak_vals),
        "n_tool_steps": len(tss_vals),
        "ac": mean(ac_vals),
        "n_ac_steps": len(ac_vals),
    }


def run_pair_matrix(runs: Dict[str, RunMap]) -> Dict[str, Any]:
    """Build R x R symmetric matrices over all run pairs, for each metric."""
    run_ids = sorted(runs.keys())
    METRICS = [
        "output_exact_match",
        "output_levenshtein",
        "output_jaccard",
        "tss",
        "ac",
        "js_kernel",
        "gak",
    ]
    # matrices[metric][i][j]
    matrices: Dict[str, List[List[Optional[float]]]] = {
        m: [[None] * len(run_ids) for _ in run_ids] for m in METRICS
    }
    pair_records: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for (i, ri), (j, rj) in combinations(enumerate(run_ids), 2):
        cmp = compare_runs(runs[ri], runs[rj])
        pair_records[(ri, rj)] = cmp
        for m in METRICS:
            v = cmp[m]
            matrices[m][i][j] = v
            matrices[m][j][i] = v
    for m in METRICS:  # diagonal: a run vs itself is perfectly consistent for every metric
        for i in range(len(run_ids)):
            matrices[m][i][i] = 1.0

    # Per-run "distance to pack": 1 - mean similarity to the other runs (higher = more of
    # an outlier). Uses output_levenshtein as the summary similarity.
    outlier: Dict[str, float] = {}
    for i, ri in enumerate(run_ids):
        sims = [matrices["output_levenshtein"][i][j] for j in range(len(run_ids)) if j != i]
        sims = [s for s in sims if s is not None]
        outlier[ri] = (1.0 - statistics.mean(sims)) if sims else 0.0

    return {
        "run_ids": run_ids,
        "metrics": METRICS,
        "matrices": matrices,
        "pairs": {f"{a}|{b}": v for (a, b), v in pair_records.items()},
        "outlier_score": outlier,
    }


# ------------------------------------------------------------ U-statistic + CI

# Two-sided t critical values for common alphas, df 1..30, then a normal-approx tail.
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}
_T_TABLE_99 = {
    1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499, 8: 3.355,
    9: 3.250, 10: 3.169, 11: 3.106, 12: 3.055, 13: 3.012, 14: 2.977, 15: 2.947,
    16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861, 20: 2.845, 21: 2.831, 22: 2.819,
    23: 2.807, 24: 2.797, 25: 2.787, 26: 2.779, 27: 2.771, 28: 2.763, 29: 2.756,
    30: 2.750,
}


def t_critical(df: int, alpha: float) -> float:
    """Two-sided t critical value. Table for the two common alphas up to df=30, else the
    normal approximation (t -> z as df grows, and our M is usually small so the table
    covers the realistic cases)."""
    if df < 1:
        return float("inf")
    table = _T_TABLE_95 if abs(alpha - 0.05) < 1e-9 else (_T_TABLE_99 if abs(alpha - 0.01) < 1e-9 else None)
    if table is not None and df <= 30:
        return table[df]
    # Normal approx of the two-sided quantile (Acklam-free, adequate here): z for common
    # alphas, else invert via a rational approximation.
    z = {0.05: 1.959964, 0.01: 2.575829, 0.10: 1.644854}.get(round(alpha, 2))
    if z is None:
        z = _inv_norm(1.0 - alpha / 2.0)
    return z


def _inv_norm(p: float) -> float:
    """Inverse standard-normal CDF (Beasley-Springer/Moro), for uncommon alphas."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def output_kernel(fa: "Feature", fb: "Feature", kind: str) -> Optional[float]:
    """Similarity kernel k(y,y') in [0,1] with k(y,y)=1, used for the U-statistic.

    - exact:       1 if the (content + canonical tool-args) signatures match, else 0.
    - levenshtein: graded content similarity (compare.fast_content_ratio — difflib ratio).

    Returns None when the kernel does not apply, and u_statistic skips those pairs (so they
    don't pad theta). Two cases return None:
      - either record is unusable (errored);
      - for the CONTENT kernel (levenshtein), neither side produced prose — there is no text
        to compare. Most such pairs are tool turns (output went to the tool channel); scoring
        ('','') as 1.0 content-agreement would inflate theta exactly as it did the matrix mean.
        This mirrors the channel-aware aggregation in compare_runs.

    The `exact` kernel needs no channel gate: it compares the whole response signature, so a
    mixed (one side acted, one stalled) pair already differs -> 0.0, and two genuinely empty
    responses are genuinely identical -> 1.0. Only the content kernel gets the guard below."""
    if not fa.ok or not fb.ok:
        return None
    if kind == "exact":
        return 1.0 if fa.signature == fb.signature else 0.0
    if kind == "levenshtein":
        if not fa.content.strip() and not fb.content.strip():
            return None  # no prose on either side — content kernel does not apply
        if fa.has_output != fb.has_output:
            return 0.0   # one side acted, the other emitted nothing — a divergence, not 1.0
        return fast_content_ratio(fa.content, fb.content)
    raise ValueError(f"unknown kernel kind: {kind}")


def u_statistic(
    runs: Dict[str, RunMap],
    kind: str,
    alpha: float,
    judge_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Consistency U-statistic theta with a t-based CI across instances (Raj et al.).

    Each INSTANCE is a (trace, call-position) key. For that key we take the outputs from
    every run and form U_n = C(n,2)^-1 * sum_{i<j} k(y_i, y_j) — the mean pairwise kernel.
    Then across the M instances we report the mean Ubar, sample stdev, and the t-CI
    Ubar +/- t_{M-1,1-alpha/2} * sigma_hat / sqrt(M).
    """
    # Gather, per instance key, the list of usable Features across runs.
    per_key: Dict[Tuple[str, str], List["Feature"]] = defaultdict(list)
    for rm in runs.values():
        for key, feat in rm.items():
            if feat.ok:
                per_key[key].append(feat)

    # The judge kernel needs the group's raw contents; it clusters once per divergent
    # group and reports the cluster count, while the U-statistic pair values fall back to
    # exact-match agreement (the judge gives counts, not pairwise assignments).
    effective_kind = "exact" if kind == "judge" else kind
    instance_u: List[float] = []
    judge_clusters: List[int] = []
    for key, feats in per_key.items():
        if len(feats) < 2:
            continue
        vals: List[float] = []
        for a, b in combinations(feats, 2):
            v = output_kernel(a, b, effective_kind)
            if v is not None:
                vals.append(v)
        if vals:
            instance_u.append(statistics.mean(vals))
        if kind == "judge" and judge_cfg is not None:
            non_empty = [f.content for f in feats if f.content.strip()]
            if len(non_empty) >= 2 and len(set(non_empty)) > 1:
                verdict = judge_group(non_empty, judge_cfg)
                if isinstance(verdict.get("clusters"), int):
                    judge_clusters.append(verdict["clusters"])

    judge_summary = None
    if kind == "judge" and judge_clusters:
        judge_summary = {
            "n_judged": len(judge_clusters),
            "mean_clusters": statistics.mean(judge_clusters),
            "max_clusters": max(judge_clusters),
        }

    m = len(instance_u)
    if m == 0:
        return {"kernel": kind, "n_instances": 0, "theta": None,
                "note": "no comparable instances", "judge": judge_summary}
    theta = statistics.mean(instance_u)
    if m == 1:
        return {
            "kernel": kind, "n_instances": 1, "theta": theta,
            "ci_low": None, "ci_high": None, "note": "single instance; CI undefined",
            "judge": judge_summary,
        }
    sd = statistics.stdev(instance_u)  # sample stdev (M-1 denominator)
    se = sd / math.sqrt(m)
    tc = t_critical(m - 1, alpha)
    half = tc * se
    return {
        "kernel": kind,
        "n_instances": m,
        "theta": theta,
        "stdev": sd,
        "std_error": se,
        "alpha": alpha,
        "t_crit": tc,
        "ci_low": max(0.0, theta - half),
        "ci_high": min(1.0, theta + half),
        "judge": judge_summary,
    }


def _aggregate_u(instance_u: List[float], alpha: float) -> Dict[str, Any]:
    """Aggregate per-instance U values into theta + a t-based CI across the M instances.

    Shared by the output U-statistic and the trajectory U-statistic: mean Ubar, sample
    stdev, and Ubar +/- t_{M-1,1-alpha/2} * sigma_hat / sqrt(M), clamped to [0,1]."""
    m = len(instance_u)
    if m == 0:
        return {"n_instances": 0, "theta": None, "ci_low": None, "ci_high": None,
                "note": "no comparable instances"}
    theta = statistics.mean(instance_u)
    if m == 1:
        return {"n_instances": 1, "theta": theta, "ci_low": None, "ci_high": None,
                "note": "single instance; CI undefined"}
    sd = statistics.stdev(instance_u)
    se = sd / math.sqrt(m)
    tc = t_critical(m - 1, alpha)
    half = tc * se
    return {
        "n_instances": m, "theta": theta, "stdev": sd, "std_error": se,
        "alpha": alpha, "t_crit": tc,
        "ci_low": max(0.0, theta - half), "ci_high": min(1.0, theta + half),
    }


def trajectory_u_statistic(runs: Dict[str, RunMap], alpha: float) -> Dict[str, Any]:
    """Trajectory-level consistency U-statistic theta with a t-CI (Raj et al., Eq. 1/3).

    Same all-pairs construction as `u_statistic`, but the kernel acts on the tool-call
    *trajectory* of each repetition rather than its text output. Under Assumption 1
    (x_mi = x_m0) the N repetitions of one instance are a valid sample, so no base-vs-
    perturbed split is needed:

        U_n = C(n,2)^-1 * sum_{i<j} k(tau_i, tau_j)   per instance (Eq. 1)
        theta = mean_m U_n                            across M instances (Eq. 3)

    Two kernels, both normalized to [0,1] with k(tau,tau)=1:
      - js:  Jensen-Shannon kernel over action histograms  -> action *composition*
      - gak: Global Alignment Kernel over name sequences    -> action *ordering*
    """
    per_key: Dict[Tuple[str, str], List["Feature"]] = defaultdict(list)
    for rm in runs.values():
        for key, feat in rm.items():
            if feat.ok:
                per_key[key].append(feat)

    js_u: List[float] = []
    gak_u: List[float] = []
    for feats in per_key.values():
        if len(feats) < 2:
            continue
        js_vals: List[float] = []
        gak_vals: List[float] = []
        for a, b in combinations(feats, 2):
            js_vals.append(js_kernel(a.hist, b.hist))
            gak_vals.append(global_alignment_kernel(a.names, b.names))
        if js_vals:
            js_u.append(statistics.mean(js_vals))
        if gak_vals:
            gak_u.append(statistics.mean(gak_vals))

    return {
        "js": {"kernel": "js", **_aggregate_u(js_u, alpha)},
        "gak": {"kernel": "gak", **_aggregate_u(gak_u, alpha)},
    }


# --------------------------------------------------------------- cross-condition MMD


def condition_trajectories(runs: Dict[str, RunMap], key: str) -> List[Tuple[str, ...]]:
    """All runs' tool-name sequences for one event_id (call-position) key in a condition."""
    seqs = []
    for rm in runs.values():
        feat = rm.get(key)
        if feat is not None and feat.ok:
            seqs.append(feat.names)
    return seqs


def _mmd2_unbiased(
    xs: List[Tuple[str, ...]], ys: List[Tuple[str, ...]], kfn: Callable
) -> Optional[float]:
    """Unbiased two-sample MMD^2 estimator for one instance.

    MMD^2_u = 1/(n(n-1)) sum_{i!=j} k(x_i,x_j) + 1/(m(m-1)) sum_{i!=j} k(y_i,y_j)
              - 2/(nm) sum_{i,j} k(x_i,y_j)
    """
    n, m = len(xs), len(ys)
    if n < 2 or m < 2:
        return None

    def cross(a, b):
        return sum(kfn(x, y) for x in a for y in b)

    def within(a):
        s = 0.0
        for i in range(len(a)):
            for j in range(len(a)):
                if i != j:
                    s += kfn(a[i], a[j])
        return s

    term_x = within(xs) / (n * (n - 1))
    term_y = within(ys) / (m * (m - 1))
    term_xy = cross(xs, ys) / (n * m)
    return term_x + term_y - 2 * term_xy


def _kernel_js(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    return js_kernel(action_histogram(a), action_histogram(b))


def cross_condition_mmd(
    conditions: Dict[str, Dict[str, RunMap]],
    perm: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """For each condition pair, aggregate per-instance unbiased MMD^2 over shared keys,
    under both the JS (composition) and GAK (ordering) kernels, with an optional
    permutation p-value."""
    names = sorted(conditions.keys())
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    for na, nb in combinations(names, 2):
        ca, cb = conditions[na], conditions[nb]
        # Instances = keys present (with >=2 usable trajectories) in BOTH conditions.
        keys_a = {k for rm in ca.values() for k in rm}
        keys_b = {k for rm in cb.values() for k in rm}
        shared_keys = sorted(keys_a & keys_b)

        for kname, kfn in (("js", _kernel_js), ("gak", global_alignment_kernel)):
            per_instance: List[Tuple[List, List, float]] = []
            for key in shared_keys:
                xs = condition_trajectories(ca, key)
                ys = condition_trajectories(cb, key)
                v = _mmd2_unbiased(xs, ys, kfn)
                if v is not None:
                    per_instance.append((xs, ys, v))
            if not per_instance:
                results.append({
                    "condition_a": na, "condition_b": nb, "kernel": kname,
                    "mmd2": None, "note": "no shared instances with >=2 trajectories each",
                })
                continue

            mmd2 = statistics.mean(v for _, _, v in per_instance)
            entry = {
                "condition_a": na, "condition_b": nb, "kernel": kname,
                "n_instances": len(per_instance), "mmd2": mmd2,
            }
            if perm > 0:
                ge = 0
                for _ in range(perm):
                    perm_vals = []
                    for xs, ys, _v in per_instance:
                        pool = list(xs) + list(ys)
                        rng.shuffle(pool)
                        px, py = pool[: len(xs)], pool[len(xs):]
                        pv = _mmd2_unbiased(px, py, kfn)
                        if pv is not None:
                            perm_vals.append(pv)
                    if perm_vals and statistics.mean(perm_vals) >= mmd2:
                        ge += 1
                entry["perm"] = perm
                entry["p_value"] = (ge + 1) / (perm + 1)  # add-one smoothing
            results.append(entry)
    return results


# ------------------------------------------------------------- H1 (TSS vs AC)


def h1_check(matrix: Dict[str, Any]) -> Dict[str, Any]:
    """Yagubyan Hypothesis 1: E[TSS] >> E[AC]. Average the off-diagonal run-pair values."""
    def off_diag_mean(metric: str) -> Optional[float]:
        vals = [v for v in matrix["pairs"].values() if v.get(metric) is not None]
        xs = [v[metric] for v in vals]
        return statistics.mean(xs) if xs else None

    tss = off_diag_mean("tss")
    ac = off_diag_mean("ac")
    gap = (tss - ac) if (tss is not None and ac is not None) else None
    return {
        "mean_tss": tss,
        "mean_ac": ac,
        "gap": gap,
        "supports_h1": (gap is not None and gap > 0),
    }


# ----------------------------------------------------------------------- report


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def print_report(analysis: Dict[str, Any]) -> None:
    print("\n" + "=" * 74)
    print("PAPER-GROUNDED AGENT CONSISTENCY")
    print("=" * 74)

    for name, cond in analysis["conditions"].items():
        mat = cond["run_pair_matrix"]
        rids = mat["run_ids"]
        print(f"\nCondition: {name}   ({len(rids)} runs: {', '.join(rids)})")
        print("-" * 74)

        # U-statistic
        u = cond["u_statistic"]
        if u.get("theta") is None:
            print(f"  U-statistic theta ({u['kernel']}): n/a — {u.get('note','')}")
        elif u.get("ci_low") is None:
            print(f"  U-statistic theta ({u['kernel']}): {u['theta']:.3f}  ({u.get('note','')})")
        else:
            print(
                f"  U-statistic theta ({u['kernel']}): {u['theta']:.3f}  "
                f"[{u['ci_low']:.3f}, {u['ci_high']:.3f}]  "
                f"(M={u['n_instances']} instances, {int((1-u['alpha'])*100)}% CI)"
            )

        # Trajectory U-statistic (same Eq. 1/3 all-pairs construction, tool-call kernels)
        traj = cond.get("trajectory_u_statistic")
        if traj:
            for kind, label in (("js", "composition"), ("gak", "ordering")):
                t = traj.get(kind, {})
                if t.get("theta") is None:
                    print(f"  Trajectory theta [{kind}/{label}]: n/a — {t.get('note','')}")
                elif t.get("ci_low") is None:
                    print(f"  Trajectory theta [{kind}/{label}]: {t['theta']:.3f}  ({t.get('note','')})")
                else:
                    print(
                        f"  Trajectory theta [{kind}/{label}]: {t['theta']:.3f}  "
                        f"[{t['ci_low']:.3f}, {t['ci_high']:.3f}]  (M={t['n_instances']} instances)"
                    )

        # H1
        h = cond["h1"]
        arrow = ">>" if h["supports_h1"] else "<= "
        print(
            f"  Hypothesis 1 (E[TSS] {arrow} E[AC]):  "
            f"TSS={_fmt(h['mean_tss'])}  AC={_fmt(h['mean_ac'])}  "
            f"gap={_fmt(h['gap'])}  -> {'supports H1' if h['supports_h1'] else 'does NOT support H1'}"
        )

        # Aggregate off-diagonal means per metric
        print("  Mean over all run pairs:")
        for m in mat["metrics"]:
            vals = [v[m] for v in mat["pairs"].values() if v.get(m) is not None]
            if vals:
                print(f"    {m:22} {statistics.mean(vals):.3f}   (min pair {min(vals):.3f})")

        # Outliers
        outl = sorted(mat["outlier_score"].items(), key=lambda kv: -kv[1])
        worst = outl[0] if outl else None
        if worst and worst[1] > 0:
            print(f"  Most outlying run: {worst[0]} (distance-to-pack {worst[1]:.3f})")

    # Cross-condition MMD
    mmd = analysis.get("mmd")
    print("\n" + "-" * 74)
    if not mmd:
        n = len(analysis["conditions"])
        print(f"Cross-condition MMD: n/a (need >=2 conditions; have {n}).")
    else:
        print("Cross-condition trajectory MMD^2  (0 = indistinguishable distributions)")
        for e in mmd:
            base = f"  {e['condition_a']} vs {e['condition_b']}  [{e['kernel']}]:"
            if e.get("mmd2") is None:
                print(f"{base} n/a — {e.get('note','')}")
                continue
            line = f"{base} MMD^2={e['mmd2']:+.4f}  (n={e['n_instances']})"
            if "p_value" in e:
                line += f"  p={e['p_value']:.3f} ({e['perm']} perms)"
            print(line)
    print("=" * 74)


# ------------------------------------------------------------------------ driver


def parse_conditions(args: argparse.Namespace) -> Dict[str, str]:
    """Resolve --condition NAME=DIR entries (and a bare positional dir) to {name: dir}."""
    conds: Dict[str, str] = {}
    for spec in args.condition or []:
        if "=" not in spec:
            print(f"error: --condition must be NAME=DIR, got {spec!r}", file=sys.stderr)
            sys.exit(2)
        name, _, d = spec.partition("=")
        conds[name] = d
    if args.base_dir:
        conds.setdefault("default", args.base_dir)
    return conds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_dir", nargs="?", help="Bare dir, treated as --condition default=DIR")
    ap.add_argument(
        "--condition", action="append", metavar="NAME=DIR",
        help="A labelled condition (a dir of run_* subdirs). Repeatable.",
    )
    ap.add_argument("--kernel", choices=["exact", "levenshtein", "judge"], default="exact",
                    help="Similarity kernel for the U-statistic (default exact).")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI significance level (default 0.05).")
    ap.add_argument("--perm", type=int, default=0, help="MMD permutation-test rounds (0 = off).")
    ap.add_argument("--seed", type=int, default=41, help="Seed for permutation shuffles.")
    ap.add_argument("--judge", action="store_true", help="Enable LLM judge (kernel=judge).")
    ap.add_argument("--out", default=None, help="Write full JSON analysis here.")
    args = ap.parse_args()

    conds = parse_conditions(args)
    if not conds:
        print("error: supply at least one --condition NAME=DIR or a bare directory.", file=sys.stderr)
        return 2

    # Judge config (only used when kernel=judge). Mirrors analyze_consistency.py: the key
    # comes from RITS_API_KEY so no secret lives in the repo.
    kernel = "judge" if args.judge else args.kernel
    judge_cfg = None
    if kernel == "judge":
        judge_key = os.environ.get("RITS_API_KEY", "")
        if not judge_key:
            print("warning: kernel=judge but RITS_API_KEY is empty; judge calls will fail",
                  file=sys.stderr)
        judge_cfg = {
            "url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/"
            "qwen3-vl-235b-a22b-instruct/v1/chat/completions",
            "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "headers": {"RITS_API_KEY": judge_key, "Content-Type": "application/json"},
        }

    conditions_runs: Dict[str, Dict[str, RunMap]] = {}
    for name, d in conds.items():
        runs = load_condition(d)
        if not runs:
            print(f"warning: condition {name!r} ({d}) has no usable run_* dirs — skipping",
                  file=sys.stderr)
            continue
        conditions_runs[name] = runs
        print(f"Loaded condition {name!r}: {len(runs)} runs from {d}")

    if not conditions_runs:
        print("error: no conditions had usable data.", file=sys.stderr)
        return 1

    analysis: Dict[str, Any] = {"conditions": {}}
    for name, runs in conditions_runs.items():
        mat = run_pair_matrix(runs)
        analysis["conditions"][name] = {
            "run_pair_matrix": mat,
            "u_statistic": u_statistic(runs, kernel, args.alpha, judge_cfg),
            "trajectory_u_statistic": trajectory_u_statistic(runs, args.alpha),
            "h1": h1_check(mat),
        }

    if len(conditions_runs) >= 2:
        analysis["mmd"] = cross_condition_mmd(conditions_runs, args.perm, args.seed)

    print_report(analysis)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nFull analysis written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
