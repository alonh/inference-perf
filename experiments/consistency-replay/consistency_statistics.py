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

SCOPE — everything is computed WITHIN A SESSION. A session (one recorded trace / task) is
the unit of behavior: its events are the turns of one agent episode, and only repeated runs
OF THE SAME SESSION are comparable. Two consequences drive the whole design:

  * The U-statistic's INSTANCE is a session and its SAMPLE UNIT is a run. For session m with
    n runs, theta_m = C(n,2)^-1 * sum_{i<j} k(run_i, run_j | session m) — n=10 runs give 45
    pairs. Pooling events across sessions (the pre-fix behavior) event-weighted the estimate,
    so a 15-event session counted ~7x a 2-event one, and no per-session number was reported
    at all.
  * A TRAJECTORY is a session's whole action sequence — the tool names of event 000, then
    001, ... concatenated in event order. Applying the "trajectory" kernels (GAK, JS) to a
    single turn's tool list (the pre-fix behavior) made them degenerate: with almost every
    turn emitting 0 or 1 tool call, ordering and composition had nothing to measure.

Because the C(n,2) pairs of one session share runs (each run appears in n-1 of them) they are
NOT independent, so a t-interval over pairs understates the variance. Per-session intervals
therefore come from a delete-one-RUN jackknife. Across sessions the instances ARE independent,
so the session-mean keeps the paper's t-interval (Raj Eq. 3) with df = M-1 sessions.

Levels of output:
  1. Per session          — for each session: theta per metric over its C(n,2) run pairs,
     (the primary output)   a jackknife CI, an R x R run matrix, the H1 (TSS vs AC) check,
                            and an event-count agreement check across runs.
  2. Session mean         — sessions are equally weighted instances; the mean over sessions
                            carries a t-CI with df = M-1. This is the only cross-session
                            number, and it is a mean OF session thetas, never a pool of events.
  3. Event-scope output U — per-event output consistency (instance = one identical input),
                            retained as a separate, differently-scoped quantity.
  4. Cross-condition MMD — when >=2 conditions are supplied, a two-sample test over SESSION
                           trajectories per condition pair (optional permutation p-value).

Usage:
  consistency_statistics.py --condition NAME=DIR [--condition NAME2=DIR2 ...] \
      [--kernel exact|levenshtein|judge] [--alpha 0.05] [--perm 0] [--seed 41] \
      [--judge] [--out analysis_papers.json]

A single bare positional DIR is accepted as `--condition default=DIR`. Each DIR is an
experiment directory — `<base>/sessions/<session_id>/run_<i>/` — read via
`replay_parsing.iter_sessions`, so a condition arrives already grouped by session and this
module never does its own layout walk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Record reading and grouping come from replay_parsing; the per-pair metric primitives come
# from compare/ (each the single source of truth for its half). This module owns only the
# pairwise aggregation (Run x Run matrices, U-statistic theta/CI, MMD). The judge helper
# still lives in analyze_consistency.py. Add our own dir to sys.path so all three imports
# resolve regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_parsing import (  # noqa: E402
    Session,
    iter_sessions,
    parse_response,
    extract_tool_names,
)
from compare import (  # noqa: E402
    # Canonical units the metrics are taken over.
    tool_kv_set,
    response_signature,
    # Metrics.
    jaccard,
    fast_content_ratio,
    tss,
    argument_consistency,
    action_histogram,
    js_kernel,
    js_divergence,
    global_alignment_kernel,
)
from analyze_consistency import judge_group  # noqa: E402


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
    kv:         frozenset of (tool.key, canonical-value) — the AC unit (Def. 4), unioned
                over the turn's calls.
    calls_kv:   per-CALL kv frozensets, index-aligned with `names`. The session-level,
                step-aligned AC needs one set per call, not one per turn.
    hist:       action histogram over names — the JS-kernel unit.
    start_time: wall-clock start; used only as a tiebreak when ordering a session's events.
    """

    __slots__ = ("ok", "has_output", "content", "signature", "names", "kv", "calls_kv",
                 "hist", "start_time")

    def __init__(self, record: dict):
        # Every derived field comes from replay_parsing.py / compare/signatures.py, so this
        # struct is a cache of
        # the shared definitions rather than a second implementation of them.
        p = parse_response(record)
        self.ok = p["ok"]
        self.has_output = p["has_output"]
        self.content = p["content"]
        calls = p["tool_calls"]
        self.names: Tuple[str, ...] = extract_tool_names(calls)
        self.kv = tool_kv_set(calls)
        # One kv set per CALL, for step-aligned session AC. extract_tool_names drops unnamed
        # calls, so filter identically here to keep calls_kv index-aligned with names.
        self.calls_kv: Tuple[frozenset, ...] = tuple(
            tool_kv_set([tc]) for tc in calls if extract_tool_names([tc])
        )
        self.hist = action_histogram(self.names)
        # Whitespace-collapsed content + canonical tool args; None when not ok.
        self.signature: Optional[str] = response_signature(p)
        self.start_time = p["start_time"]


RunMap = Dict[str, Feature]  # event_id -> features

# event_id is "<session_id>:event_{i:03d}_{call_id}" — the ordinal after the ':' is the call
# position within the session, which is what puts a session's turns into replay order.
_EVENT_ORDINAL_RE = re.compile(r"^event_(\d+)")


def event_ordinal(event_id: str) -> Optional[int]:
    """Call position encoded in an event_id, or None when it carries no ordinal.

    Only the part after the ':' is inspected, so a session_id that itself contains "event_"
    cannot be mistaken for the ordinal.
    """
    tail = event_id.split(":", 1)[1] if ":" in event_id else event_id
    m = _EVENT_ORDINAL_RE.match(tail)
    return int(m.group(1)) if m else None


class SessionRun:
    """One session as replayed by ONE run: its events in order, plus its trajectory.

    event_ids: the session's event_ids in replay order (by ordinal, start_time as tiebreak).
    features:  event_id -> Feature.
    steps:     ((tool_name, kv), ...) — every tool call of the session, concatenated across
               its events in order. This is the step-aligned AC unit (Yagubyan Def. 4).
    names:     the SESSION TRAJECTORY — the steps' names. This is what the ordering (GAK) and
               composition (JS) kernels are meant to act on, and what TSS (Def. 3) compares.
               Applying them per TURN is the bug this class exists to fix: with almost every
               turn emitting 0 or 1 call, per-turn ordering/composition measure nothing.
    """

    __slots__ = ("session_id", "run_id", "event_ids", "features", "steps", "names", "hist",
                 "n_ok")

    def __init__(self, session_id: str, run_id: str, features: Dict[str, Feature]):
        self.session_id = session_id
        self.run_id = run_id
        self.features = features

        # Order by ordinal where present. Events without one sort last, by start_time, so a
        # corpus predating event_ids still yields a deterministic order.
        def order(eid: str) -> Tuple[int, int, float, str]:
            ordinal = event_ordinal(eid)
            if ordinal is not None:
                return (0, ordinal, 0.0, eid)
            st = features[eid].start_time
            return (1, 0, st if st is not None else 0.0, eid)

        self.event_ids: Tuple[str, ...] = tuple(sorted(features, key=order))

        steps: List[Tuple[str, frozenset]] = []
        for eid in self.event_ids:
            f = features[eid]
            if not f.ok:
                continue  # an errored turn contributes no actions to the trajectory
            for i, name in enumerate(f.names):
                steps.append((name, f.calls_kv[i] if i < len(f.calls_kv) else frozenset()))
        self.steps: Tuple[Tuple[str, frozenset], ...] = tuple(steps)
        self.names: Tuple[str, ...] = tuple(n for n, _ in steps)
        self.hist = action_histogram(self.names)
        self.n_ok = sum(1 for eid in self.event_ids if features[eid].ok)

    @property
    def n_events(self) -> int:
        return len(self.event_ids)


SessionMap = Dict[str, Dict[str, SessionRun]]  # session_id -> {run_id: SessionRun}


class Condition:
    """One labelled condition: its sessions, and the run ids they were replayed under.

    sessions: {session_id: {run_id: SessionRun}} — the session-scope view that every
              paper-grounded quantity is computed over.
    run_ids:  the union of run ids across sessions, in run order. Only an AXIS — the labels
              the R x R matrices are indexed by. There is deliberately no merged
              {run_id: all events of that run} view: a run id means "the i-th replay", and
              the i-th replay of session A shares nothing with the i-th replay of session B
              beyond the ordinal, so merging their events would invite exactly the
              cross-session pooling the module's SCOPE note rules out. The one quantity that
              genuinely wants that pool builds it explicitly, and says so
              (:func:`pooled_event_groups`).
    """

    __slots__ = ("name", "base_dir", "sessions", "run_ids")

    def __init__(self, name: str, base_dir: str, sessions: SessionMap, run_ids: Sequence[str]):
        self.name = name
        self.base_dir = base_dir
        self.sessions = sessions
        self.run_ids: Tuple[str, ...] = tuple(run_ids)


def load_condition(base_dir: str, name: str = "default") -> Condition:
    """Load one condition from an experiment directory, session by session.

    The layout read and the per-session (event, run) table come from
    :func:`replay_parsing.iter_sessions`; ``Feature`` is the per-record transform, so each
    record is parsed exactly once and every run pair below reads cached fields.

    A session_id is the ``sessions/<id>`` directory name and a run_id is its ``run_<i>``
    subdirectory. An event key is stable across runs for a given (session, call-position), so
    the same key in two runs was fed byte-identical input — that is what makes a run pair
    comparable at all.
    """
    sessions: SessionMap = {}
    run_ids: List[str] = []
    for session in iter_sessions(base_dir, Feature):
        by_run = {
            run_id: SessionRun(session.session_id, run_id, features)
            for run_id, features in session.by_run().items()
        }
        sessions[session.session_id] = by_run
        # Union in first-seen order, which is per-session run order (run_1 .. run_10,
        # numerically), so the matrix axes read in replay order rather than lexicographically.
        for run_id in session.runs:
            if run_id not in run_ids:
                run_ids.append(run_id)
    return Condition(name, base_dir, sessions, run_ids)


def pooled_event_groups(cond: Condition) -> Dict[str, List[Feature]]:
    """{event key: usable Features across runs}, POOLED OVER ALL SESSIONS.

    The one deliberate exception to the module's within-a-session scope, and the reason it is
    a named function rather than an inline regroup: the event-scope U-statistic's instance is
    "one identical input", and every event of every session is such an input. So this mixes
    sessions on purpose, and the number it feeds
    (:func:`u_statistic`) is reported as an explicitly event-scoped contrast to the headline —
    not as an alternative estimate of it. Its instances are not independent (the events of one
    session share a task and a prefix), which is exactly why it is not the headline.
    """
    groups: Dict[str, List[Feature]] = defaultdict(list)
    for by_run in cond.sessions.values():
        for sr in by_run.values():
            for key, feat in sr.features.items():
                if feat.ok:
                    groups[key].append(feat)
    return dict(groups)


# The kernels (js_divergence, action_histogram, js_kernel, global_alignment_kernel) and the
# paper primitives (tss, argument_consistency, fast_content_ratio) are imported from compare/
# above. Feature.kv already holds the flattened {(k,v)} AC set, so a step-pair's AC is just
# argument_consistency(fa.kv, fb.kv); a pair's TSS is tss(fa.names, fb.names).


# --------------------------------------------------------- run-vs-run comparison


def compare_runs(
    run_a: RunMap, run_b: RunMap, keys: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """Compare two repeated runs over the (session, call-position) keys they share.

    `keys` restricts the comparison to a subset of event_ids — pass one session's events to
    get that session's OUTPUT metrics. Omit it for the flat, all-events view. Restricting is
    what makes the session-scope numbers reuse this exact channel-aware logic instead of a
    second implementation of it.

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
    common = set(run_a) & set(run_b)
    if keys is not None:
        common &= set(keys)
    shared = sorted(common)
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


# ------------------------------------------------------ session-scope comparison

# The metrics reported at session scope. The tool-channel four (tss/ac/js_kernel/gak) are the
# paper metrics and are computed on the SESSION trajectory; the output three are per-event
# means within the session, via compare_runs.
SESSION_METRICS = [
    "output_exact_match",
    "output_levenshtein",
    "output_jaccard",
    "tss",
    "ac",
    "js_kernel",
    "gak",
    "trajectory_exact_match",
]


def step_aligned_ac(
    steps_a: Sequence[Tuple[str, frozenset]], steps_b: Sequence[Tuple[str, frozenset]]
) -> Optional[float]:
    """Argument Consistency (Yagubyan Def. 4) over two SESSION step sequences.

    Steps are aligned by position in the session's action sequence (not by turn), which is
    what Def. 4 specifies: per step, Jaccard over the flattened {(k,v)} argument sets, and 0
    when the two traces call DIFFERENT tools at that step.

    Folding rules, consistent with compare.argument_consistency's policy of record:
      - different tools at a step        -> 0.0 (a real divergence)
      - one trace has no step there      -> 0.0 (it stopped early / went longer: divergence)
      - same tool, neither side has args -> excluded from the mean (0/0, nothing compared)
    Returns None when no step was comparable at all.
    """
    n = max(len(steps_a), len(steps_b))
    if n == 0:
        return None
    vals: List[float] = []
    for i in range(n):
        if i >= len(steps_a) or i >= len(steps_b):
            vals.append(0.0)
            continue
        (name_a, kv_a), (name_b, kv_b) = steps_a[i], steps_b[i]
        if name_a != name_b:
            vals.append(0.0)
            continue
        v = argument_consistency(kv_a, kv_b)
        if v is not None:
            vals.append(v)
    return statistics.mean(vals) if vals else None


def session_pair(sa: SessionRun, sb: SessionRun) -> Dict[str, Any]:
    """Compare two runs OF THE SAME SESSION — the kernel k(run_i, run_j | session).

    Output channel: per-event means over the session's events, delegated to compare_runs
    (identical channel-aware and has_output rules as before).

    Tool channel: computed on the session TRAJECTORY, so the paper metrics measure what they
    were defined to measure —
      tss    Def. 3, edit distance over the session's tool-name sequence
      ac     Def. 4, step-aligned argument Jaccard over the session's steps
      js     composition of the session's action histogram
      gak    ordering of the session's action sequence
    The per-TURN values of the same four are retained under `turn_*` as diagnostics; they are
    what the pre-fix code reported as tss/ac/js_kernel/gak.

    The two channels treat a length mismatch differently, on purpose. Output metrics compare
    the events the two runs SHARE (there is no counterpart to compare a missing turn against),
    so a run that stopped after one turn can still score 1.0 on the turn it did produce. Tool
    metrics compare the FULL trajectories, so the missing steps count as divergence. Read
    together with n_events / n_steps that is the informative pair: "identical where both ran,
    but one ran far less". A session whose runs disagree on event count is flagged in the
    report for exactly this reason.
    """
    out = compare_runs(sa.features, sb.features)
    # Per-turn tool metrics are diagnostics, not the paper quantities — move them aside so
    # the session-scope values own the canonical names.
    for k in ("tss", "ac", "js_kernel", "gak"):
        out[f"turn_{k}"] = out.pop(k)

    # Tool channel, gated on the channel being exercised at all. When NEITHER run called a
    # tool in this whole session there is no trajectory to compare, so every tool metric is
    # undefined (None) rather than 1.0 — the same channel-aware rule compare_runs applies per
    # turn, and the same fold argument_consistency's policy of record fixes for AGGREGATION
    # surfaces. Scoring a tool-free session as tss=1.0 would be agreement-by-emptiness, and
    # since most sessions here never call a tool it would dominate the mean: it put session
    # mean TSS at 0.988 over 10 sessions while AC was defined for only 4, so H1 was comparing
    # two different populations.
    if sa.names or sb.names:
        out["tss"] = tss(sa.names, sb.names)
        out["js_kernel"] = js_kernel(sa.hist, sb.hist)
        out["gak"] = global_alignment_kernel(sa.names, sb.names)
        out["trajectory_exact_match"] = 1.0 if sa.names == sb.names else 0.0
    else:
        out["tss"] = out["js_kernel"] = out["gak"] = out["trajectory_exact_match"] = None
    out["ac"] = step_aligned_ac(sa.steps, sb.steps)
    out["n_steps"] = [len(sa.steps), len(sb.steps)]
    out["n_events"] = [sa.n_events, sb.n_events]
    return out


def _jackknife_ci(
    pair_vals: Dict[Tuple[str, str], Optional[float]], run_ids: Sequence[str], alpha: float
) -> Dict[str, Any]:
    """theta over a session's run pairs, with a delete-one-RUN jackknife CI.

    theta = C(n,2)^-1 * sum_{i<j} k(run_i, run_j) (Raj Eq. 1) — but the C(n,2) summands are
    DEPENDENT: each run appears in n-1 of them. A t-interval over pairs would treat 45 pairs
    as 45 independent observations and understate the variance badly. The jackknife resamples
    the actual independent units — the n runs:

        theta_(-r) = mean of the pairs not involving run r
        var_jack   = (n-1)/n * sum_r (theta_(-r) - mean_r theta_(-r))^2
        CI         = theta +/- t_{n-1,1-alpha/2} * sqrt(var_jack)

    Needs n >= 3: with 2 runs there is a single pair and deleting either leaves nothing.
    """
    defined = {p: v for p, v in pair_vals.items() if v is not None}
    n = len(run_ids)
    if not defined:
        return {"theta": None, "n_pairs": 0, "n_runs": n, "ci_low": None, "ci_high": None,
                "note": "no comparable run pairs"}
    theta = statistics.mean(defined.values())
    base = {"theta": theta, "n_pairs": len(defined), "n_runs": n}
    if n < 3:
        return {**base, "ci_low": None, "ci_high": None,
                "note": f"n_runs={n}; jackknife needs >=3"}

    loo: List[float] = []
    for r in run_ids:
        kept = [v for (a, b), v in defined.items() if r not in (a, b)]
        if kept:
            loo.append(statistics.mean(kept))
    if len(loo) < 2:
        return {**base, "ci_low": None, "ci_high": None,
                "note": "too few leave-one-out replicates"}
    mean_loo = statistics.mean(loo)
    var = (len(loo) - 1) / len(loo) * sum((x - mean_loo) ** 2 for x in loo)
    se = math.sqrt(var)
    tc = t_critical(len(loo) - 1, alpha)
    half = tc * se
    return {**base, "stdev_jack": se, "std_error": se, "alpha": alpha, "t_crit": tc,
            "ci_method": "jackknife-over-runs",
            "ci_low": max(0.0, theta - half), "ci_high": min(1.0, theta + half)}


def session_analysis(
    session_id: str, by_run: Dict[str, SessionRun], alpha: float
) -> Dict[str, Any]:
    """Everything reported for ONE session: theta + jackknife CI per metric, R x R matrices,
    the H1 check, and an event-count agreement check across the session's runs."""
    run_ids = sorted(by_run)
    n = len(run_ids)
    event_counts = {r: by_run[r].n_events for r in run_ids}
    step_counts = {r: len(by_run[r].steps) for r in run_ids}
    modal_events = Counter(event_counts.values()).most_common(1)[0] if event_counts else (0, 0)

    # The session's event axis is the UNION over its runs, so a run that stopped early is a
    # set of holes in the (event, run) table rather than a shorter axis. Counting the holes
    # says how ragged this session is directly; event_count_agreement below says the same
    # thing as a share of runs, and is kept because it is what the report prints.
    union = set().union(*(sr.features.keys() for sr in by_run.values())) if by_run else set()
    n_missing = sum(len(union) - by_run[r].n_events for r in run_ids)

    base: Dict[str, Any] = {
        "session_id": session_id,
        "n_runs": n,
        "run_ids": run_ids,
        "n_events": event_counts,
        "n_steps": step_counts,
        # Every event any run of this session reached, and how many (event, run) cells no
        # response was recorded for. 0 means every run replayed the whole session.
        "n_events_union": len(union),
        "n_missing_event_runs": n_missing,
        # Did every run of this session replay the same number of turns? A run that dropped
        # turns is not comparable on equal footing, and silently averaging it in would hide
        # exactly the failure the experiment is meant to surface.
        "event_count_agreement": (
            modal_events[1] / n if n else None
        ),
        "modal_event_count": modal_events[0],
    }
    if n < 2:
        return {**base, "note": "single run; nothing to compare", "metrics": {}, "matrices": {},
                "pairs": {}, "h1": None}

    pair_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ra, rb in combinations(run_ids, 2):
        pair_records[(ra, rb)] = session_pair(by_run[ra], by_run[rb])

    idx = {r: i for i, r in enumerate(run_ids)}
    matrices: Dict[str, List[List[Optional[float]]]] = {
        m: [[None] * n for _ in range(n)] for m in SESSION_METRICS
    }
    metrics: Dict[str, Dict[str, Any]] = {}
    for m in SESSION_METRICS:
        vals = {p: rec[m] for p, rec in pair_records.items()}
        metrics[m] = _jackknife_ci(vals, run_ids, alpha)
        for (ra, rb), v in vals.items():
            matrices[m][idx[ra]][idx[rb]] = v
            matrices[m][idx[rb]][idx[ra]] = v
    for m in SESSION_METRICS:  # a run vs itself is perfectly consistent by construction
        for i in range(n):
            matrices[m][i][i] = 1.0

    tss_theta = metrics["tss"]["theta"]
    ac_theta = metrics["ac"]["theta"]
    gap = (tss_theta - ac_theta) if (tss_theta is not None and ac_theta is not None) else None
    return {
        **base,
        "metrics": metrics,
        "matrices": matrices,
        "pairs": {f"{a}|{b}": v for (a, b), v in pair_records.items()},
        # supports_h1 is None, not False, when the gap is undefined (a session that called no
        # tools, or called them without arguments). "H1 untestable here" is not "H1 refuted".
        "h1": {"mean_tss": tss_theta, "mean_ac": ac_theta, "gap": gap,
               "supports_h1": None if gap is None else gap > 0},
    }


def per_session_analysis(cond: Condition, alpha: float) -> Dict[str, Any]:
    """Session-scope analysis for a whole condition: one entry per session, plus the
    equally-weighted session mean with a t-CI across sessions (Raj Eq. 3).

    Sessions ARE independent instances, so the cross-session interval keeps the paper's
    t-form with df = M-1. Only sessions with >=2 runs contribute.
    """
    per_session = {
        sid: session_analysis(sid, by_run, alpha)
        for sid, by_run in sorted(cond.sessions.items())
    }
    usable = [s for s in per_session.values() if s.get("metrics")]

    summary: Dict[str, Any] = {}
    for m in SESSION_METRICS:
        thetas = [s["metrics"][m]["theta"] for s in usable
                  if s["metrics"].get(m, {}).get("theta") is not None]
        summary[m] = {
            **_aggregate_u(thetas, alpha),
            "scope": "session_mean",
            "note_scope": "mean of per-session thetas, each session weighted equally",
        }

    # Only sessions where the gap is DEFINED are in the H1 denominator: a tool-free session
    # cannot support or refute "structure is more stable than arguments".
    h1s = [s["h1"] for s in usable if s.get("h1") and s["h1"]["gap"] is not None]
    supporting = [h for h in h1s if h["supports_h1"]]
    gaps = [h["gap"] for h in h1s]
    return {
        "n_sessions": len(per_session),
        "n_sessions_compared": len(usable),
        "sessions": per_session,
        "session_mean": summary,
        "h1_by_session": {
            "n_sessions": len(h1s),
            "n_testable": len(h1s),
            "n_untestable": len(usable) - len(h1s),
            "n_supporting": len(supporting),
            "mean_gap": statistics.mean(gaps) if gaps else None,
        },
        "event_count_agreement": {
            sid: s["event_count_agreement"] for sid, s in per_session.items()
        },
    }


def run_pair_matrix(per_session: Dict[str, Dict[str, Any]], run_ids: Sequence[str]) -> Dict[str, Any]:
    """R x R symmetric matrices per metric, each cell the SESSION MEAN for that run pair.

    cell[i][j] = mean over sessions of k(run_i, run_j | session), sessions weighted equally.
    The pre-fix version pooled every event of every session into one mean per run pair, which
    weighted a 15-event session ~7x a 2-event one and compared turns belonging to different
    tasks. Values are read back out of the already-computed per-session pair records rather
    than recomputed, so this view and the per-session table can never disagree.
    """
    run_ids = list(run_ids)
    idx = {r: i for i, r in enumerate(run_ids)}
    n = len(run_ids)
    # (run_a, run_b) -> metric -> per-session values
    collected: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_sessions_per_pair: Counter = Counter()

    for sess in per_session.values():
        for pair_key, rec in (sess.get("pairs") or {}).items():
            # run_ids are run_* directory names, so '|' can only be the separator.
            ra, rb = pair_key.split("|", 1)
            if ra not in idx or rb not in idx:
                continue
            n_sessions_per_pair[(ra, rb)] += 1
            for m in SESSION_METRICS:
                v = rec.get(m)
                if v is not None:
                    collected[(ra, rb)][m].append(v)

    matrices: Dict[str, List[List[Optional[float]]]] = {
        m: [[None] * n for _ in range(n)] for m in SESSION_METRICS
    }
    pair_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (ra, rb), by_metric in collected.items():
        rec: Dict[str, Any] = {"n_sessions": n_sessions_per_pair[(ra, rb)]}
        for m in SESSION_METRICS:
            vals = by_metric.get(m) or []
            v = statistics.mean(vals) if vals else None
            rec[m] = v
            rec[f"n_sessions_{m}"] = len(vals)
            matrices[m][idx[ra]][idx[rb]] = v
            matrices[m][idx[rb]][idx[ra]] = v
        pair_records[(ra, rb)] = rec
    for m in SESSION_METRICS:  # a run vs itself is perfectly consistent for every metric
        for i in range(n):
            matrices[m][i][i] = 1.0

    # Per-run "distance to pack": 1 - mean similarity to the other runs (higher = more of
    # an outlier). Uses output_levenshtein as the summary similarity.
    outlier: Dict[str, float] = {}
    for i, ri in enumerate(run_ids):
        sims = [matrices["output_levenshtein"][i][j] for j in range(n) if j != i]
        sims = [s for s in sims if s is not None]
        outlier[ri] = (1.0 - statistics.mean(sims)) if sims else 0.0

    return {
        "scope": "session_mean_per_run_pair",
        "run_ids": run_ids,
        "metrics": SESSION_METRICS,
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
    per_key: Dict[str, List["Feature"]],
    kind: str,
    alpha: float,
    judge_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """EVENT-SCOPE consistency U-statistic theta with a t-based CI (Raj et al.).

    Each INSTANCE is one event key — one identical input — and `per_key` maps it to that
    input's usable outputs across runs, as built by :func:`pooled_event_groups`. For one input
    we form U_n = C(n,2)^-1 * sum_{i<j} k(y_i, y_j) — the mean pairwise kernel. Then across the
    M instances we report the mean Ubar, sample stdev, and the t-CI
    Ubar +/- t_{M-1,1-alpha/2} * sigma_hat / sqrt(M).

    Taking the pool as an ARGUMENT rather than building it from a condition is the point: the
    caller has to name the pooling, so this function cannot quietly become the headline. It is
    a legitimate quantity — "given the same prompt, how repeatable is one turn?" — but it is
    NOT the session-level consistency the papers' hypotheses are about, and its M instances are
    not independent (the events of one session share a task and a prefix).
    `session_u_statistic` is the headline one. This is also where the LLM-judge clustering
    lives, since the judge clusters the repeated outputs of a single identical input.
    """
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

    return {
        "kernel": kind,
        "scope": "event",
        "instance": "event_id (one identical input)",
        **_aggregate_u(instance_u, alpha),
        "judge": judge_summary,
    }


def session_output_kernel(sa: SessionRun, sb: SessionRun, kind: str) -> Optional[float]:
    """Output kernel for a run PAIR of one session: mean output_kernel over its events.

    The session is the instance and the run is the sample unit, so the kernel has to reduce
    a whole session to one number in [0,1]. It is the mean of the per-event kernel over the
    events the two runs share — inapplicable events (errored, or no prose for the content
    kernel) drop out rather than padding it, exactly as in compare_runs. None when the pair
    shares no applicable event.
    """
    vals: List[float] = []
    for key in sorted(set(sa.features) & set(sb.features)):
        v = output_kernel(sa.features[key], sb.features[key], kind)
        if v is not None:
            vals.append(v)
    return statistics.mean(vals) if vals else None


def session_u_statistic(
    cond: Condition, kind: str, alpha: float
) -> Dict[str, Any]:
    """SESSION-SCOPE consistency U-statistic — the headline theta (Raj et al., Eq. 1/3).

    INSTANCE = session, SAMPLE UNIT = run:
        theta_m = C(n,2)^-1 * sum_{i<j} k(run_i, run_j | session m)      (Eq. 1)
        theta   = mean_m theta_m,  CI = theta +/- t_{M-1,1-a/2} * s/sqrt(M)   (Eq. 3)
    with M = number of sessions. Per-session intervals use the delete-one-run jackknife (the
    C(n,2) pairs share runs and are not independent); the cross-session interval keeps the
    paper's t-form because sessions ARE independent instances.

    The judge kernel is not available here — it clusters the repeated outputs of ONE identical
    input, which is an event-scope question. kind='judge' therefore scores with 'exact' and
    the cluster counts are reported by u_statistic (event scope).
    """
    effective_kind = "exact" if kind == "judge" else kind
    per_session: Dict[str, Dict[str, Any]] = {}
    thetas: List[float] = []
    for sid, by_run in sorted(cond.sessions.items()):
        run_ids = sorted(by_run)
        if len(run_ids) < 2:
            continue
        vals = {
            (ra, rb): session_output_kernel(by_run[ra], by_run[rb], effective_kind)
            for ra, rb in combinations(run_ids, 2)
        }
        est = _jackknife_ci(vals, run_ids, alpha)
        per_session[sid] = est
        if est["theta"] is not None:
            thetas.append(est["theta"])

    out: Dict[str, Any] = {
        "kernel": kind,
        "effective_kernel": effective_kind,
        "scope": "session",
        "instance": "session (sample unit = run)",
        **_aggregate_u(thetas, alpha),
        "per_session": per_session,
    }
    if kind == "judge":
        out["judge_note"] = "judge clusters are event-scope; see event_u_statistic.judge"
    return out


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


def trajectory_u_statistic(
    per_session: Dict[str, Dict[str, Any]], alpha: float
) -> Dict[str, Any]:
    """Trajectory-level consistency U-statistic theta with a t-CI (Raj et al., Eq. 1/3).

    Same instance/sample structure as `session_u_statistic` — instance = session, sample
    unit = run — but the kernel acts on the session's tool-call TRAJECTORY rather than its
    text output. Under Assumption 1 (x_mi = x_m0) the n repetitions of one session are a
    valid sample, so no base-vs-perturbed split is needed:

        theta_m = C(n,2)^-1 * sum_{i<j} k(tau_i, tau_j)   per session (Eq. 1)
        theta   = mean_m theta_m                          across M sessions (Eq. 3)

    Two kernels, both normalized to [0,1] with k(tau,tau)=1:
      - js:  Jensen-Shannon kernel over action histograms  -> action *composition*
      - gak: Global Alignment Kernel over name sequences    -> action *ordering*

    tau is the SESSION trajectory (SessionRun.names). Applying these kernels to one turn's
    tool list — the pre-fix behavior — left them nothing to measure: a turn emits 0 or 1
    call, so "composition" was a single bin and "ordering" a single element, and both
    collapsed to "same tool name or not". Values are read from the per-session table
    (js_kernel / gak) instead of recomputed, so the two views cannot disagree.
    """
    out: Dict[str, Any] = {}
    for kind, metric, layer in (("js", "js_kernel", "composition"), ("gak", "gak", "ordering")):
        est = {sid: s["metrics"][metric] for sid, s in per_session.items() if s.get("metrics")}
        thetas = [e["theta"] for e in est.values() if e.get("theta") is not None]
        out[kind] = {
            "kernel": kind,
            "layer": layer,
            "scope": "session",
            "instance": "session (sample unit = run)",
            **_aggregate_u(thetas, alpha),
            "per_session": est,
        }
    return out


# --------------------------------------------------------------- cross-condition MMD


def condition_trajectories(cond: Condition, session_id: str) -> List[Tuple[str, ...]]:
    """Every run's SESSION trajectory for one session in a condition.

    One trajectory per run — the session's whole action sequence, not one turn's tool list.
    That is the object the MMD kernels are defined over: the two-sample question is "do the
    two conditions produce the same DISTRIBUTION of trajectories for this task?".
    """
    return [sr.names for sr in cond.sessions.get(session_id, {}).values()]


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
    conditions: Dict[str, Condition],
    perm: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """For each condition pair, aggregate per-SESSION unbiased MMD^2 over the sessions both
    conditions replayed, under both the JS (composition) and GAK (ordering) kernels, with an
    optional permutation p-value.

    The instance is a session: within it, condition A contributes its runs' trajectories and
    condition B contributes its own, and MMD^2 asks whether those two samples of trajectories
    came from the same distribution. Averaging over sessions weights each task equally.
    """
    names = sorted(conditions.keys())
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    for na, nb in combinations(names, 2):
        ca, cb = conditions[na], conditions[nb]
        # Instances = sessions present (with >=2 usable trajectories) in BOTH conditions.
        shared_sessions = sorted(set(ca.sessions) & set(cb.sessions))

        for kname, kfn in (("js", _kernel_js), ("gak", global_alignment_kernel)):
            per_instance: List[Tuple[List, List, float]] = []
            for sid in shared_sessions:
                xs = condition_trajectories(ca, sid)
                ys = condition_trajectories(cb, sid)
                v = _mmd2_unbiased(xs, ys, kfn)
                if v is not None:
                    per_instance.append((xs, ys, v))
            if not per_instance:
                results.append({
                    "condition_a": na, "condition_b": nb, "kernel": kname,
                    "mmd2": None, "note": "no shared sessions with >=2 trajectories each",
                })
                continue

            mmd2 = statistics.mean(v for _, _, v in per_instance)
            # Sessions where NEITHER condition called a tool contribute an exact 0 (the two
            # samples are genuinely identical), which is correct but dilutes the mean toward
            # "indistinguishable". Report how many instances were trivial in that way so the
            # number is read against the right denominator.
            trivial = sum(
                1 for xs, ys, _v in per_instance if not any(xs) and not any(ys)
            )
            entry = {
                "condition_a": na, "condition_b": nb, "kernel": kname,
                "instance": "session", "n_instances": len(per_instance),
                "n_trivial": trivial, "mmd2": mmd2,
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


def h1_check(session_result: Dict[str, Any]) -> Dict[str, Any]:
    """Yagubyan Hypothesis 1: E[TSS] >> E[AC], at session scope.

    Both expectations are means of per-SESSION thetas (each session weighted equally), so the
    comparison is between two quantities computed over the same trajectories. The pre-fix
    version averaged per-TURN values pooled across sessions, where TSS reduced to "same tool
    name?" and the gap mostly reflected that degeneracy rather than the hypothesis.

    `n_supporting` reports how many individual sessions show the gap, which is the more
    informative claim: H1 supported on 9 of 10 sessions says more than one pooled mean.
    """
    mean_tss = session_result["session_mean"]["tss"].get("theta")
    mean_ac = session_result["session_mean"]["ac"].get("theta")
    gap = (mean_tss - mean_ac) if (mean_tss is not None and mean_ac is not None) else None
    by_session = session_result.get("h1_by_session") or {}
    return {
        "scope": "session_mean",
        "mean_tss": mean_tss,
        "mean_ac": mean_ac,
        "gap": gap,
        "supports_h1": (gap is not None and gap > 0),
        "n_sessions": by_session.get("n_sessions"),
        "n_supporting": by_session.get("n_supporting"),
        "mean_session_gap": by_session.get("mean_gap"),
        "per_session": {
            sid: s["h1"] for sid, s in (session_result.get("sessions") or {}).items()
            if s.get("h1")
        },
    }


# ----------------------------------------------------------------------- report


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _fmt_theta(est: Dict[str, Any]) -> str:
    """theta with its interval, or the reason there isn't one."""
    if est.get("theta") is None:
        return f"n/a — {est.get('note', 'no data')}"
    if est.get("ci_low") is None:
        return f"{est['theta']:.3f}  ({est.get('note', 'CI undefined')})"
    return f"{est['theta']:.3f}  [{est['ci_low']:.3f}, {est['ci_high']:.3f}]"


# Columns of the per-session table: the paper metrics first, then output agreement.
_TABLE_METRICS = [("tss", "TSS"), ("ac", "AC"), ("js_kernel", "JS"), ("gak", "GAK"),
                  ("output_exact_match", "exact"), ("output_levenshtein", "lev")]


def print_report(analysis: Dict[str, Any]) -> None:
    print("\n" + "=" * 74)
    print("PAPER-GROUNDED AGENT CONSISTENCY  (session-scope)")
    print("=" * 74)

    for name, cond in analysis["conditions"].items():
        mat = cond["run_pair_matrix"]
        rids = mat["run_ids"]
        sess = cond["per_session"]
        print(
            f"\nCondition: {name}   ({len(rids)} runs x {sess['n_sessions']} sessions; "
            f"{sess['n_sessions_compared']} comparable)"
        )
        print("-" * 74)

        # Headline: the session-scope U-statistic. Instance = session, sample unit = run.
        u = cond["u_statistic"]
        alpha_pct = int((1 - (u.get("alpha") or 0.05)) * 100)
        print(f"  Session U-statistic theta ({u['kernel']}): {_fmt_theta(u)}"
              f"   (M={u.get('n_instances')} sessions, {alpha_pct}% CI)")

        traj = cond.get("trajectory_u_statistic") or {}
        for kind, label in (("js", "composition"), ("gak", "ordering")):
            t = traj.get(kind)
            if t:
                print(f"  Trajectory theta [{kind}/{label}]: {_fmt_theta(t)}"
                      f"   (M={t.get('n_instances')} sessions)")

        h = cond["h1"]
        arrow = ">>" if h["supports_h1"] else "<= "
        verdict = "supports H1" if h["supports_h1"] else "does NOT support H1"
        support = ""
        if h.get("n_sessions"):
            support = f"; {h['n_supporting']}/{h['n_sessions']} sessions individually"
        print(
            f"  Hypothesis 1 (E[TSS] {arrow} E[AC]):  "
            f"TSS={_fmt(h['mean_tss'])}  AC={_fmt(h['mean_ac'])}  "
            f"gap={_fmt(h['gap'])}  -> {verdict}{support}"
        )

        # Per-session table — the primary output. theta per metric over that session's run
        # pairs; a session whose runs disagreed on event COUNT is flagged, because averaging
        # it in on equal footing would hide a dropped-turn failure.
        rows = [(sid, s) for sid, s in sess["sessions"].items() if s.get("metrics")]
        if rows:
            header = f"  {'session':<30}{'runs':>5}{'evts':>10}" + "".join(
                f"{lbl:>8}" for _, lbl in _TABLE_METRICS
            )
            print("\n" + header)
            print("  " + "-" * (len(header) - 2))
            for sid, s in rows:
                counts = list(s["n_events"].values())
                # When the runs agree, one number says it all. When they don't, the RANGE is
                # the finding — a session whose runs replayed 1 to 30 events did not diverge
                # slightly, it fell apart, and the modal count alone would hide that.
                if (s["event_count_agreement"] or 0) >= 1.0:
                    evts = str(s["modal_event_count"])
                else:
                    evts = f"{min(counts)}-{max(counts)}*"
                cells = "".join(
                    f"{_fmt(s['metrics'][m].get('theta')):>8}" for m, _ in _TABLE_METRICS
                )
                label = sid if len(sid) <= 29 else sid[:28] + "~"
                print(f"  {label:<30}{s['n_runs']:>5}{evts:>10}{cells}")
            if any((s["event_count_agreement"] or 0) < 1.0 for _, s in rows):
                print("  * runs of this session replayed different numbers of events "
                      "(range shown); metrics below are over the events they share")
            print("  (theta per session over its C(n,2) run pairs; "
                  "CIs are in the JSON, jackknifed over runs)")

        # Session mean per metric — the one cross-session number, with a t-CI (df = M-1).
        print("\n  Session mean (equal weight per session, t-CI over sessions):")
        for m, est in sess["session_mean"].items():
            if est.get("theta") is not None:
                print(f"    {m:24}{_fmt_theta(est)}   (M={est['n_instances']})")

        # Event-scope U — kept, but explicitly a different question.
        ev = cond.get("event_u_statistic")
        if ev:
            print(f"\n  Event-scope output U ({ev['kernel']}, one identical input = one "
                  f"instance): {_fmt_theta(ev)}   (M={ev.get('n_instances')} events)")
            if ev.get("judge"):
                j = ev["judge"]
                print(f"    judge: {j['n_judged']} groups judged, "
                      f"mean {j['mean_clusters']:.2f} clusters (max {j['max_clusters']})")

        # Outliers
        outl = sorted(mat["outlier_score"].items(), key=lambda kv: -kv[1])
        worst = outl[0] if outl else None
        if worst and worst[1] > 0:
            print(f"\n  Most outlying run: {worst[0]} (distance-to-pack {worst[1]:.3f})")

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
            line = f"{base} MMD^2={e['mmd2']:+.4f}  (n={e['n_instances']} sessions"
            if e.get("n_trivial"):
                line += f", {e['n_trivial']} tool-free"
            line += ")"
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

    conditions: Dict[str, Condition] = {}
    for name, d in conds.items():
        try:
            cond = load_condition(d, name)
        except (FileNotFoundError, ValueError) as e:
            print(f"warning: condition {name!r} ({d}) not analyzable — skipping: {e}",
                  file=sys.stderr)
            continue
        if not cond.sessions:
            print(f"warning: condition {name!r} ({d}) has no usable sessions — skipping",
                  file=sys.stderr)
            continue
        conditions[name] = cond
        print(f"Loaded condition {name!r}: {len(cond.sessions)} sessions x "
              f"{len(cond.run_ids)} runs from {d}")

    if not conditions:
        print("error: no conditions had usable data.", file=sys.stderr)
        return 1

    analysis: Dict[str, Any] = {"conditions": {}}
    for name, cond in conditions.items():
        # Session scope first: every other view is derived from these pair records, so the
        # per-session table, the run x run matrix and the trajectory thetas cannot disagree.
        sess = per_session_analysis(cond, args.alpha)
        analysis["conditions"][name] = {
            "per_session": sess,
            "run_pair_matrix": run_pair_matrix(sess["sessions"], cond.run_ids),
            "u_statistic": session_u_statistic(cond, kernel, args.alpha),
            "trajectory_u_statistic": trajectory_u_statistic(sess["sessions"], args.alpha),
            "h1": h1_check(sess),
            # A different, explicitly event-scoped question — and where the judge lives. The
            # pooling across sessions is named at the call site on purpose.
            "event_u_statistic": u_statistic(
                pooled_event_groups(cond), kernel, args.alpha, judge_cfg
            ),
        }

    if len(conditions) >= 2:
        analysis["mmd"] = cross_condition_mmd(conditions, args.perm, args.seed)

    print_report(analysis)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nFull analysis written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
