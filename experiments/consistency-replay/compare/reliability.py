"""Reliability :: Consistency (C) metrics — session-level replay port of hal-harness.

Canonical, importable definitions of the five Consistency sub-metrics from
steverab/hal-harness (``reliability_eval/metrics/consistency.py``, the implementation
accompanying arXiv:2602.16666), adapted to this output-consistency **replay** setting. Kept in
the library (not a script) so every caller — notebook, CLI, viewer — shares ONE definition,
matching the rest of ``compare``.

**This module is ONE metric, not the shared machinery.** Everything here is specific to that
paper's C-metrics: what counts as success, which six resource channels enter C_res, the 1/3
weights. The general layer every analysis shares — the layout read, the (event, run) table,
and the two-stage ``session_conclusion`` / ``combine`` driver — is :mod:`replay_parsing`, which
lives outside ``compare`` precisely so that this module and its siblings can import from one
place. So: this file imports the contract, it does not define it, and nothing else should
import the contract *through* it.

This module is also the library's **documented exception** to three of ``compare``'s own
rules. The rest of ``compare`` is pure, stdlib-only, and per-pair: it takes two already-parsed
objects and returns a dict of floats, leaving aggregation to the caller. hal-harness's C
metrics are not per-pair — each one is defined as a variance or a mean over all K runs of a
session — so the aggregation *is* the definition, and splitting it out would leave nothing to
share. Hence, only here: statistics over a whole run set, a ``numpy`` dependency, and file
reading (:func:`compute_all` and :func:`load_session_runs` drive it, delegating every actual
read to ``replay_parsing``).

Unit of analysis
----------------
hal-harness runs an agent **task** K times and compares the K whole-task runs. Here the
analog of a "task" is a **session** (one recorded trace); its K "runs" are the K replays
(``run_1`` .. ``run_K``). That maps onto the shared contract directly:
:func:`session_reliability` is stage 1 (combine one session's runs) and :func:`aggregate` is
stage 2 (mean and standard error across sessions).

Metric mapping (hal-harness formula -> replay mapping)
------------------------------------------------------
- ``C_out``   : ``1 - var/(p(1-p)+eps)`` clipped to [0,1] over a binary per-run success
  flag. Replay has no task reward, so success is a PROXY (see :func:`session_success`):
  every event ``ok`` (no transport/parse error) AND no ``length``-truncation.
- ``C_traj_d``: ``1 - mean pairwise JSD`` of per-run action histograms (successful runs).
  Trajectory = the session's tool-name sequence (concat of each event's tool calls, in
  event order).
- ``C_traj_s``: ``mean pairwise normalized-Levenshtein similarity`` (``1 - dist/max_len``)
  of the same trajectories (successful runs), ordering-sensitive.
- ``C_conf``  : ``exp(-CV_conf)`` over self-reported confidence — N/A here (no confidence
  signal in the replay data). Always NaN.
- ``C_res``   : ``exp(-mean CV)`` over cost / time / api_calls / num_actions / num_errors /
  latency across all runs.
- ``R_con``   : equal-weight (1/3, 1/3, 1/3) aggregate of C_out, mean(C_traj_d, C_traj_s),
  C_res. C_conf is excluded (hal-harness reports confidence separately).

CAVEAT — replay is not an agent rollout. Every event is fed the ORIGINAL recorded input,
so a session "trajectory" is a bundle of independent per-event outputs, not a causally
chained agent path: C_traj measures "do identical fixed inputs yield the same tool-call
structure", not "does the agent follow the same route".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from replay_parsing import (
    Session,
    extract_tool_names,
    iter_sessions,
    parse_response,
)

from .kernels import action_histogram, js_divergence

# --------------------------------------------------------------------------- constants
# hal-harness reliability_eval constants (reliability_eval/constants.py).
EPSILON = 1e-8
W_OUTCOME = 1.0 / 3.0
W_TRAJECTORY = 1.0 / 3.0
W_RESOURCE = 1.0 / 3.0

# Token pricing hal-harness uses to estimate cost from usage.
INPUT_COST_PER_TOKEN = 5.0 / 1_000_000  # $5 / 1M input tokens
OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 / 1M output tokens


# ----------------------------------------------------------------- per-(session, run) data
@dataclass
class SessionRun:
    """One session's execution within one replay run (analog of a single task run)."""

    success: bool
    trajectory: Tuple[str, ...]  # tool names, in event order, across the whole session
    cost: float
    time: float
    api_calls: int  # number of model requests (events) in the session
    num_actions: int  # total tool calls across the session
    num_errors: int  # number of errored events
    latency_mean: float  # mean per-event wall-clock latency


def session_success(events: Sequence[dict]) -> bool:
    """Binary outcome PROXY for a replayed session (see module docstring).

    True iff every event parsed cleanly (``ok``) and none was truncated by the token
    budget (``finish_reason == 'length'``). Stands in for hal-harness's task reward, which
    replay lacks. Swap this one function to change the outcome definition. NOTE: on
    reasoning-heavy traces ``length`` truncation is common, so with this proxy C_out
    largely reflects *truncation* consistency — revisit if that is not the intent.
    """
    return all(ev["ok"] for ev in events) and all(
        ev["finish_reason"] != "length" for ev in events
    )


def _event_latency(ev: dict) -> Optional[float]:
    st, et = ev.get("start_time"), ev.get("end_time")
    if st is None or et is None:
        return None
    return max(0.0, et - st)


def summarize_session_run(events: Sequence[dict]) -> SessionRun:
    """Collapse one session's parsed events (from a single run) into a SessionRun.

    ``events`` are :func:`replay_parsing.parse_response` dicts, in event order.
    """
    trajectory: List[str] = []
    cost = 0.0
    total_time = 0.0
    num_actions = 0
    num_errors = 0
    latencies: List[float] = []

    for ev in events:
        trajectory.extend(extract_tool_names(ev["tool_calls"]))
        num_actions += len(ev["tool_calls"])
        if not ev["ok"]:
            num_errors += 1
        pt = ev["prompt_tokens"] or 0
        ct = ev["completion_tokens"] or 0
        cost += pt * INPUT_COST_PER_TOKEN + ct * OUTPUT_COST_PER_TOKEN
        lat = _event_latency(ev)
        if lat is not None:
            latencies.append(lat)
            total_time += lat

    return SessionRun(
        success=session_success(events),
        trajectory=tuple(trajectory),
        cost=cost,
        time=total_time,
        api_calls=len(events),
        num_actions=num_actions,
        num_errors=num_errors,
        latency_mean=float(np.mean(latencies)) if latencies else 0.0,
    )


# --------------------------------------------------------------------------- metric maths
def cv(values: Sequence[float], *, errors_like: bool = False) -> Optional[float]:
    """Coefficient of variation (std ddof=1 / mean). None when undefined.

    ``errors_like`` mirrors hal-harness's error-count channel: zeros are meaningful (not
    dropped), and a zero mean with nonzero spread yields +inf.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if mean == 0.0:
        if errors_like:
            return math.inf if std > 0 else 0.0
        return None  # a non-error channel that is identically zero carries no CV signal
    return std / mean


def compute_outcome_consistency(successes: Sequence[bool]) -> float:
    """C_out = 1 - var / (p(1-p) + eps), clipped to [0, 1]. NaN if < 2 runs."""
    if len(successes) < 2:
        return math.nan
    arr = np.asarray(successes, dtype=float)
    p = float(arr.mean())
    var = float(arr.var(ddof=1))
    c = 1.0 - var / (p * (1.0 - p) + EPSILON)
    return float(np.clip(c, 0.0, 1.0))


def seq_levenshtein_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """1 - (edit distance / max length), element-wise over two tool-name sequences.

    Two empty sequences are identical (1.0); one empty and one non-empty is 0.0.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost_ij = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost_ij)
        prev = cur
    return 1.0 - prev[m] / max(n, m)


def _mean_pairwise(items: Sequence, fn) -> Optional[float]:
    if len(items) < 2:
        return None
    vals = [fn(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
    return float(np.mean(vals)) if vals else None


def compute_trajectory_consistency(
    trajectories: Sequence[Tuple[str, ...]],
) -> Tuple[float, float]:
    """Return (C_traj_d, C_traj_s) over the given (successful-run) trajectories.

    C_traj_d = 1 - mean pairwise Jensen-Shannon divergence of action histograms.
    C_traj_s = mean pairwise normalized-Levenshtein similarity of tool-name sequences.
    NaN for either when fewer than 2 trajectories are available.
    """
    if len(trajectories) < 2:
        return math.nan, math.nan
    hists = [action_histogram(t) for t in trajectories]
    mean_jsd = _mean_pairwise(hists, js_divergence)
    mean_sim = _mean_pairwise(trajectories, seq_levenshtein_similarity)
    c_traj_d = math.nan if mean_jsd is None else 1.0 - mean_jsd
    c_traj_s = math.nan if mean_sim is None else mean_sim
    return c_traj_d, c_traj_s


def compute_resource_consistency(runs: Sequence[SessionRun]) -> float:
    """C_res = exp(-mean of available CVs) over cost/time/api_calls/actions/errors/latency."""
    if len(runs) < 2:
        return math.nan
    channels = {
        "cost": ([r.cost for r in runs], False),
        "time": ([r.time for r in runs], False),
        "api_calls": ([r.api_calls for r in runs], False),
        "num_actions": ([r.num_actions for r in runs], False),
        "num_errors": ([r.num_errors for r in runs], True),
        "latency": ([r.latency_mean for r in runs], False),
    }
    cvs = []
    for values, errors_like in channels.values():
        c = cv(values, errors_like=errors_like)
        if c is not None and math.isfinite(c):
            cvs.append(c)
    if not cvs:
        return math.nan
    return math.exp(-float(np.mean(cvs)))


def weighted_r_con(c_out: float, c_traj_d: float, c_traj_s: float, c_res: float) -> float:
    """Equal-weight aggregate of outcome, mean(trajectory), resource; NaN-masked."""
    traj_pair = [v for v in (c_traj_d, c_traj_s) if not math.isnan(v)]
    c_traj = float(np.mean(traj_pair)) if traj_pair else math.nan
    parts = [(c_out, W_OUTCOME), (c_traj, W_TRAJECTORY), (c_res, W_RESOURCE)]
    avail = [(v, w) for v, w in parts if not math.isnan(v)]
    if not avail:
        return math.nan
    wsum = sum(w for _, w in avail)
    return sum(v * w for v, w in avail) / wsum


# ------------------------------------------------------------------- per-session + aggregate
@dataclass
class SessionMetrics:
    session_id: str
    n_runs: int
    n_success: int
    c_out: float
    c_traj_d: float
    c_traj_s: float
    c_conf: float
    c_res: float
    r_con: float


def compute_session_metrics(session_id: str, runs: Sequence[SessionRun]) -> SessionMetrics:
    """Compute the five C-metrics + R_con for one session across its replay runs."""
    successes = [r.success for r in runs]
    success_trajs = [r.trajectory for r in runs if r.success]

    c_out = compute_outcome_consistency(successes)
    c_traj_d, c_traj_s = compute_trajectory_consistency(success_trajs)
    c_conf = math.nan  # no self-reported confidence in replay data
    c_res = compute_resource_consistency(runs)
    r_con = weighted_r_con(c_out, c_traj_d, c_traj_s, c_res)

    return SessionMetrics(
        session_id=session_id,
        n_runs=len(runs),
        n_success=sum(successes),
        c_out=c_out,
        c_traj_d=c_traj_d,
        c_traj_s=c_traj_s,
        c_conf=c_conf,
        c_res=c_res,
        r_con=r_con,
    )


def mean_se(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Mean and standard error over the non-NaN values."""
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], float)
    if arr.size == 0:
        return {"mean": None, "se": None, "n": 0}
    se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "se": se, "n": int(arr.size)}


def aggregate(session_metrics: Sequence[SessionMetrics]) -> Dict[str, Dict]:
    """Aggregate per-session metrics into {metric: {mean, se, n}} across sessions."""
    keys = ["c_out", "c_traj_d", "c_traj_s", "c_conf", "c_res", "r_con"]
    return {k: mean_se([getattr(m, k) for m in session_metrics]) for k in keys}


# -------------------------------------------------------------------- the two-stage contract
def session_runs(session: Session) -> List[SessionRun]:
    """One SessionRun per run of `session`, in run order.

    Events are taken in the session's canonical ``events`` order, not file order: a run's
    ``per_request_lifecycle_metrics.json`` is written in completion order, which async
    completion under randomized schedule delays can permute, and a run-varying order would
    spuriously deflate the sequence metric C_traj_s. The axis is ``event_key`` order, whose
    embedded zero-padded call index makes it the same call order for every run.

    A run that never reached some events contributes the ones it has, so a truncated run
    yields a shorter trajectory rather than gaps — the same semantics as before, now visible
    as ``session.missing()`` instead of being invisible.
    """
    by_run = session.by_run()
    return [
        summarize_session_run([by_run[r][e] for e in session.events if e in by_run[r]])
        for r in session.runs
    ]


def session_reliability(session: Session) -> SessionMetrics:
    """This module's stage 1: combine the RUNS of one session into its C-metrics.

    A valid ``session_conclusion`` for :func:`replay_parsing.analyze` — it takes a Session and
    nothing else, so the result cannot depend on which other sessions were on disk. The name
    is the metric's, not the contract's: these are the paper's C-metrics specifically, and
    another analysis plugs its own stage 1 into the same driver.
    """
    return compute_session_metrics(session.session_id, session_runs(session))


# --------------------------------------------------------------------------------- loading
def load_session_runs(base: str) -> Dict[str, List[SessionRun]]:
    """Return {session_id: [SessionRun per replay run]} for one experiment directory.

    Kept for callers that want the raw per-run summaries rather than the metrics. The layout
    read and the (event, run) table are :mod:`replay_parsing`'s; this is only the projection
    into ``SessionRun``.
    """
    return {s.session_id: session_runs(s) for s in iter_sessions(base, parse_response)}


def compute_all(base: str, min_runs: int = 2) -> Tuple[List[SessionMetrics], Dict[str, Dict], int]:
    """End-to-end: load a base dir and return (per-session metrics, aggregate, n_skipped).

    Stage 1 is :func:`session_metrics` per session, stage 2 is :func:`aggregate` across them —
    the mean is over SESSIONS, never over a pool of runs, because the session is the unit of
    independence. Sessions with fewer than ``min_runs`` runs cannot support a consistency
    number at all and are counted as skipped rather than folded in.
    """
    metrics: List[SessionMetrics] = []
    skipped = 0
    for session in iter_sessions(base, parse_response):
        if len(session.runs) < min_runs:
            skipped += 1
            continue
        metrics.append(session_reliability(session))
    metrics.sort(key=lambda m: m.session_id)
    return metrics, aggregate(metrics), skipped
