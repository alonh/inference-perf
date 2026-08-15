"""Consistency comparison library — the single source of truth for per-pair metrics.

Everything here defines or computes a metric. Reading a file, parsing a raw record and
deriving a record's grouping identity are general operations that no longer live here: they
are in `replay_parsing`, a stdlib-only module one level up, which a notebook or a DataFrame
script can import without pulling in any of this. This library imports it, never the reverse.

  replay_parsing (external)  files / raw records / JSON  -> parsed responses, grouping keys
    ↑
  signatures.py           the canonical forms equality and a Jaccard are taken over
    ↑
  response_similarity.py  Level 1: compare two parsed responses (one turn)
    ↑
  traces_similarity.py    Level 2: build a trace profile, compare two of them (one full run)

Parsing happens ONCE, up front, in the caller (see README.md for the worked example). The
similarity modules never read a file and never parse a record; hand them a raw record and
they raise. Both return Dict[str, float] with metrics in [0, 1]. There is deliberately NO
composite "overall similarity" — for one grounded consistency figure use the U-statistic
theta in consistency_statistics.py, which carries a confidence interval.

The analyzers (analyze_consistency.py, consistency_statistics.py) and the viewer / witness
exporters import the primitives below so every caller shares one metric definition; they own
the aggregation layer (modal fractions, U-statistic theta/CI, run x run matrices, MMD, the
LLM judge).
"""

from .signatures import (
    # The parsed-input guard: this library's contract with replay_parsing.
    require_parsed,
    # Canonical tool-call / response units that equality and Jaccard are taken over.
    tool_signature,
    tool_args_signature,
    tool_kv_set,
    response_signature,
)
from .traces_similarity import (
    # Records -> trace profile (the input format compare_profiles consumes).
    extract_profile,
    compare_profiles,
    compare_response_lists,
    compare_session_depths,
)
from .response_similarity import (
    compare_responses,
    # Content similarity — exact vs fast (see each docstring).
    normalized_levenshtein,
    fast_content_ratio,
    jaccard,
    # Tool-call comparators.
    compare_tool_calls,
    compare_tool_calls_ordered_dedup,
    collapse_adjacent,
    tool_sequence_lcs,
    tss,
    compare_tool_set_overlap,
    compare_tool_arguments,
    argument_consistency,
)
from .kernels import (
    js_kernel,
    global_alignment_kernel,
    action_histogram,
    js_divergence,
)
from .reliability import (
    # hal-harness reliability_eval Consistency (C) metrics — session-level replay port.
    SessionRun,
    SessionMetrics,
    session_success,
    summarize_session_run,
    session_runs,
    session_reliability,
    load_session_runs,
    compute_outcome_consistency,
    compute_trajectory_consistency,
    compute_resource_consistency,
    seq_levenshtein_similarity,
    weighted_r_con,
    compute_session_metrics,
    aggregate as aggregate_consistency,
    compute_all as compute_consistency,
)

__all__ = [
    # Contract with replay_parsing
    "require_parsed",
    # Canonical units equality / Jaccard are taken over
    "tool_signature",
    "tool_args_signature",
    "tool_kv_set",
    "response_signature",
    # Level 2: Profile extraction and comparison
    "extract_profile",
    "compare_profiles",
    "compare_response_lists",
    "compare_session_depths",
    # Level 1: Response comparison
    "compare_responses",
    # Content similarity
    "normalized_levenshtein",
    "fast_content_ratio",
    "jaccard",
    # Tool-call comparators
    "compare_tool_calls",
    "compare_tool_calls_ordered_dedup",
    "collapse_adjacent",
    "tool_sequence_lcs",
    "tss",
    "compare_tool_set_overlap",
    "compare_tool_arguments",
    "argument_consistency",
    # Trajectory kernels
    "js_kernel",
    "global_alignment_kernel",
    "action_histogram",
    "js_divergence",
    # Reliability: Consistency (C) metrics (session-level replay port of hal-harness)
    "SessionRun",
    "SessionMetrics",
    "session_success",
    "summarize_session_run",
    "session_runs",
    "session_reliability",
    "load_session_runs",
    "compute_outcome_consistency",
    "compute_trajectory_consistency",
    "compute_resource_consistency",
    "seq_levenshtein_similarity",
    "weighted_r_con",
    "compute_session_metrics",
    "aggregate_consistency",
    "compute_consistency",
]
