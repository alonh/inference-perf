"""Consistency comparison library — the single source of truth for per-pair metrics.

Two entry points:
  Level 1: compare_responses() — compare two individual response records
  Level 2: compare_profiles()  — compare two full trace profiles

Both return Dict[str, float] with metrics in [0, 1]. There is deliberately NO composite
"overall similarity" — for one grounded consistency figure use the U-statistic theta in
consistency_statistics.py, which carries a confidence interval.

The analyzers (analyze_consistency.py, consistency_statistics.py) import the primitives
below so every caller shares one metric definition; they own the aggregation layer
(modal fractions, U-statistic theta/CI, run x run matrices, MMD, the LLM judge).
"""

from .traces_similarity import compare_profiles, extract_profile
from .response_similarity import (
    compare_responses,
    parse_response,
    # Content similarity — exact vs fast (see each docstring).
    normalized_levenshtein,
    fast_content_ratio,
    jaccard,
    # Whitespace normalization.
    collapse_ws,
    strip_ws,
    # Tool-call signatures / names.
    extract_tool_names,
    tool_signature,
    tool_args_signature,
    response_signature,
    tool_kv_set,
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

__all__ = [
    # Level 2: Profile comparison
    "compare_profiles",
    "extract_profile",
    # Level 1: Response comparison
    "compare_responses",
    "parse_response",
    # Content similarity
    "normalized_levenshtein",
    "fast_content_ratio",
    "jaccard",
    # Whitespace
    "collapse_ws",
    "strip_ws",
    # Signatures / names
    "extract_tool_names",
    "tool_signature",
    "tool_args_signature",
    "response_signature",
    "tool_kv_set",
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
]
