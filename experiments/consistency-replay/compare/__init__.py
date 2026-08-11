"""Consistency comparison library — the single source of truth for per-pair metrics.

Three modules, one direction of dependency:

  parsing.py              files / raw records / JSON  -> comparison-ready inputs
  response_similarity.py  Level 1: compare two parsed responses (one turn)
  traces_similarity.py    Level 2: compare two trace profiles (one full run)

Parsing happens ONCE, up front, in the caller — compare_runs.py is the library's own worked
example. The two similarity modules never read a file and never parse a record; hand them a
raw record and they raise. Both return Dict[str, float] with metrics in [0, 1]. There is
deliberately NO composite "overall similarity" — for one grounded consistency figure use the
U-statistic theta in consistency_statistics.py, which carries a confidence interval.

The analyzers (analyze_consistency.py, consistency_statistics.py) and the viewer / witness
exporters import the primitives below so every caller shares one metric definition AND one
parse definition; they own the aggregation layer (modal fractions, U-statistic theta/CI,
run x run matrices, MMD, the LLM judge).
"""

from .parsing import (
    # File IO.
    find_run_dirs,
    load_records,
    load_run,
    # Record -> parsed response.
    parse_response,
    parse_records,
    require_parsed,
    # Record identity / grouping.
    request_key,
    session_key,
    event_key,
    # Records -> trace profile.
    extract_profile,
    # Whitespace normalization.
    collapse_ws,
    strip_ws,
    # Tool-call signatures / names.
    extract_tool_names,
    tool_signature,
    tool_args_signature,
    response_signature,
    tool_kv_set,
)
from .traces_similarity import (
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

__all__ = [
    # Parsing: file IO
    "find_run_dirs",
    "load_records",
    "load_run",
    # Parsing: record -> parsed response
    "parse_response",
    "parse_records",
    "require_parsed",
    # Parsing: record identity / grouping
    "request_key",
    "session_key",
    "event_key",
    # Parsing: records -> profile
    "extract_profile",
    # Parsing: whitespace
    "collapse_ws",
    "strip_ws",
    # Parsing: signatures / names
    "extract_tool_names",
    "tool_signature",
    "tool_args_signature",
    "response_signature",
    "tool_kv_set",
    # Level 2: Profile comparison
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
]
