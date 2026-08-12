"""Traces Similarity: build a trace profile, and compare two of them.

Each profile represents one execution of a complete trace (one run of one model).
`extract_profile` turns a run's records into one, and the comparison functions take two
profiles and return similarity scores. All metrics are normalized to [0, 1].

`extract_profile` is the only place in this library that walks raw records, and it does so
only by delegating to replay_parsing.parse_response — it reads no files and decodes no JSON
itself. It lives here rather than in replay_parsing.py because its field set is chosen
entirely by compare_profiles: `tool_sequence` exists for the LCS metric,
`tool_call_sequence` for the arg-aware ordered-dedup metric, `unique_tools` for the Jaccard.
It is the profile *metric's* input format, not a general view of a run.
"""

from typing import Dict, List, Tuple

from replay_parsing import extract_tool_names, parse_response

from .signatures import response_signature, tool_args_signature
from .response_similarity import (
    collapse_adjacent,
    compare_responses,
    compare_tool_set_overlap,
    tool_sequence_lcs,
)


# ------------------------------------------------------- records -> trace profile

def extract_profile(records: List[dict]) -> dict:
    """Extract a trace profile from a list of request-response records.

    Args:
        records: List of dicts from per_request_lifecycle_metrics.json

    Returns:
        dict with keys:
          - tool_sequence: Tuple[str, ...] — ordered tool NAMES across all requests
          - tool_call_sequence: Tuple[Tuple[str, str], ...] — ordered (name, canonical-args)
            tokens across all requests; the arg-aware counterpart of tool_sequence, used by
            the ordered-dedup trace metric (two calls match only if name AND args agree)
          - unique_tools: set — set of tool names used
          - num_requests: int — how many request-response pairs
          - num_tool_calls: int — total tool invocations
          - num_errors: int — how many requests errored
          - responses: List[dict] — parsed responses (ok, content, tool_calls, etc.)
          - records: List[dict] — original records

    This is the parsing step for a whole run: the resulting profile is what
    compare_profiles consumes, and its `responses` are already parsed, so the comparison
    functions below never re-read a raw record.
    """
    all_tool_names: List[str] = []
    all_tool_calls: List[Tuple[str, str]] = []
    responses = []
    num_errors = 0

    for rec in records:
        parsed = parse_response(rec)
        responses.append(parsed)

        if not parsed["ok"]:
            num_errors += 1
            continue

        # Extract tool names in order.
        all_tool_names.extend(extract_tool_names(parsed["tool_calls"]))
        # Arg-aware tokens in order: (name, canonical-args) per call, spanning the whole trace.
        all_tool_calls.extend(tool_args_signature(parsed["tool_calls"]))

    tool_sequence = tuple(all_tool_names)
    unique_tools = set(all_tool_names)

    return {
        "tool_sequence": tool_sequence,
        "tool_call_sequence": tuple(all_tool_calls),
        "unique_tools": unique_tools,
        "num_requests": len(records),
        "num_tool_calls": len(all_tool_names),
        "num_errors": num_errors,
        "responses": responses,
        "records": records,
    }


# ------------------------------------------------------------ profile comparison

def compare_session_depths(depth_a: int, depth_b: int) -> float:
    """Compare session depths (# of requests).

    Returns 1.0 if equal, else (1 - |diff| / max).
    """
    if depth_a == depth_b:
        return 1.0
    if depth_a == 0 or depth_b == 0:
        return 0.0
    diff = abs(depth_a - depth_b)
    max_depth = max(depth_a, depth_b)
    return 1.0 - (diff / max_depth)


def compare_response_lists(
    responses_a: List[dict], responses_b: List[dict], align_by_position: bool = True
) -> float:
    """Compare two lists of PARSED responses (profile["responses"]).

    If align_by_position=True (default), aligns by index and compares only min(len).
    If False, returns 0 if lists have different lengths.

    Returns the mean per-response **content Levenshtein** across aligned pairs — a single
    grounded metric, not the mean of the (now-removed) composite score. Errored pairs
    contribute 0 (compare_responses returns all-zero metrics when either side errored).
    """
    if not responses_a and not responses_b:
        return 1.0
    if not responses_a or not responses_b:
        return 0.0

    if not align_by_position and len(responses_a) != len(responses_b):
        return 0.0

    similarities = []
    for i in range(min(len(responses_a), len(responses_b))):
        comparison = compare_responses(responses_a[i], responses_b[i])
        similarities.append(comparison.get("content_levenshtein", 0.0))

    return sum(similarities) / len(similarities) if similarities else 0.0


def compare_profiles(profile_a: dict, profile_b: dict) -> Dict[str, float]:
    """Compare two trace profiles, returning all metrics.

    Input: two dicts from extract_profile() above, with keys:
           tool_sequence, tool_call_sequence, unique_tools, num_requests, num_errors,
           responses, records

    Output: dict mapping metric name → similarity in [0, 1].
    """
    result = {}

    # Tool sequence comparison
    seq_a = profile_a.get("tool_sequence", ())
    seq_b = profile_b.get("tool_sequence", ())
    result["tool_sequence_similarity"] = tool_sequence_lcs(seq_a, seq_b)

    # Order-preserving, arg-aware trace comparison that forgives only back-to-back repeats.
    # Uses the (name, canonical-args) token stream across the whole trace, collapses adjacent
    # duplicates (a tool re-emitted on consecutive turns is a stutter, not a new step), then
    # LCS-ratios the two deduped sequences. Complements tool_sequence_similarity, which is
    # name-only and keeps every repeat.
    call_seq_a = profile_a.get("tool_call_sequence", ())
    call_seq_b = profile_b.get("tool_call_sequence", ())
    result["tool_calls_ordered_dedup"] = tool_sequence_lcs(
        collapse_adjacent(call_seq_a), collapse_adjacent(call_seq_b)
    )

    # Tool set overlap (Jaccard)
    tools_a = profile_a.get("unique_tools", set())
    tools_b = profile_b.get("unique_tools", set())
    result["tool_set_overlap"] = compare_tool_set_overlap(tools_a, tools_b)

    # Session depth comparison
    depth_a = profile_a.get("num_requests", 0)
    depth_b = profile_b.get("num_requests", 0)
    result["session_depth_agreement"] = compare_session_depths(depth_a, depth_b)

    # Error rate comparison
    errors_a = profile_a.get("num_errors", 0)
    errors_b = profile_b.get("num_errors", 0)
    error_rate_a = errors_a / depth_a if depth_a > 0 else 0.0
    error_rate_b = errors_b / depth_b if depth_b > 0 else 0.0
    if error_rate_a == error_rate_b:
        result["error_rate_agreement"] = 1.0
    else:
        result["error_rate_agreement"] = 1.0 - abs(error_rate_a - error_rate_b)

    # Response-by-response comparison (align by position)
    responses_a = profile_a.get("responses", [])
    responses_b = profile_b.get("responses", [])
    result["response_similarity"] = compare_response_lists(
        responses_a, responses_b, align_by_position=True
    )

    # Exact match: identical tool sequences and every aligned response byte-identical
    # (same content + canonical tool args). Uses response_signature rather than a
    # similarity threshold so "identical" means truly identical, not "scored 1.0".
    seq_match = seq_a == seq_b
    if seq_match and len(responses_a) == len(responses_b):
        all_ok = all(
            response_signature(responses_a[i]) is not None
            and response_signature(responses_a[i]) == response_signature(responses_b[i])
            for i in range(len(responses_a))
        )
        result["exact_match"] = 1.0 if all_ok else 0.0
    else:
        result["exact_match"] = 0.0

    # No composite roll-up (see compare_responses): report the individual metrics and, for
    # a single grounded consistency figure, use the U-statistic theta in
    # consistency_statistics.py.
    return result
