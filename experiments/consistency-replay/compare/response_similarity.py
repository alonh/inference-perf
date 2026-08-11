"""Response Similarity: Compare two individual response records.

Each response is one turn's output (content + tool calls).
Functions in this module take two responses and return similarity scores.
All metrics are normalized to [0, 1]:
  - 0 = completely different
  - 1 = identical
  - 0.5 = moderate similarity

This module only COMPARES. It never reads a file and never parses a raw record: its inputs
are already-parsed responses (parsing.parse_response) and the extracted tool-call structures
that parsing.py produces. Hand it a raw record and it raises rather than quietly parsing —
see parsing.require_parsed for why.
"""

import difflib
from itertools import groupby
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

from .parsing import (
    collapse_ws,
    extract_tool_names,
    require_parsed,
    tool_args_signature,
    tool_kv_set,
)


def normalized_levenshtein(a: str, b: str) -> float:
    """EXACT content similarity in [0,1] = 1 - edit_distance / max_len.

    True character-level Levenshtein via full DP. This is the canonical, exact metric —
    the headline Levenshtein figure in FINDINGS comes from here. Cost is O(len_a*len_b)
    per pair in pure Python; when you need to compare many pairs (a run x run matrix),
    prefer `fast_content_ratio`, which trades exactness for speed.
    """
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


def fast_content_ratio(a: str, b: str) -> float:
    """FAST (approximate) content similarity in [0,1], 1.0 = identical.

    difflib's SequenceMatcher.ratio() (2*M/T over matching blocks). NOT true edit distance:
    it finds matching blocks greedily rather than the minimum edit path, so mid-range values
    can differ slightly from `normalized_levenshtein` (they agree at the endpoints: 1.0 for
    identical, ~0 for disjoint). Use it where a pair is compared many times — e.g. the
    run x run matrix in consistency_statistics.py, where full DP would cost minutes — and
    keep `normalized_levenshtein` for the exact, headline figure.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity of word sets."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compare_tool_calls(calls_a: List[dict], calls_b: List[dict]) -> float:
    """Compare tool calls: exact match of complete signatures (names + args).

    Returns 1.0 if identical, 0.0 otherwise.
    """
    sig_a = tool_args_signature(calls_a)
    sig_b = tool_args_signature(calls_b)
    return 1.0 if sig_a == sig_b else 0.0


def tool_sequence_lcs(names_a: Tuple[str, ...], names_b: Tuple[str, ...]) -> float:
    """Compare tool sequences by longest common subsequence (LCS) ratio.

    Returns ratio of LCS length to max length, in [0, 1]. This is a *subsequence* measure
    (order-preserving, gaps allowed) — distinct from the paper's edit-distance TSS below.
    For `(a,b,c)` vs `(a,c,b)`: LCS -> 0.667, TSS -> 0.333.
    """
    if names_a == names_b:
        return 1.0

    max_len = max(len(names_a), len(names_b))
    if max_len == 0:
        return 1.0  # both empty

    # Compute LCS length
    m, n = len(names_a), len(names_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if names_a[i - 1] == names_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    return lcs_len / max_len


def tss(names_a: Sequence[str], names_b: Sequence[str]) -> float:
    """Tool Sequence Similarity (Yagubyan Def. 3): token-level normalized-Levenshtein
    similarity over tool-NAME sequences (edit distance over NAME tokens, not characters).

    This is the paper's TSS — the pairwise summand whose mean over N runs is E[TSS] in the
    Hypothesis-1 check. Unlike `tool_sequence_lcs` it counts substitutions, so reordering a
    tool is penalized more heavily. Prefer this for paper-grounded reporting.
    """
    la, lb = len(names_a), len(names_b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = names_a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == names_b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def collapse_adjacent(seq: Sequence) -> Tuple:
    """Run-length dedup: collapse only *consecutive* equal elements.

    (A, A, B) -> (A, B) — a stutter is treated as one call. But (A, B, A) -> (A, B, A):
    the two A's are not adjacent, so a genuine revisit is preserved. This is the middle
    ground between a set (drops all order and multiplicity) and the raw sequence (keeps
    every repeat): it forgives a tool being emitted twice back-to-back but not the model
    genuinely returning to a tool after doing something else.
    """
    return tuple(k for k, _ in groupby(seq))


def compare_tool_calls_ordered_dedup(calls_a: List[dict], calls_b: List[dict]) -> float:
    """Order-preserving, arg-aware tool-call similarity that ignores *consecutive* repeats.

    Each call is tokenized as (name, canonical-args) via tool_args_signature — so two calls
    count as "the same" only when both the tool name AND its argument values match (order /
    spacing of arg keys is canonicalized, so formatting can't fork them). Adjacent identical
    calls are then collapsed with collapse_adjacent before an LCS-ratio comparison.

    Result in [0, 1]: 1.0 when the two call sequences are equal modulo back-to-back repeats;
    lower as their (deduped) order diverges. Contrast:
      - tool_calls_exact       : 1/0, punishes a single duplicated call
      - tool_sequence_lcs      : order over NAMES only, keeps every repeat, ignores args
      - this metric            : order over (name, args), collapsing only adjacent repeats
    """
    seq_a = collapse_adjacent(tool_args_signature(calls_a))
    seq_b = collapse_adjacent(tool_args_signature(calls_b))
    return tool_sequence_lcs(seq_a, seq_b)


def compare_tool_set_overlap(tools_a: set, tools_b: set) -> float:
    """Jaccard similarity of tool sets."""
    if not tools_a and not tools_b:
        return 1.0
    if not tools_a or not tools_b:
        return 0.0
    intersection = len(tools_a & tools_b)
    union = len(tools_a | tools_b)
    return intersection / union if union > 0 else 0.0


def compare_tool_arguments(calls_a: List[dict], calls_b: List[dict]) -> Optional[float]:
    """Compare tool argument consistency using Jaccard over {(tool.key, value)} pairs.

    This aligns arguments by tool position, so only valid when tool sequences match.
    Returns None if no arguments present in either call.
    """
    return argument_consistency(tool_kv_set(calls_a), tool_kv_set(calls_b))


def argument_consistency(kv_a: AbstractSet, kv_b: AbstractSet) -> Optional[float]:
    """Argument Consistency (Yagubyan Def. 4): Jaccard over two flattened {(k,v)} sets.

    Returns None when NEITHER turn has arguments (AC undefined — nothing to compare) and
    0.0 when one side is empty or the tools differ (disjoint sets).

    Why None and not a number: the empty-vs-empty case is a genuine 0/0 — there were no
    arguments on either side, so "how consistent were the arguments?" has no answer. Every
    caller therefore has to CHOOSE how to fold that None, and the choice depends on what the
    caller is building. To keep the three folds from silently drifting apart, they are fixed
    here as the single policy of record:

      * AGGREGATION (consistency_statistics.compare_runs): EXCLUDE the pair from the mean.
        A step with no arguments should be in neither the numerator nor the denominator of a
        mean "argument consistency" — folding it to 1.0 would inflate the headline with steps
        that never exercised the metric, folding to 0.0 would deflate it. Report a coverage
        count (n_ac_steps) alongside.
      * FIXED-WIDTH VECTOR that cannot hold None (compare_responses, find_metric_witnesses):
        fold to 1.0, NOT 0.0. This matches the sibling structural metrics — compare_tool_calls
        ([],[]), tool_sequence_lcs((),()), compare_tool_set_overlap(set(),set()) all return 1.0
        for empty-vs-empty — so AC does not become a spurious outlier that alone claims two
        argument-free turns "disagreed". Where the surface can flag it (the metric viewer),
        mark such a pair "trivial (empty vs empty)" so the 1.0 is read as true-by-emptiness,
        not real agreement.
      * A SURFACE THAT CAN HOLD None (export_viewer_data): keep None and render it as blank.

    The one fold that is always wrong is 0.0 for empty-vs-empty: it asserts disagreement where
    nothing was compared.
    """
    if not kv_a and not kv_b:
        return None
    if not kv_a or not kv_b:
        return 0.0
    return len(kv_a & kv_b) / len(kv_a | kv_b)


def compare_responses(response_a: dict, response_b: dict) -> Dict[str, float]:
    """Compare two individual responses, returning all metrics.

    Input: two PARSED responses — dicts from parsing.parse_response, with keys ok, content,
           tool_calls, finish_reason, etc. A raw record raises ValueError; parse it once in
           the run script instead (a pairwise sweep would otherwise re-parse the same record
           on every comparison).

    Output: dict mapping metric name → similarity in [0, 1].
            All metrics always present (no None values).
    """
    require_parsed(response_a, "response_a")
    require_parsed(response_b, "response_b")

    result = {}

    # If either response had an error, similarity is 0
    if not response_a.get("ok") or not response_b.get("ok"):
        return {
            "content_levenshtein": 0.0,
            "content_jaccard": 0.0,
            "tool_calls_exact": 0.0,
            "tool_calls_ordered_dedup": 0.0,
            "tool_sequence_similarity": 0.0,
            "tool_set_overlap": 0.0,
            "tool_args_consistency": 0.0,
            "finish_reason_agreement": 0.0,
        }

    content_a = response_a.get("content", "")
    content_b = response_b.get("content", "")

    # Content similarity
    result["content_levenshtein"] = normalized_levenshtein(
        collapse_ws(content_a), collapse_ws(content_b)
    )
    result["content_jaccard"] = jaccard(content_a, content_b)

    # Tool calls
    calls_a = response_a.get("tool_calls") or []
    calls_b = response_b.get("tool_calls") or []

    result["tool_calls_exact"] = compare_tool_calls(calls_a, calls_b)

    # Order-preserving, arg-aware, collapsing only back-to-back repeats (see function).
    result["tool_calls_ordered_dedup"] = compare_tool_calls_ordered_dedup(calls_a, calls_b)

    # Tool sequence similarity
    names_a = extract_tool_names(calls_a)
    names_b = extract_tool_names(calls_b)
    result["tool_sequence_similarity"] = tool_sequence_lcs(names_a, names_b)

    # Tool set overlap
    tools_a = set(names_a)
    tools_b = set(names_b)
    result["tool_set_overlap"] = compare_tool_set_overlap(tools_a, tools_b)

    # Tool argument consistency. This dict is a fixed-width vector (no None allowed), so the
    # empty-vs-empty case (AC undefined) is folded to 1.0 — matching the sibling structural
    # metrics above (tool_calls_exact / tool_sequence_similarity / tool_set_overlap all score
    # 1.0 for empty-vs-empty), so AC is not a lone outlier claiming two argument-free turns
    # disagreed. See argument_consistency's docstring for the full fold policy.
    ac = compare_tool_arguments(calls_a, calls_b)
    result["tool_args_consistency"] = ac if ac is not None else 1.0

    # Finish reason agreement
    reason_a = response_a.get("finish_reason")
    reason_b = response_b.get("finish_reason")
    result["finish_reason_agreement"] = 1.0 if reason_a == reason_b else 0.0

    # No composite roll-up: each metric measures a different thing in a different unit, so
    # a single weighted average would need arbitrary weights and mean nothing precise. For
    # one grounded "how consistent" number, use the U-statistic theta in
    # consistency_statistics.py, which carries a confidence interval.
    return result
