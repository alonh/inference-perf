# Compare Library: Quick Reference

## Import

```python
from compare import (
    # Parsing / IO — the ONLY place records are read (parsing.py)
    find_run_dirs,
    load_records,             # one metrics file → List[dict]
    load_run,                 # one run_* dir → List[dict] (tolerant)
    parse_response,           # raw record → parsed response
    parse_records,            # List[dict] raw → List[dict] parsed
    extract_profile,          # one run's raw records → profile dict
    request_key, session_key, event_key,

    # Level 2: Full traces
    compare_profiles,

    # Level 1: Individual responses (PARSED input only)
    compare_responses,

    # Content similarity
    normalized_levenshtein,   # exact edit distance (headline figure)
    fast_content_ratio,       # approximate, for big run x run matrices
    jaccard,

    # Tool-call names / signatures
    extract_tool_names,
    tool_signature,           # (name, sorted arg KEYS)
    tool_args_signature,      # (name, canonical arg VALUES)
    response_signature,       # exact-match unit: content + tool args
    tool_kv_set,              # {(tool.key, value)} set

    # Tool-call comparators
    compare_tool_calls,               # exact names+args, 1/0
    compare_tool_calls_ordered_dedup, # order-preserving, arg-aware, ignores adjacent repeats
    tool_sequence_lcs,                # LCS ratio over tool names
    tss,                              # edit-distance TSS over tool names (paper Def. 3)
    compare_tool_set_overlap,         # Jaccard of tool sets
    compare_tool_arguments,           # Jaccard over {(k,v)} pairs
    argument_consistency,             # paper Def. 4 (Jaccard of kv-sets)

    # Trajectory kernels (paper-grounded, positive-definite)
    js_kernel,                # composition similarity (Jensen-Shannon)
    global_alignment_kernel,  # ordering similarity (GAK)
)
```

> **Parse once, then compare.** `parsing.py` owns every read of a file or a raw record; the
> comparators only compare. `compare_responses` and `response_signature` raise `ValueError`
> on a dict with no `ok` key rather than parsing it for you, so call `parse_response` /
> `parse_records` / `extract_profile` up front in your run script.

> **No composite score.** Neither `compare_responses` nor `compare_profiles` returns an
> `overall_similarity`. Each metric measures a different thing in a different unit, so a single
> weighted average would need arbitrary weights and mean nothing precise. For one grounded
> "how consistent" number (a U-statistic θ **with a confidence interval**), run
> `consistency_statistics.py` over the run set.

## Two Entry Points

Both take already-parsed input. Parsing happens once, in the caller.

### Level 2: Compare Profiles
```python
# Parse: one run's records → a profile (this is the parsing step)
profile = extract_profile(load_records(path)) → dict

# Compare two profiles
result = compare_profiles(profile_a, profile_b) → Dict[str, float]
```

**Metrics**: tool_sequence_similarity, tool_calls_ordered_dedup, tool_set_overlap,
session_depth_agreement, error_rate_agreement, response_similarity, exact_match

### Level 1: Compare Responses
```python
# Parse first — REQUIRED. A raw record raises ValueError in compare_responses.
parsed = parse_response(record: dict) → dict          # or parse_records(records)

# Compare two parsed responses
result = compare_responses(response_a, response_b) → Dict[str, float]
```

**Metrics**: content_levenshtein, content_jaccard, tool_calls_exact,
tool_calls_ordered_dedup, tool_sequence_similarity, tool_set_overlap,
tool_args_consistency, finish_reason_agreement

## Common Patterns

### Reproducibility (same model, different runs)
```python
import statistics
profiles = [extract_profile(load_run(f"run_{i}")) for i in range(5)]
seq_sims = [
    compare_profiles(profiles[i], profiles[j])["tool_sequence_similarity"]
    for i in range(len(profiles))
    for j in range(i + 1, len(profiles))
]
print(f"Mean tool-sequence similarity: {statistics.mean(seq_sims):.1%}")
# For a single grounded figure with a CI, use consistency_statistics.py instead.
```

### Cross-model (same trace, different models)
```python
profiles = {
    "claude": extract_profile(load_run("claude_trial")),
    "gpt4":   extract_profile(load_run("gpt4_trial")),
}
result = compare_profiles(profiles["claude"], profiles["gpt4"])
print(f"Tool overlap:        {result['tool_set_overlap']:.1%}")
print(f"Sequence similarity: {result['tool_sequence_similarity']:.1%}")
print(f"Response similarity: {result['response_similarity']:.1%}")
```

### Find divergence point
```python
responses_a = parse_records(records_a)
responses_b = parse_records(records_b)
for i, (ra, rb) in enumerate(zip(responses_a, responses_b)):
    sim = compare_responses(ra, rb)
    if sim["content_levenshtein"] < 0.9:
        print(f"Divergence at turn {i + 1}")
        print(f"  Response A: {ra['content'][:50]}...")
        print(f"  Response B: {rb['content'][:50]}...")
        print(f"  Content Levenshtein: {sim['content_levenshtein']:.3f}")
        break
```

## All Metrics at a Glance

### Profile Metrics (all in [0, 1])
- `tool_sequence_similarity` — LCS ratio over ordered tool names
- `tool_calls_ordered_dedup` — order-preserving, arg-aware LCS; collapses adjacent repeats
- `tool_set_overlap` — Jaccard of tool sets
- `session_depth_agreement` — same # of requests?
- `error_rate_agreement` — same error rates?
- `response_similarity` — mean per-response content Levenshtein (aligned by position)
- `exact_match` — identical tool sequence AND every aligned response byte-identical?

### Response Metrics (all in [0, 1])
- `content_levenshtein` — exact text edit distance (whitespace-collapsed)
- `content_jaccard` — word-set overlap
- `tool_calls_exact` — identical tools & args? (1/0)
- `tool_calls_ordered_dedup` — order-preserving, arg-aware, ignores adjacent repeats
- `tool_sequence_similarity` — LCS ratio over tool names
- `tool_set_overlap` — Jaccard of tools
- `tool_args_consistency` — Jaccard over {(tool.key, value)} pairs (**1.0** when neither side has
  args — undefined, folded to 1.0 in this fixed-width vector; see `argument_consistency`)
- `finish_reason_agreement` — same ending? (1/0)

## Notes

- All metrics in [0, 1], where 1 = identical, 0 = completely different
- Every comparison function is pure; only `parsing.py`'s `load_*` helpers touch the filesystem
- No external dependencies (standard library only)
- If either response errored (`ok=False`), every response metric is 0.0
- See `README.md` for full documentation

## Files

| File | Purpose |
|------|---------|
| parsing.py | **The only parsing layer**: file IO, raw record → parsed, profiles, grouping keys, tool-call extraction, signatures |
| response_similarity.py | Level 1: response comparison + the string / tool-call metrics |
| traces_similarity.py | Level 2: profile comparison |
| kernels.py | Paper-grounded trajectory kernels (JS, GAK) |
| compare_runs.py | CLI: compare two runs' metrics files — the only caller that parses |
| __init__.py | Module exports |
| test_compare.py | Unit tests |
| README.md | Full documentation |

Dependency direction is one-way: `parsing` ← `response_similarity` ← `traces_similarity`.
