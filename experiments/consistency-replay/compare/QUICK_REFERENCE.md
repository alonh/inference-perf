# Compare Library: Quick Reference

## Import

```python
from compare import (
    # Level 2: Full traces
    extract_profile,
    compare_profiles,

    # Level 1: Individual responses
    compare_responses,
    parse_response,

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

> **No composite score.** Neither `compare_responses` nor `compare_profiles` returns an
> `overall_similarity`. Each metric measures a different thing in a different unit, so a single
> weighted average would need arbitrary weights and mean nothing precise. For one grounded
> "how consistent" number (a U-statistic θ **with a confidence interval**), run
> `consistency_statistics.py` over the run set.

## Two Entry Points

### Level 2: Compare Profiles
```python
# Extract a profile from one run's records
profile = extract_profile(records: List[dict]) → dict

# Compare two profiles
result = compare_profiles(profile_a, profile_b) → Dict[str, float]
```

**Metrics**: tool_sequence_similarity, tool_calls_ordered_dedup, tool_set_overlap,
session_depth_agreement, error_rate_agreement, response_similarity, exact_match

### Level 1: Compare Responses
```python
# Parse a record (optional — compare_responses parses raw dicts itself)
parsed = parse_response(record: dict) → dict

# Compare two responses
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
responses_a = [parse_response(r) for r in records_a]
responses_b = [parse_response(r) for r in records_b]
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
- `tool_args_consistency` — Jaccard over {(tool.key, value)} pairs (0.0 if no args)
- `finish_reason_agreement` — same ending? (1/0)

## Notes

- All metrics in [0, 1], where 1 = identical, 0 = completely different
- All functions are pure (no side effects)
- No external dependencies (standard library only)
- If either response errored (`ok=False`), every response metric is 0.0
- See `README.md` for full documentation

## Files

| File | Purpose |
|------|---------|
| response_similarity.py | Level 1: response comparison + all primitives |
| traces_similarity.py | Level 2: profile comparison |
| kernels.py | Paper-grounded trajectory kernels (JS, GAK) |
| compare_runs.py | CLI: compare two runs' metrics files |
| __init__.py | Module exports |
| test_compare.py | Unit tests |
| README.md | Full documentation |
