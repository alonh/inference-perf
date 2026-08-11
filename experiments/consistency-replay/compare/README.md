# Compare: Consistency Analysis Library

A lightweight, general-purpose comparison library for analyzing consistency between LLM
responses and traces.

## Design Philosophy

**All comparison functions follow a simple contract:**
- Input: two comparable objects (**already-parsed** responses, or profiles)
- Output: `Dict[str, float]` with metrics normalized to [0, 1]
- No aggregation, no statistics, no side effects, **no parsing** — just comparison

This makes the library composable: any caller can use it for any comparison task. It is the
**single source of truth** for per-pair metrics; the analyzers (`analyze_consistency.py`,
`consistency_statistics.py`) and the viewer/witness exporters all import these primitives
so every caller shares one metric definition and owns only the aggregation layer.

### Parsing is one module, and callers own it

`parsing.py` is the single place that touches a file, a raw record, or a JSON string.
Nothing in it computes a metric; nothing outside it parses.

```
parsing.py            files + raw records → parsed responses / profiles   (no metrics)
  ↑
response_similarity.py Level 1 comparators + the string/tool metrics       (no parsing)
  ↑
traces_similarity.py   Level 2 profile comparison                          (no parsing)
kernels.py             positive-definite trajectory kernels (JS, GAK)
compare_runs.py        the CLI — the only place in the library that parses
```

The comparators therefore **refuse** raw records rather than quietly parsing them:
`compare_responses(a, b)` and `response_signature(r)` raise `ValueError` when handed a dict
with no `ok` key. Parse once, up front, and pass the parsed data down — that way a record is
never re-parsed per pairwise comparison, and a raw record can't be silently compared as
though it were parsed.

```python
from compare import load_records, parse_records, compare_responses

responses = parse_records(load_records(path))       # parse once, here
compare_responses(responses[0], responses[1])       # comparators only compare
```

## Two-Level Architecture

### Level 2: Profile Comparison

Compare two complete trace profiles (a full execution of a task/conversation — i.e. one run).

```python
from compare import extract_profile, compare_profiles

# Load and extract profiles from two runs
profile_1 = extract_profile(load_run("run_1"))
profile_2 = extract_profile(load_run("run_2"))

similarity = compare_profiles(profile_1, profile_2)
# Returns:
# {
#     "tool_sequence_similarity": 0.95,   # LCS ratio of ordered tool names
#     "tool_calls_ordered_dedup": 0.90,   # order-preserving, arg-aware, adjacent repeats collapsed
#     "tool_set_overlap": 0.80,           # Jaccard of tool sets
#     "session_depth_agreement": 1.0,     # same # of requests?
#     "error_rate_agreement": 1.0,        # same error rates?
#     "response_similarity": 0.87,        # mean per-response content Levenshtein
#     "exact_match": 0.0,                 # tool seq identical AND all responses byte-identical?
# }
```

**Use cases:**
- Compare two runs of the same model on the same trace (reproducibility)
- Compare traces from different models on the same task (cross-model analysis)
- Compare the same trace across different conditions/hyperparameters

### Level 1: Response Comparison

Compare two individual response records (one turn of a conversation).

```python
from compare import compare_responses, parse_response

response_a = parse_response(record_a)  # required — compare_responses will not parse for you
response_b = parse_response(record_b)

similarity = compare_responses(response_a, response_b)
# Returns:
# {
#     "content_levenshtein": 0.92,        # exact text edit distance (whitespace-collapsed)
#     "content_jaccard": 0.88,            # word-set overlap
#     "tool_calls_exact": 1.0,            # identical tools & args? (1/0)
#     "tool_calls_ordered_dedup": 1.0,    # order-preserving, arg-aware, adjacent repeats collapsed
#     "tool_sequence_similarity": 1.0,    # LCS ratio of tool names
#     "tool_set_overlap": 1.0,            # Jaccard of tool sets
#     "tool_args_consistency": 0.85,      # Jaccard of {(tool.key, value)} pairs (1.0 if no args)
#     "finish_reason_agreement": 1.0,     # same ending? (1/0)
# }
```

**Use cases:**
- Fine-grained debugging of where two traces diverge
- Analyzing argument consistency for multi-tool turns
- Building blocks for profile comparisons

## Core Functions

### Parsing / IO (`parsing.py`)

Everything that reads a file or a raw record. Import these from `compare`; call them from your
run script, not from inside a comparison loop.

| Function | Input | Output |
|----------|-------|--------|
| `find_run_dirs(base)` | a reports base dir | sorted list of `run_*` paths |
| `load_records(path)` | one metrics JSON file | List[dict]; raises `ValueError` on an unrecognized shape |
| `load_run(run_dir)` | a `run_*` dir | List[dict] — tolerant wrapper (`[]` if absent/unreadable) |
| `parse_response(record)` | one raw record | Dict[str, Any] (ok, has_output, content, tool_calls, timing, tokens, …) |
| `parse_records(records)` | List[dict] raw | List[dict] parsed |
| `extract_profile(records)` | List[dict] raw, one run | dict of profile features (incl. parsed `responses`) |
| `require_parsed(r, what)` | a dict | the dict, or `ValueError` if it isn't parsed |
| `request_key` / `session_key` / `event_key` | one raw record | str — grouping identity |

Both tool-call wire shapes are accepted: `{"function": {"name", "arguments"}}` and the flat
`{"name", "arguments"}` used by the viewer's flattened summaries.

### Profile Functions (`traces_similarity.py`)

| Function | Input | Output |
|----------|-------|--------|
| `compare_profiles(p1, p2)` | two profile dicts from `extract_profile` | Dict[str, float] |
| `compare_response_lists(a, b)` | two lists of **parsed** responses | Dict[str, float] |

### Response Functions (`response_similarity.py`)

| Function | Input | Output |
|----------|-------|--------|
| `compare_responses(r1, r2)` | two **parsed** response dicts | Dict[str, float] |

### Helper Functions

Content similarity (all in [0, 1]):
- `normalized_levenshtein(a, b)` → float — exact edit distance; the headline figure
- `fast_content_ratio(a, b)` → float — approximate (difflib), for large run × run matrices
- `jaccard(a, b)` → float — word-set overlap

Tool-call names / signatures (in `parsing.py` — these read raw tool-call dicts):
- `extract_tool_names(tool_calls)` → Tuple[str, ...]
- `tool_signature(tool_calls)` → Tuple of (name, sorted arg **keys**)
- `tool_args_signature(tool_calls)` → Tuple of (name, canonical arg **values**)
- `response_signature(response)` → Optional[str] — exact-match unit (content + tool args)
- `tool_kv_set(tool_calls)` → frozenset of `(tool.key, canonical-value)`

Tool-call comparators (all in [0, 1] unless noted):
- `compare_tool_calls(a, b)` → float (1/0: exact names + args)
- `compare_tool_calls_ordered_dedup(a, b)` → float (order + args, adjacent repeats collapsed)
- `tool_sequence_lcs(a, b)` → float (LCS ratio over names)
- `tss(a, b)` → float (edit-distance TSS over names; paper Def. 3)
- `compare_tool_set_overlap(a, b)` → float (Jaccard of tool sets)
- `compare_tool_arguments(a, b)` → Optional[float] (Jaccard of {(k,v)}; None if no args)
- `argument_consistency(kv_a, kv_b)` → Optional[float] (paper Def. 4)

Trajectory kernels (positive-definite, normalized to (0, 1] with k(x, x) = 1):
- `js_kernel(hist_a, hist_b)` → float — composition similarity (Jensen-Shannon)
- `global_alignment_kernel(seq_a, seq_b)` → float — ordering similarity (GAK)
- plus `action_histogram(name_seq)` and `js_divergence(p, q)` building blocks

## Example Usage

### Simple: compare two runs

```python
from compare import load_records, extract_profile, compare_profiles

records_1 = load_records("run_1/per_request_lifecycle_metrics.json")
records_2 = load_records("run_2/per_request_lifecycle_metrics.json")

result = compare_profiles(extract_profile(records_1), extract_profile(records_2))
print(f"Tool-sequence similarity: {result['tool_sequence_similarity']:.1%}")
print(f"Response similarity:      {result['response_similarity']:.1%}")
print(f"Exact match:              {bool(result['exact_match'])}")
```

Or from the command line:

```bash
python compare/compare_runs.py run_1/per_request_lifecycle_metrics.json \
                               run_2/per_request_lifecycle_metrics.json --verbose
```

### Advanced: pairwise run comparison with statistics

```python
from compare import extract_profile, compare_profiles
import statistics

profiles = [extract_profile(load_run(f"run_{i}")) for i in range(5)]
sims = [
    compare_profiles(profiles[i], profiles[j])["tool_sequence_similarity"]
    for i in range(len(profiles))
    for j in range(i + 1, len(profiles))
]
print(f"Mean tool-sequence similarity: {statistics.mean(sims):.1%}")
print(f"Std dev:                       {statistics.stdev(sims):.1%}")
# For a single grounded figure with a confidence interval, use consistency_statistics.py.
```

### Fine-grained: find divergence point

```python
from compare import compare_responses, parse_records

responses_a = parse_records(records_a)   # parse once; the comparator won't do it for you
responses_b = parse_records(records_b)

for i, (resp_a, resp_b) in enumerate(zip(responses_a, responses_b)):
    sim = compare_responses(resp_a, resp_b)
    if sim["content_levenshtein"] < 0.9:
        print(f"Divergence at turn {i}: content Levenshtein {sim['content_levenshtein']:.3f}")
        break
```

## Integration with Existing Code

The library is standalone and is imported wherever parsing or per-pair metrics are needed. All
four sibling scripts get their record reading, grouping keys, tool-call extraction and
exact-match signature from `compare.parsing`, so none of them re-implements the derive step;
each owns only its own aggregation:

- `analyze_consistency.py` — imports the primitives for group-level modal/distinct/CV metrics.
- `consistency_statistics.py` — uses the primitives and kernels for the U-statistic θ,
  confidence intervals, and the paper-grounded hypothesis checks.
- `find_metric_witnesses.py` / `build_metric_viewer.py` — call the comparators directly to
  find and render example pairs for each metric.
- `export_viewer_data.py` — reuses the same primitives so the viewer's numbers match the
  analyzer's exactly.

## Metric Details

### Profile Metrics

- **tool_sequence_similarity**: Longest Common Subsequence (LCS) ratio between ordered tool names.
- **tool_calls_ordered_dedup**: LCS ratio over the `(name, canonical-args)` token stream after
  collapsing *adjacent* duplicates — forgives a tool re-emitted back-to-back, but not a genuine
  revisit; arg-aware, so two calls match only if name AND args agree.
- **tool_set_overlap**: Jaccard similarity of tool sets.
- **session_depth_agreement**: how similar are the # of requests? (`1 − |Δ| / max`)
- **error_rate_agreement**: how similar are error rates? (`1 − |Δrate|`)
- **response_similarity**: mean per-response **content Levenshtein** across aligned pairs.
- **exact_match**: 1 iff tool sequences are identical AND every aligned response is
  byte-identical (compared via `response_signature`, not a similarity threshold).

### Response Metrics

- **content_levenshtein**: exact normalized edit distance (whitespace-collapsed).
- **content_jaccard**: word-set overlap.
- **tool_calls_exact**: 1 iff tools + arguments are identical, else 0.
- **tool_calls_ordered_dedup**: order-preserving, arg-aware; collapses only adjacent repeats.
- **tool_sequence_similarity**: LCS ratio of tool names.
- **tool_set_overlap**: Jaccard similarity of tools.
- **tool_args_consistency**: Jaccard of `{(tool.key, value)}` pairs; **1.0** in the returned
  dict when neither call has args — the underlying `compare_tool_arguments` returns `None`
  (undefined, not a divergence), and `compare_responses` folds that to 1.0 to match its
  sibling structural metrics in the fixed-width vector. Aggregators that average this metric
  should instead *exclude* the None pairs rather than count them; see `argument_consistency`'s
  docstring for the full fold policy.
- **finish_reason_agreement**: 1 iff same finish reason (stop / tool_calls / …), else 0.

## Design Notes

### Whitespace Handling

- **Content**: collapsed to single spaces (`collapse_ws`) — normalizes indentation/wrapping.
- **Arguments**: valid JSON is re-serialized canonically (key order + spacing don't matter);
  the unparseable fallback is whitespace-stripped (`strip_ws`).

### Error / empty Handling

- If either response errored (`ok=False`), `compare_responses` returns every metric as 0.0.
- `compare_tool_arguments` / `argument_consistency` return `None` when neither call has
  arguments (undefined, not a divergence — the caller excludes it rather than scoring 0.0).
- **Empty-vs-empty is a library caveat, not something the primitives guard against.** The
  content primitives score two empty strings as *identical* — `normalized_levenshtein("", "")`
  and `fast_content_ratio("", "")` both return 1.0 (the `a == b` fast path). So a raw
  `compare_responses` on two empty completions reports `content_levenshtein = 1.0`, which is
  vacuous agreement, not real consistency.
- To handle that, `parse_response` exposes **`has_output`** — True only when an ok response
  actually carries content or tool calls (an ok turn that spent its whole budget on reasoning
  and emitted nothing is `ok=True, has_output=False`). The library itself does **not** apply
  this flag; the aggregation layer in `consistency_statistics.py` does, with two guards:
  - **mixed output** (one side produced output, the other was empty) → every similarity
    metric is forced to 0.0 — the sides genuinely diverged;
  - **both sides empty on a channel** → that pair is *excluded* from the channel's mean (and
    skipped in the U-statistic), rather than being counted as a 1.0 match.
  Channel-aware aggregation means each metric is only averaged over pairs that actually
  exercise its channel (content metrics over pairs with prose, tool metrics over pairs with a
  tool call), so a metric's headline is never padded by both-sides-absent pairs.

## Testing

```bash
python -m pytest compare/test_*.py -v  # unit tests
```
