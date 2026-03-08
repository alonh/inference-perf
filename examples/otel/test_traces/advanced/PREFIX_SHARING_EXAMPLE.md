# Prefix Sharing Example

This example demonstrates how prefix sharing detection works for KV cache optimization measurement.

## Overview

The `prefix_sharing_example.json` trace contains:
- **3 independent calls** (no shared context)
- **1 conversation chain with 4 turns** (progressive prefix sharing)

## Structure

### Independent Calls (No Prefix Sharing)

These are standalone requests with no shared context:

1. **Independent Call 1** (`span_independent_1`)
   - Question: "What is the capital of France?"
   - No dependencies, no prefix sharing

2. **Independent Call 2** (`span_independent_2`)
   - Question: "What is 2 + 2?"
   - No dependencies, no prefix sharing

3. **Independent Call 3** (`span_independent_3`)
   - Question: "What's the weather like?"
   - No dependencies, no prefix sharing

### Conversation Chain (With Prefix Sharing)

This is a multi-turn conversation about hash tables where each turn extends the previous context:

1. **Turn 1** (`span_conv_turn1`)
   ```
   System: "You are a helpful assistant..."
   User: "Can you explain what a hash table is?"
   ```
   - No dependencies (first turn)
   - Prompt tokens: 100

2. **Turn 2** (`span_conv_turn2`)
   ```
   System: "You are a helpful assistant..."
   User: "Can you explain what a hash table is?"
   Assistant: "A hash table is a data structure..."
   User: "What is the time complexity for lookups?"
   ```
   - **Prefix shares with Turn 1** (100% match on system + first user message)
   - Prompt tokens: 180 (80 new tokens)
   - **KV cache benefit**: Can reuse 100 tokens from Turn 1

3. **Turn 3** (`span_conv_turn3`)
   ```
   [All previous messages...]
   User: "How do you handle collisions?"
   ```
   - **Prefix shares with Turn 2** (100% match on all previous messages)
   - Prompt tokens: 270 (90 new tokens)
   - **KV cache benefit**: Can reuse 180 tokens from Turn 2

4. **Turn 4** (`span_conv_turn4`)
   ```
   [All previous messages...]
   User: "Which strategy is better?"
   ```
   - **Prefix shares with Turn 3** (100% match on all previous messages)
   - Prompt tokens: 380 (110 new tokens)
   - **KV cache benefit**: Can reuse 270 tokens from Turn 3

## Running the Example

Convert the trace to replay format:

```bash
python3 inference_perf/datagen/otel_trace_to_replay_csv.py \
  --input examples/trace-replay/test_traces/prefix_sharing_example.json \
  --out_csv examples/trace-replay/test_traces/prefix_sharing_example.csv \
  --out_payload_jsonl examples/trace-replay/test_traces/prefix_sharing_example_payloads.jsonl \
  --out_metadata_jsonl examples/trace-replay/test_traces/prefix_sharing_example_metadata.jsonl \
  --validate \
  --detect_prefix_sharing
```

## Expected Output

```
Statistics:
  Events with output: 7/7
  Events with errors: 0/7
  Total dependencies: 15
  Avg dependencies per event: 2.14

Prefix Sharing (KV Cache Opportunities):
  Events with prefix sharing: 3/7
  Total prefix sharing relationships: 6
  Avg prefix sharing per event: 2.00

Dependency Types:
  timestamp: 9
  sequential+prefix_sharing: 6
```

## Understanding the Results

### CSV Output

The generated CSV shows:
- **Independent calls**: Have `depends_on` based only on timestamps (no prefix sharing)
- **Conversation turns**: Have `sequential+prefix_sharing` dependencies

Example row for Turn 2:
```csv
span_conv_turn2,...,depends_on=span_conv_turn1,dependency_types=sequential+prefix_sharing,prefix_sharing_candidates=span_conv_turn1,...
```

### Metadata Output

The metadata JSONL shows detailed prefix sharing information:

```json
{
  "event_id": "span_conv_turn2",
  "dependency_types": ["sequential+prefix_sharing"],
  "prefix_sharing_candidates": ["span_conv_turn1"],
  "prefix_scores": {"span_conv_turn1": 1.0}
}
```

- `prefix_scores`: 1.0 means 100% prefix match
- `prefix_sharing_candidates`: Lists all events that share a prefix with this event

## KV Cache Benefits

In a real deployment with KV cache enabled:

| Turn | Total Tokens | New Tokens | Reused Tokens | Cache Hit Rate |
|------|--------------|------------|---------------|----------------|
| 1    | 100          | 100        | 0             | 0%             |
| 2    | 180          | 80         | 100           | 55.6%          |
| 3    | 270          | 90         | 180           | 66.7%          |
| 4    | 380          | 110        | 270           | 71.1%          |

**Total savings**: 550 tokens reused out of 930 total tokens = **59.1% cache hit rate**

## Key Insights

1. **Prefix sharing only works in one direction**: Turn 2 extends Turn 1, not vice versa
2. **Perfect prefix matches** (score 1.0) indicate ideal KV cache reuse opportunities
3. **Independent calls** have no prefix sharing, so they get full KV cache misses
4. **Multi-turn conversations** are the primary use case for KV cache optimization

## Visualization

```
Independent Calls (No Prefix Sharing):
  [Call 1] ──┐
  [Call 2] ──┼──> No shared context
  [Call 3] ──┘

Conversation Chain (With Prefix Sharing):
  [Turn 1] ──> [Turn 2] ──> [Turn 3] ──> [Turn 4]
     │           │            │            │
     └───────────┴────────────┴────────────┘
         Each turn extends all previous turns
         (100% prefix match on shared context)
```

## Use Cases

This pattern is common in:
- **Chatbots**: Multi-turn conversations with context
- **Code assistants**: Iterative code review/refinement
- **Document Q&A**: Multiple questions about the same document
- **Agent workflows**: Sequential reasoning steps with shared context

## Notes

- The `--detect_prefix_sharing` flag must be enabled to detect these patterns
- Prefix scores range from 0.0 (no overlap) to 1.0 (perfect match)
- The minimum prefix length threshold is 16 characters by default
- Normalization (whitespace collapsing) is applied before comparison