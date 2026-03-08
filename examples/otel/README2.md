# OTel Trace Replay

This directory contains tools and examples for replaying LLM workloads from OpenTelemetry (OTel) trace files.

## Overview

The OTel Trace Replay feature allows you to:
- **Capture** real-world LLM workloads using OpenTelemetry instrumentation
- **Process** trace files to extract LLM calls and infer dependencies
- **Replay** the workload on a target inference server with proper timing and dependency enforcement
- **Benchmark** different models, configurations, or infrastructure using real production patterns

## Quick Start

### 1. Prepare OTel Trace Files

Place your OTel JSON trace files in a directory (e.g., `test_traces/`). Each file should contain spans with LLM call information.

Example trace structure:
```json
{
  "trace_id": "...",
  "spans": [
    {
      "span_id": "...",
      "name": "chat gpt-4",
      "start_time": "2026-02-24T11:11:34.000000",
      "end_time": "2026-02-24T11:11:36.000000",
      "attributes": {
        "exgentic.session.id": "session_001",
        "gen_ai.request.model": "gpt-4",
        "gen_ai.input.messages": "[{\"role\": \"user\", \"content\": \"...\"}]",
        "gen_ai.output.text": "..."
      }
    }
  ]
}
```

### 2. Configure Replay

Create a configuration file (see `otel-trace-replay-config.yml`):

```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_directory: "path/to/traces"
    concurrent_sessions: 2
    time_scale: 1.0
    use_static_model: true
    static_model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

### 3. Run Replay

```bash
inference-perf --config examples/trace-replay/otel-trace-replay-config.yml
```

## Configuration Options

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace_directory` | string | required | Directory containing OTel JSON files |
| `concurrent_sessions` | int | 1 | Number of sessions to replay in parallel (0 = all) |
| `session_start_delay_sec` | float | 0.0 | Delay between session batch starts |
| `time_scale` | float | 1.0 | Time scale factor (0.5 = half speed, 2.0 = double speed) |

### Model Configuration

**Option 1: Static Model** (all requests use same model)
```yaml
use_static_model: true
static_model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

**Option 2: Model Mapping** (map recorded models to target models)
```yaml
use_static_model: false
model_mapping:
  "gpt-4": "meta-llama/Llama-3.1-70B-Instruct"
  "gpt-3.5-turbo": "meta-llama/Llama-3.1-8B-Instruct"
```

### Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_max_tokens` | int | 1000 | Default max_tokens if not in trace |
| `dependency_window_ms` | int | 120000 | Time window for dependency inference (ms) |
| `max_dependencies_per_event` | int | 3 | Maximum dependencies per request |
| `include_errors` | bool | true | Include spans with error status |
| `skip_invalid_files` | bool | true | Skip invalid files instead of failing |
| `validate_dependencies` | bool | false | Validate dependency graph and log warnings |

## How It Works

### 1. Trace Processing

Each OTel JSON file is processed to:
- Extract LLM call spans (identified by `gen_ai.input.messages` or `chat` prefix in name)
- Determine session ID (from `exgentic.session.id` attribute)
- Extract request parameters (model, messages, max_tokens, temperature)
- Record timing information (start/end timestamps)

### 2. Dependency Inference

Dependencies between requests are inferred using:

**Timestamp Analysis:**
- Events within a time window (default: 2 minutes) are considered
- Overlapping events don't force ordering
- Non-overlapping events create sequential dependencies

**Output→Input Containment:**
- If request B's input contains text from request A's output, B depends on A
- Uses shingle matching (8 substrings of 60 characters)
- Threshold: 50% of shingles must match

**Example:**
```
Request A: "Analyze this code..."
  Output: "The code has 3 issues: 1) SQL injection..."

Request B: "Fix these issues: 1) SQL injection..."
  → B depends on A (output text found in input)
```

### 3. Replay Scheduling

Requests are replayed with:
- **Wall-clock timing**: Maintains original time intervals (scaled by `time_scale`)
- **Dependency enforcement**: Request B waits for A to complete if B depends on A
- **Concurrent sessions**: Multiple trace files replay in parallel
- **Staggered starts**: Sessions start with configurable delays

## Tools

### otel_trace_to_replay_csv.py

Standalone script to convert OTel traces to CSV format for analysis:

```bash
python inference_perf/datagen/otel_trace_to_replay_csv.py \
  --input trace.json \
  --out_csv replay.csv \
  --out_payload_jsonl payloads.jsonl \
  --include_errors \
  --validate
```

**Options:**
- `--input`: OTel JSON trace file
- `--out_csv`: Output CSV file with events and dependencies
- `--out_payload_jsonl`: Output JSONL file with request payloads
- `--session_id`: Specific session to extract (optional)
- `--include_errors`: Include spans with error status
- `--validate`: Validate dependency graph
- `--window_ms`: Dependency inference window (default: 120000)
- `--max_deps`: Max dependencies per event (default: 3)

**Output CSV Schema:**
```csv
event_id,session_id,trace_id,parent_span_id,t_start_ms,t_end_ms,depends_on,model_recorded,payload_ref,has_output,error_type,status_code,turn_number
span_001,session_001,trace_001,,0,2500,,gpt-4,span_001,1,,,0
span_002,session_001,trace_001,,3000,5000,span_001,gpt-4,span_002,1,,,1
```

## Test Traces

The `test_traces/` directory contains example traces for testing:

### Simple Traces
- **simple_chain.json**: Sequential 3-request conversation
- **parallel_requests.json**: Fork-join pattern with parallel branches
- **fan_in_dependencies.json**: Multiple independent roots converging

### Complex Multi-Agent Scenarios
- **multi_agent_research.json**: Multi-phase research workflow (12 events)
  - Coordinator → 3 parallel agents → synthesis → follow-up
- **code_review_workflow.json**: Iterative code review (15 events)
  - 3 review agents → synthesis → 3 refactor agents → synthesis → 3 test agents → synthesis

Run tests:
```bash
python examples/trace-replay/test_traces/test_dependency_inference.py
```

## Use Cases

### 1. Model Comparison
Compare different models on the same real-world workload:
```yaml
# Run 1: GPT-4 equivalent
static_model_name: "meta-llama/Llama-3.1-70B-Instruct"

# Run 2: GPT-3.5 equivalent  
static_model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

### 2. Infrastructure Testing
Test different serving configurations:
- Batch sizes
- Tensor parallelism
- Prefix caching settings
- Quantization methods

### 3. Capacity Planning
Determine required resources for production workloads:
```yaml
concurrent_sessions: 10  # Simulate 10 concurrent users
time_scale: 0.5  # Run at half speed for stress testing
```

### 4. Regression Testing
Validate that changes don't degrade performance on real workloads.

## Best Practices

### Trace Collection
1. **Include outputs**: Capture LLM response text for accurate dependency inference
2. **Use session IDs**: Group related requests with `exgentic.session.id`
3. **Record metadata**: Include model names, token counts, and timing info
4. **Handle errors**: Mark failed requests with appropriate status codes

### Replay Configuration
1. **Start small**: Test with `concurrent_sessions: 1` and `time_scale: 1.0`
2. **Validate dependencies**: Use `validate_dependencies: true` initially
3. **Monitor resources**: Watch server CPU/memory during replay
4. **Adjust timing**: Use `time_scale` to match target server capacity

### Troubleshooting
- **No dependencies detected**: Check if traces include output text
- **Cycles in graph**: Review `validate_dependencies` warnings
- **Missing events**: Verify `include_errors` setting
- **Slow replay**: Increase `time_scale` or reduce `concurrent_sessions`

## Architecture

```
OTel Trace Files
    ↓
[otel_trace_to_replay_csv.py]
    ↓
Events + Dependencies
    ↓
[OTelTraceReplayDataGenerator]
    ↓
Scheduled Requests
    ↓
[LoadGenerator]
    ↓
Target Inference Server
```

## References

- [OTel Trace Replay Plan](../../otel_trace_replay_csv_plan.md)
- [Test Traces Documentation](test_traces/README.md)
- [Complex Scenarios Analysis](test_traces/COMPLEX_SCENARIOS.md)