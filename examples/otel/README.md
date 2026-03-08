# OpenTelemetry Trace Replay

This directory contains configurations and test traces for replaying LLM workloads from OpenTelemetry traces.

## Overview

The OTEL trace replay system allows you to:
- Replay real-world LLM request patterns from production traces
- Test agentic workflows with complex dependencies
- Benchmark systems with realistic multi-turn conversations
- Validate KV cache optimizations with prefix sharing
- Simulate multi-agent systems with parallel execution

## Directory Structure

```
otel/
├── README.md                    # This file
├── configs/                     # Configuration files
│   ├── simple/                  # Simple use cases (learning)
│   │   ├── single-trace.yml     # Single trace replay
│   │   └── mock-simple.yml      # Simple mock client
│   └── advanced/                # Advanced use cases (testing)
│       ├── session-based.yml    # Session-based replay (recommended)
│       ├── multi-session.yml    # Multiple concurrent sessions
│       ├── mock-advanced.yml    # Advanced mock scenarios
│       └── graph-replay.yml     # Graph-based replay
└── test_traces/                 # Test trace files
    ├── simple/                  # Simple 3-node chains
    │   ├── simple_chain.json
    │   └── README.md
    └── advanced/                # Complex multi-agent scenarios
        ├── multi_agent_research.json
        ├── code_review_workflow.json
        ├── parallel_requests.json
        ├── fan_in_dependencies.json
        ├── prefix_sharing_example.json
        ├── DEPENDENCY_INFERENCE_EXPLAINED.md
        ├── COMPLEX_SCENARIOS.md
        └── README.md
```

## Quick Start

### 1. Simple Single Trace
Start with a basic 3-turn conversation:
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/simple/single-trace.yml
```

### 2. Advanced Session-Based Replay
Replay multiple complex traces with proper dependency handling:
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/advanced/session-based.yml
```

### 3. Mock Client Testing
Test without a real LLM server:
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/simple/mock-simple.yml
```

## Key Features

### Async Dependency Handling
- **No Deadlocks**: Workers use async waiting instead of blocking
- **Scalable**: Works with any number of workers
- **Efficient**: Multiple requests processed concurrently

### Output Substitution
- **Runtime Replacement**: Recorded assistant messages replaced with actual LLM outputs
- **Growing Context**: Maintains realistic multi-turn conversation patterns
- **KV Cache Friendly**: Ensures proper cache behavior

### Session-Based Dispatch
- **Trace-Level Control**: Dispatch entire sessions instead of individual events
- **Rate Limiting**: Control session start rate
- **Concurrency**: Manage concurrent session execution

## Configuration Options

### Data Generator Config
```yaml
data:
  type: otel_trace_replay
  otel_trace_replay:
    trace_directory: "path/to/traces"
    time_scale: 1.0                    # 1.0 = real-time, 0.5 = half speed
    use_static_model: true             # Override model from traces
    static_model_name: "model-name"
    default_max_tokens: 100
    dependency_window_ms: 120000       # 2 minutes
    max_dependencies_per_event: 3
    validate_dependencies: true
```

### Load Config (Session-Based)
```yaml
load:
  type: trace_session_replay
  stages:
  - session_rate: 2.0      # Start 2 sessions/sec
    duration: 30           # For 30 seconds
  
  # OR count-based:
  # - num_sessions: 100
  #   max_concurrent_sessions: 10
  
  num_workers: 1
  worker_max_concurrency: 10
```

## Trace File Format

Each trace file is a JSON object:
```json
{
  "trace_id": "unique_id",
  "span_count": 3,
  "collected_at": "2026-01-01T10:00:00.000000",
  "spans": [
    {
      "trace_id": "unique_id",
      "span_id": "span_001",
      "start_time": "2026-01-01T10:00:00.000000",
      "end_time": "2026-01-01T10:00:02.000000",
      "attributes": {
        "gen_ai.request.model": "gpt-4",
        "gen_ai.input.messages": "[{\"role\": \"user\", \"content\": \"Hello\"}]",
        "gen_ai.output.text": "Hi there!"
      }
    }
  ]
}
```

## Dependency Inference

The system automatically infers dependencies between LLM calls:

1. **Causal Dependencies**: When output of call A appears in input of call B
2. **Temporal Dependencies**: Fallback based on timing when no causal link exists
3. **Prefix Sharing**: Detected when calls share common message prefixes

See `test_traces/advanced/DEPENDENCY_INFERENCE_EXPLAINED.md` for details.

## Performance Tips

### Worker Configuration
- **Single Worker**: Good for simple traces, prevents over-parallelization
- **Multiple Workers**: Better for complex traces with many parallel branches
- **Concurrency**: Set `worker_max_concurrency` based on server capacity

### Rate Limiting
- **session_rate**: Control how fast sessions start
- **time_scale**: Speed up/slow down replay timing
- **max_concurrent_sessions**: Limit concurrent session execution

### Memory Management
- Large traces with many nodes may require increased worker memory
- Use `skip_invalid_files: true` to handle malformed traces gracefully

## Troubleshooting

### Issue: Nodes waiting indefinitely
**Solution**: Check that:
- All predecessor nodes complete successfully
- NodeOutputRegistry is properly shared (mp_manager)
- Timeout is sufficient (default: 5 minutes)

### Issue: Output substitution not working
**Solution**: Verify:
- `process_response` registers outputs
- Input segments correctly identify output sources
- Registry is shared across workers

### Issue: High memory usage
**Solution**:
- Reduce `max_concurrent_sessions`
- Lower `worker_max_concurrency`
- Use smaller trace files

## Examples

### Simple Learning Path
1. `simple/single-trace.yml` - Basic 3-turn conversation
2. `advanced/session-based.yml` - Multiple sessions with dependencies
3. `advanced/multi-session.yml` - Concurrent session execution

### Testing Patterns
1. **Linear Chains**: `simple_chain.json`
2. **Parallel Execution**: `parallel_requests.json`
3. **Multi-Agent**: `multi_agent_research.json`
4. **Complex Workflows**: `code_review_workflow.json`

## Creating Custom Traces

To create your own traces:
1. Instrument your application with OpenTelemetry
2. Export traces to JSON format
3. Place in `test_traces/simple/` or `test_traces/advanced/`
4. Create a config file pointing to your trace directory
5. Run with `inference_perf.main`

See existing traces as templates for the required structure.

## Related Documentation

- **Implementation Guide**: `../../../OTEL_TRACE_REPLAY_IMPLEMENTATION.md`
- **Test Suite**: `../../../tests/test_otel_trace_to_replay_graph.py`
- **Data Generator**: `../../../inference_perf/datagen/otel_trace_replay_datagen.py`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test traces for examples
3. Examine the implementation guide
4. Open an issue with trace details and logs