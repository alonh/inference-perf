# Simple OTEL Test Traces

This directory contains simple OpenTelemetry trace files for basic testing and learning.

## Files

### simple_chain.json
A basic 3-turn conversational chain demonstrating sequential dependencies:
- **Nodes**: 3
- **Pattern**: Linear chain (A → B → C)
- **Use case**: Basic multi-turn conversation
- **Complexity**: ⭐ Beginner

**Structure**:
1. User asks: "What is the capital of France?"
2. Assistant responds, user follows up: "Tell me about Paris landmarks"
3. Assistant responds, user asks: "How tall is the Eiffel Tower?"

**Key Features**:
- Sequential dependencies (each turn depends on previous)
- Growing context (messages accumulate)
- Output substitution (actual LLM outputs replace recorded ones)

## Usage

### With Simple Config
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/simple/single-trace.yml
```

### With Mock Client
```bash
python -m inference_perf.main \
  --config examples/trace-replay/otel/configs/simple/mock-simple.yml
```

## Learning Path

1. **Start here**: Understand basic trace structure and dependencies
2. **Next**: Explore advanced traces in `../advanced/` directory
3. **Then**: Create your own traces using the patterns learned

## Trace Format

Each trace file is a JSON object with:
- `trace_id`: Unique identifier for the trace
- `span_count`: Number of LLM calls in the trace
- `spans`: Array of span objects with timing, messages, and outputs

See the test file for detailed structure examples.