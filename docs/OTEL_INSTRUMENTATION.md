# OpenTelemetry Instrumentation for inference-perf

This document describes the OpenTelemetry (OTEL) instrumentation added to inference-perf for tracing LLM API calls.

## Overview

The OTEL instrumentation provides distributed tracing capabilities for LLM inference requests, following the [OpenTelemetry Semantic Conventions for GenAI operations](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

## Features

- **Automatic tracing** of all LLM API calls (chat completions, completions)
- **Standard GenAI semantic conventions** for consistent observability
- **Rich span attributes** including:
  - Model name and operation type
  - Request parameters (max_tokens, temperature, top_p, streaming)
  - Input messages and output text
  - Token usage (input/output tokens)
  - Latency metrics (TTFT, TPOT, total latency)
  - Finish reasons and error information
- **Support for all model servers**: vLLM, SGlang, TGI, and any OpenAI-compatible API
- **Environment-based configuration**: No code changes required
- **Graceful degradation**: Works without OTEL packages installed (disabled mode)

## Installation


```bash
pip install -e ".[otel]"
```

## Configuration

OTEL instrumentation is controlled via environment variables. Tracing is **disabled by default** and must be explicitly enabled.

### Environment Variables

- `OTEL_TRACES_ENABLED`: Set to `"true"` to enable OTEL tracing (default: `"false"`)
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint for exporting traces (e.g., `http://localhost:4317`)
  - If not set, traces are exported to console (stdout)
  - If set, traces are exported via OTLP to the specified endpoint
- `OTEL_SERVICE_NAME`: Service name for tracing (default: `"inference-perf"`)
- `OTEL_TRACE_PER_STAGE`: Set to `"true"` to create one trace per stage instead of per session (default: `"false"`)
  - When enabled, all sessions within a stage are grouped under a single stage-level trace
  - Session spans are created as children of the stage span
  - Useful for viewing all sessions in a stage together in trace visualization tools

## Usage

### Enable Tracing

To enable OTEL tracing, set the `OTEL_TRACES_ENABLED` environment variable:

```bash
export OTEL_TRACES_ENABLED="true"
python -m inference_perf.main --config config.yml
```

### Console Output

When `OTEL_EXPORTER_OTLP_ENDPOINT` is not set, traces are printed to console in JSON format:

```bash
export OTEL_TRACES_ENABLED="true"
python -m inference_perf.main --config config.yml
```

### Export to OTLP Endpoint

To export traces to an OTLP endpoint (e.g., Jaeger, Tempo, Grafana Cloud):

```bash
export OTEL_TRACES_ENABLED="true"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="my-inference-service"
python -m inference_perf.main --config config.yml
```

### Using with Jaeger

See [examples/otel/README_JAEGER.md](../examples/otel/README_JAEGER.md) for detailed instructions on using OTEL with Jaeger.

Quick start:

```bash
# Start Jaeger
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Run with tracing enabled
export OTEL_TRACES_ENABLED="true"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
python -m inference_perf.main --config config.yml

# View traces at http://localhost:16686
```

## Span Attributes

The following GenAI semantic convention attributes are captured:

### Request Attributes
- `gen_ai.system`: System identifier (e.g., "openai_compatible")
- `gen_ai.request.model`: Model name
- `gen_ai.request.max_tokens`: Maximum tokens to generate
- `gen_ai.request.temperature`: Sampling temperature
- `gen_ai.input.messages`: Input messages as JSON string

### Response Attributes
- `gen_ai.output.text`: Generated text
- `gen_ai.usage.prompt_tokens`: Number of input tokens
- `gen_ai.usage.completion_tokens`: Number of output tokens
- `gen_ai.response.total_latency`: Total request latency
- `gen_ai.response.time_to_first_token`: Time to first token (TTFT)
- `gen_ai.response.time_per_output_token`: Time per output token (TPOT)
- `gen_ai.response.finish_reason`: Reason for completion

### Additional Attributes
- `llm.request.type`: Operation type (e.g., "chat.completions")
- `llm.is_streaming`: Whether the request is streaming
- `llm.usage.total_tokens`: Total tokens (input + output)

## Architecture

The OTEL instrumentation is implemented in `inference_perf/client/modelserver/otel_instrumentation.py` and automatically integrated into all model server clients:

- `openai_client.py`: Base OpenAI-compatible client
- `vllm_client.py`: vLLM-specific client
- `sglang_client.py`: SGlang-specific client
- `tgi_client.py`: TGI-specific client

All clients automatically use the global OTEL instrumentation instance, which is configured via environment variables.

## Troubleshooting

### Traces not appearing

1. Verify OTEL is enabled:
   ```bash
   echo $OTEL_TRACES_ENABLED  # Should be "true"
   ```

2. Check OTLP endpoint is accessible:
   ```bash
   curl http://localhost:4317
   ```

3. Look for OTEL initialization messages in logs:
   ```
   INFO - OTEL tracing enabled for service: inference-perf
   INFO - Created OTEL tracer provider with OTLP exporter to http://localhost:4317
   ```

### Missing attributes

Ensure you're using the latest version of `opentelemetry-semantic-conventions-ai`:

```bash
pip install --upgrade opentelemetry-semantic-conventions-ai
```

## Examples

See the `examples/otel/` directory for complete examples:

- `examples/otel/configs/`: Example configuration files
- `examples/otel/test_traces/`: Sample trace data
- `examples/otel/run_with_jaeger.sh`: Script to run with Jaeger
- `examples/otel/README_JAEGER.md`: Detailed Jaeger integration guide
