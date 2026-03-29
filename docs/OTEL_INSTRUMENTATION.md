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
  - Token usage (input/output tokens)
  - Latency metrics (TTFT, TPOT, total latency)
  - Finish reasons and response IDs
  - Error information
- **Support for all model servers**: vLLM, SGlang, TGI, and any OpenAI-compatible API
- **Graceful degradation**: Works without OTEL packages installed (disabled mode)

## Installation

Install the required OpenTelemetry packages:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-aiohttp-client opentelemetry-semantic-conventions-ai
```

Or install inference-perf with all dependencies:

```bash
pip install -e .
```

## Usage

### Basic Usage

OTEL instrumentation is **enabled by default** for all model server clients. No configuration changes are required.

```python
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient

# OTEL instrumentation is automatically enabled
client = vLLMModelServerClient(
    metrics_collector=collector,
    api_config=config,
    uri="http://localhost:8000",
    model_name="meta-llama/Llama-2-7b-hf",
    # ... other parameters
)
```

### Disabling OTEL

To disable OTEL instrumentation:

```python
client = vLLMModelServerClient(
    # ... other parameters
    enable_otel=False,  # Disable OTEL tracing
)
```

### Configuring OTEL Exporter

By default, traces are exported to the console. To export to an OTLP endpoint (e.g., Jaeger, Tempo):

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure OTLP exporter
provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # Your OTLP endpoint
    insecure=True,
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)

# Now create your client - it will use the configured provider
client = vLLMModelServerClient(...)
```

### Environment Variables

You can also configure OTEL using environment variables:

```bash
# OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Service name
export OTEL_SERVICE_NAME=inference-perf

# Sampling rate (0.0 to 1.0)
export OTEL_TRACES_SAMPLER=traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1  # Sample 10% of traces
```

## Span Attributes

The following span attributes are recorded for each LLM request:

### Request Attributes

- `gen_ai.system`: Always set to "openai_compatible"
- `gen_ai.request.model`: Model name being used
- `gen_ai.operation.name`: Operation type (e.g., "chat.completions", "completions")
- `gen_ai.request.max_tokens`: Maximum tokens to generate
- `gen_ai.request.temperature`: Temperature parameter (if set)
- `gen_ai.request.top_p`: Top-p parameter (if set)
- `gen_ai.request.stream`: Whether streaming is enabled

### Response Attributes

- `gen_ai.usage.input_tokens`: Number of input tokens
- `gen_ai.usage.output_tokens`: Number of output tokens
- `gen_ai.response.time_to_first_token`: Time to first token (seconds)
- `gen_ai.response.time_per_output_token`: Average time per output token (seconds)
- `gen_ai.response.total_latency`: Total request latency (seconds)
- `gen_ai.response.finish_reason`: Reason for completion (e.g., "stop", "length")
- `gen_ai.response.id`: Response ID from the API

### Error Attributes

- `gen_ai.response.error`: Error message (if request failed)
- Span status set to ERROR with exception details

## Example: Viewing Traces in Jaeger

1. Start Jaeger:
```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

2. Configure inference-perf to export to Jaeger:
```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry import trace

provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)
```

3. Run your benchmark

4. View traces at http://localhost:16686

## Architecture

The OTEL instrumentation is implemented in three layers:

1. **`otel_instrumentation.py`**: Core instrumentation module
   - `OTelInstrumentation` class for managing tracing
   - Context manager for creating spans
   - Methods for recording metrics

2. **`openai_client.py`**: Base client with OTEL integration
   - Initializes OTEL instrumentation
   - Wraps API calls with tracing spans
   - Records request/response metrics

3. **Model-specific clients**: Inherit OTEL support
   - `vllm_client.py`
   - `sglang_client.py`
   - `tgi_client.py`

## Performance Impact

The OTEL instrumentation has minimal performance impact:

- **Disabled mode**: Zero overhead when OTEL packages are not installed
- **Enabled mode**: ~1-2% overhead for span creation and attribute recording
- **Async-friendly**: Uses context managers and doesn't block request processing

## Troubleshooting

### OTEL packages not found

If you see warnings about missing OTEL packages:

```
OpenTelemetry packages not installed. Install with: pip install opentelemetry-api ...
```

Install the required packages or disable OTEL with `enable_otel=False`.

### Traces not appearing

1. Check that your OTLP endpoint is reachable
2. Verify the endpoint URL is correct
3. Check for firewall/network issues
4. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### High cardinality warnings

If you see warnings about high cardinality attributes, consider:
- Reducing sampling rate
- Filtering sensitive attributes
- Using span processors to drop unnecessary attributes

## Future Enhancements

Potential improvements for future versions:

- [ ] Support for custom span processors
- [ ] Integration with OpenTelemetry metrics
- [ ] Automatic correlation with Prometheus metrics
- [ ] Support for W3C Trace Context propagation
- [ ] Custom semantic conventions for inference-perf specific attributes

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Python OpenTelemetry SDK](https://opentelemetry-python.readthedocs.io/)