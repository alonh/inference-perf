# Jaeger Integration for inference-perf

This guide shows how to send OpenTelemetry traces from inference-perf to Jaeger for visualization and analysis.

## Quick Start

### 1. Start Jaeger

Start Jaeger with OTLP support using Docker:

```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Verify Jaeger is running by opening http://localhost:16686 in your browser.

### 2. Install OTLP Exporter

```bash
pip install opentelemetry-exporter-otlp-proto-grpc
```

Or if using the project:
```bash
pip install -e .
```

### 3. Run with Jaeger

**Option A: Using environment variable**

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
python -m inference_perf.main --config examples/otel/configs/per_case_config/simple_chain.yml
```

**Option B: Using the provided script**

```bash
chmod +x examples/otel/run_with_jaeger.sh
./examples/otel/run_with_jaeger.sh
```

**Option C: Programmatically**

```python
from inference_perf.client.modelserver.otel_instrumentation import configure_otel

# Configure OTLP exporter before creating clients
configure_otel(
    service_name="inference-perf",
    enabled=True,
    otlp_endpoint="http://localhost:4317"
)

# Now run your benchmark...
```

### 4. View Traces

Open Jaeger UI at http://localhost:16686 and:
1. Select "inference-perf" from the Service dropdown
2. Click "Find Traces"
3. Click on a trace to see detailed span information

## Trace Attributes

Each LLM request span includes the following attributes following OpenTelemetry semantic conventions:

### Request Attributes
- `gen_ai.system`: "openai_compatible"
- `gen_ai.request.model`: Model name
- `llm.request.type`: Operation type (e.g., "chat.completions")
- `gen_ai.request.max_tokens`: Maximum tokens to generate
- `llm.is_streaming`: Whether streaming is enabled
- `gen_ai.request.temperature`: Temperature parameter (if set)
- `gen_ai.request.top_p`: Top-p parameter (if set)

### Response Attributes
- `gen_ai.usage.prompt_tokens`: Number of input tokens
- `gen_ai.usage.completion_tokens`: Number of output tokens
- `llm.usage.total_tokens`: Total tokens (input + output)
- `gen_ai.response.total_latency`: Total request latency in seconds
- `gen_ai.response.time_to_first_token`: TTFT in seconds (if available)
- `gen_ai.response.time_per_output_token`: Average TPOT in seconds (if available)
- `gen_ai.response.finish_reason`: Completion finish reason

## Advanced Configuration

### Custom OTLP Endpoint

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://my-otel-collector:4317"
python -m inference_perf.main --config your_config.yml
```

### Sampling

To reduce trace volume, configure sampling:

```bash
# Sample 10% of traces
export OTEL_TRACES_SAMPLER="traceidratio"
export OTEL_TRACES_SAMPLER_ARG="0.1"
```

### TLS/Authentication

For production deployments with TLS:

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry import trace

# Create OTLP exporter with TLS
otlp_exporter = OTLPSpanExporter(
    endpoint="https://my-otel-collector:4317",
    insecure=False,  # Use TLS
    headers=(("authorization", "Bearer YOUR_TOKEN"),),
)

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)
```

## Troubleshooting

### Traces not appearing in Jaeger

1. **Check Jaeger is running**:
   ```bash
   curl http://localhost:16686
   ```

2. **Check OTLP endpoint is accessible**:
   ```bash
   curl http://localhost:4317
   ```

3. **Verify OTLP exporter is installed**:
   ```bash
   python -c "from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter"
   ```

4. **Check logs for errors**:
   Look for OTEL-related log messages in the inference-perf output

### Connection refused errors

- Ensure Jaeger is running and OTLP port (4317) is exposed
- Check firewall settings
- Verify the endpoint URL is correct

### High memory usage

- Reduce sampling rate using `OTEL_TRACES_SAMPLER`
- Use `BatchSpanProcessor` instead of `SimpleSpanProcessor` (default for OTLP)

## Example Queries in Jaeger

### Find slow requests
1. In Jaeger UI, go to "Search" tab
2. Select service: "inference-perf"
3. Set "Min Duration" to filter slow traces
4. Click "Find Traces"

### Analyze token usage
1. Click on a trace
2. Expand the span details
3. Look for `gen_ai.usage.*` attributes

### Compare models
1. Search for traces with different `gen_ai.request.model` values
2. Compare latency and token usage

## Integration with Other Tools

### Grafana Tempo

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://tempo:4317"
```

### Honeycomb

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io:443"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=YOUR_API_KEY"
```

### AWS X-Ray

Use the AWS Distro for OpenTelemetry (ADOT) Collector as an intermediary.

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)