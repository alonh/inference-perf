#!/bin/bash
# Script to run inference-perf with Jaeger tracing enabled
#
# This script configures OpenTelemetry environment variables to export
# traces to Jaeger via OTLP protocol.
#
# Prerequisites:
# 1. Start Jaeger with OTLP support:
#    docker run -d --name jaeger \
#      -e COLLECTOR_OTLP_ENABLED=true \
#      -p 16686:16686 \
#      -p 4317:4317 \
#      -p 4318:4318 \
#      jaegertracing/all-in-one:latest
#
# 2. Install OTLP exporter:
#    pip install opentelemetry-exporter-otlp-proto-grpc
#
# Usage:
#    ./examples/otel/run_with_jaeger.sh

# Set OpenTelemetry environment variables
export OTEL_TRACES_ENABLED="true"
export OTEL_SERVICE_NAME="inference-perf"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_EXPORTER_OTLP_INSECURE="true"

# Optional: Configure sampling (1.0 = 100% of traces)
export OTEL_TRACES_SAMPLER="always_on"

echo "=================================================="
echo "Running inference-perf with Jaeger tracing"
echo "=================================================="
echo "Service Name: $OTEL_SERVICE_NAME"
echo "OTLP Endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT"
echo "Jaeger UI: http://localhost:16686"
echo "=================================================="
echo ""

# Check if Jaeger is running
if ! curl -s http://localhost:16686 > /dev/null 2>&1; then
    echo "⚠️  Warning: Jaeger UI not accessible at http://localhost:16686"
    echo "   Start Jaeger with:"
    echo "   docker run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest"
    echo ""
fi

# Run inference-perf
python -m inference_perf.main --config examples/otel/configs/per_case_config/simple_chain.yml

echo ""
echo "=================================================="
echo "✓ Benchmark complete!"
echo "View traces at: http://localhost:16686"
echo "=================================================="

# Made with Bob
