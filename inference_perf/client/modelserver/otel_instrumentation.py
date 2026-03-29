# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenTelemetry instrumentation for LLM API calls.

This module provides standard GenAI OTEL instrumentation following the
OpenTelemetry Semantic Conventions for GenAI operations.
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.semconv_ai import SpanAttributes
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    Span = None  # type: ignore
    SpanAttributes = None  # type: ignore

logger = logging.getLogger(__name__)


class OTelInstrumentation:
    """
    OpenTelemetry instrumentation for LLM API calls.
    
    Provides tracing capabilities following GenAI semantic conventions.
    """
    
    def __init__(
        self,
        service_name: str = "inference-perf",
        enabled: bool = True,
        otlp_endpoint: Optional[str] = None,
    ):
        """
        Initialize OTEL instrumentation.
        
        Args:
            service_name: Name of the service for tracing
            enabled: Whether to enable OTEL instrumentation
            otlp_endpoint: OTLP endpoint for exporting traces (e.g., "http://localhost:4317")
                          If None, uses console exporter. If set, uses OTLP exporter.
        """
        self.enabled = enabled and OTEL_AVAILABLE
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.tracer: Optional[Any] = None
        
        if not OTEL_AVAILABLE and enabled:
            logger.warning(
                "OpenTelemetry packages not installed. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-instrumentation-aiohttp-client opentelemetry-semantic-conventions-ai"
            )
            self.enabled = False
        
        if self.enabled:
            self._setup_tracer()
    
    def _setup_tracer(self) -> None:
        """Set up the OpenTelemetry tracer."""
        if not OTEL_AVAILABLE:
            return
            
        # Get or create tracer provider
        provider = trace.get_tracer_provider()
        
        # If no provider is set, create a default one
        if not hasattr(provider, 'add_span_processor'):
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME
            
            # Create resource with service name
            resource = Resource(attributes={SERVICE_NAME: self.service_name})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            
            # Configure exporter based on otlp_endpoint
            if self.otlp_endpoint:
                # Use OTLP exporter for Jaeger/other backends
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    
                    otlp_exporter = OTLPSpanExporter(
                        endpoint=self.otlp_endpoint,
                        insecure=True,  # Use insecure connection by default
                    )
                    # Use BatchSpanProcessor with shorter intervals for better reliability
                    batch_processor = BatchSpanProcessor(
                        otlp_exporter,
                        max_queue_size=2048,
                        schedule_delay_millis=1000,  # Export every 1 second
                        max_export_batch_size=512,
                    )
                    provider.add_span_processor(batch_processor)
                    logger.info(f"Created OTEL tracer provider with OTLP exporter to {self.otlp_endpoint}")
                except ImportError:
                    logger.warning(
                        "OTLP exporter not available. Install with: "
                        "pip install opentelemetry-exporter-otlp-proto-grpc"
                    )
                    logger.info("Falling back to console exporter")
                    console_exporter = ConsoleSpanExporter()
                    provider.add_span_processor(SimpleSpanProcessor(console_exporter))
            else:
                # Use console exporter for debugging
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(console_exporter))
                logger.info("Created OTEL tracer provider with console exporter")
        
        self.tracer = trace.get_tracer(self.service_name)
        self._provider = provider  # Store provider for shutdown
        logger.info(f"OTEL instrumentation enabled for service: {self.service_name}")

    def shutdown(self):
        """Shutdown the tracer provider and flush all pending spans."""
        if hasattr(self, '_provider') and self._provider:
            try:
                self._provider.force_flush(timeout_millis=5000)
                self._provider.shutdown()
                logger.info("OTEL tracer provider shutdown successfully")
            except Exception as e:
                logger.warning(f"Error during OTEL shutdown: {e}")
    
    @contextmanager
    def trace_llm_request(
        self,
        operation_name: str,
        model_name: str,
        request_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing LLM requests.
        
        Args:
            operation_name: Name of the operation (e.g., "chat.completions", "completions")
            model_name: Name of the model being used
            request_data: Optional request data for additional context
            
        Yields:
            Span object if OTEL is enabled, None otherwise
        """
        if not self.enabled or self.tracer is None:
            yield None
            return
        
        with self.tracer.start_as_current_span(
            f"llm.{operation_name}",
            kind=trace.SpanKind.CLIENT
        ) as span:
            try:
                # Set standard GenAI attributes using semantic conventions
                if SpanAttributes:
                    # Core GenAI attributes
                    span.set_attribute(SpanAttributes.LLM_SYSTEM, "openai_compatible")
                    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_name)
                    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, operation_name)
                    
                    # Add request-specific attributes if available
                    if request_data:
                        if "max_tokens" in request_data:
                            span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_data["max_tokens"])
                        if "temperature" in request_data:
                            span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, request_data["temperature"])
                        if "top_p" in request_data:
                            span.set_attribute(SpanAttributes.LLM_REQUEST_TOP_P, request_data["top_p"])
                        if "stream" in request_data:
                            span.set_attribute(SpanAttributes.LLM_IS_STREAMING, request_data["stream"])
                
                yield span
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def record_response_metrics(
        self,
        span: Optional[Any],
        response_info: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record response metrics on the span.
        
        Args:
            span: The OTEL span to record metrics on
            response_info: Response information including token counts, latency, etc.
            error: Error message if the request failed
        """
        if not self.enabled or span is None:
            return
        
        try:
            if error:
                span.set_status(Status(StatusCode.ERROR, error))
                # Note: There's no standard attribute for error in the semantic conventions yet
                span.set_attribute("error.message", error)
            else:
                span.set_status(Status(StatusCode.OK))
            
            if response_info and SpanAttributes:
                # Token usage
                if "prompt_tokens" in response_info:
                    span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, response_info["prompt_tokens"])
                if "completion_tokens" in response_info:
                    span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, response_info["completion_tokens"])
                
                # Calculate total tokens if both are available
                if "prompt_tokens" in response_info and "completion_tokens" in response_info:
                    total_tokens = response_info["prompt_tokens"] + response_info["completion_tokens"]
                    span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
                
                # Latency metrics (custom attributes as they're not in standard semantic conventions)
                if "time_to_first_token" in response_info:
                    span.set_attribute("gen_ai.response.time_to_first_token", response_info["time_to_first_token"])
                if "time_per_output_token" in response_info:
                    span.set_attribute("gen_ai.response.time_per_output_token", response_info["time_per_output_token"])
                if "total_latency" in response_info:
                    span.set_attribute("gen_ai.response.total_latency", response_info["total_latency"])
                
                # Finish reason
                if "finish_reason" in response_info:
                    span.set_attribute(SpanAttributes.LLM_RESPONSE_FINISH_REASON, response_info["finish_reason"])
                
                # Input messages (gen_ai.input.messages) - stored as JSON string
                if "input_messages" in response_info:
                    span.set_attribute("gen_ai.input.messages", response_info["input_messages"])
                
                # Output text (gen_ai.output.text)
                if "output_text" in response_info:
                    span.set_attribute("gen_ai.output.text", response_info["output_text"])
                
                # Response ID (custom attribute)
                if "response_id" in response_info:
                    span.set_attribute("gen_ai.response.id", response_info["response_id"])
                
        except Exception as e:
            logger.warning(f"Failed to record response metrics: {e}")


# Global instance
_global_instrumentation: Optional[OTelInstrumentation] = None


def get_otel_instrumentation(
    service_name: str = "inference-perf",
    enabled: bool = True,
    otlp_endpoint: Optional[str] = None,
) -> OTelInstrumentation:
    """
    Get or create the global OTEL instrumentation instance.
    
    Args:
        service_name: Name of the service for tracing
        enabled: Whether to enable OTEL instrumentation
        otlp_endpoint: OTLP endpoint for exporting traces (e.g., "http://localhost:4317")
        
    Returns:
        OTelInstrumentation instance
    """
    global _global_instrumentation
    
    if _global_instrumentation is None:
        # Check environment variable for OTLP endpoint
        import os
        if otlp_endpoint is None:
            otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        
        _global_instrumentation = OTelInstrumentation(
            service_name=service_name,
            enabled=enabled,
            otlp_endpoint=otlp_endpoint,
        )
    
    return _global_instrumentation


def configure_otel(
    service_name: str = "inference-perf",
    enabled: bool = True,
    otlp_endpoint: Optional[str] = None,
) -> None:
    """
    Configure global OTEL instrumentation.
    
    Args:
        service_name: Name of the service for tracing
        enabled: Whether to enable OTEL instrumentation
        otlp_endpoint: OTLP endpoint for exporting traces (e.g., "http://localhost:4317")
    """
    global _global_instrumentation
    _global_instrumentation = OTelInstrumentation(
        service_name=service_name,
        enabled=enabled,
        otlp_endpoint=otlp_endpoint,
    )

# Made with Bob
