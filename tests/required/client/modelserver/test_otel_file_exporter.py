# Copyright 2026 The Kubernetes Authors.
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
"""Unit tests for the Jaeger JSON file span exporter."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.trace import Status, StatusCode  # noqa: E402

from inference_perf.client.modelserver.otel_file_exporter import JaegerJsonFileExporter, merge_parts  # noqa: E402


def _provider(path: Path) -> tuple[TracerProvider, JaegerJsonFileExporter]:
    provider = TracerProvider(resource=Resource(attributes={SERVICE_NAME: "inference-perf"}), shutdown_on_exit=False)
    exporter = JaegerJsonFileExporter(str(path))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _emit_session(provider: TracerProvider, session_id: str, error: bool = False) -> None:
    tracer = provider.get_tracer("inference-perf")
    session = tracer.start_span(f"session.{session_id}", kind=trace.SpanKind.INTERNAL)
    session.set_attribute("session.id", session_id)
    session.set_attribute("session.num_events", 2)
    ctx = trace.set_span_in_context(session)
    for i in range(2):
        with tracer.start_as_current_span("llm.chat.completions", kind=trace.SpanKind.CLIENT, context=ctx) as span:
            span.set_attribute("gen_ai.usage.prompt_tokens", 10 + i)
            span.set_attribute("gen_ai.usage.completion_tokens", 5)
            span.set_attribute("gen_ai.response.total_latency", 0.5)
            span.set_attribute("gen_ai.response.time_to_first_token", 0.1)
            span.set_attribute("gen_ai.request.stop", ["a", "b"])
            span.set_attribute("gen_ai.input.messages", json.dumps([{"role": "user", "content": "hi"}]))
            span.set_attribute("gen_ai.output.text", "hello")
            if error and i == 1:
                span.set_status(Status(StatusCode.ERROR, "boom"))
            else:
                span.set_status(Status(StatusCode.OK))
    session.set_status(Status(StatusCode.OK))
    session.end()


def _tags(span: Dict[str, Any]) -> Dict[str, Any]:
    return {t["key"]: t["value"] for t in span["tags"]}


def test_writes_jaeger_document(tmp_path: Path) -> None:
    out = tmp_path / "traces.json"
    provider, exporter = _provider(out)
    _emit_session(provider, "s1")
    _emit_session(provider, "s2", error=True)
    provider.shutdown()

    assert not exporter.parts_dir.exists()
    data = json.loads(out.read_text())["data"]
    assert len(data) == 2
    for t in data:
        assert t["processes"]["p1"]["serviceName"] == "inference-perf"
        session = next(s for s in t["spans"] if s["operationName"].startswith("session."))
        llm = [s for s in t["spans"] if s["operationName"] == "llm.chat.completions"]
        assert len(llm) == 2
        for s in llm:
            assert s["traceID"] == t["traceID"]
            assert s["references"] == [{"refType": "CHILD_OF", "traceID": t["traceID"], "spanID": session["spanID"]}]
            assert len(s["spanID"]) == 16 and len(s["traceID"]) == 32
            tags = _tags(s)
            assert tags["span.kind"] == "client"
            assert tags["gen_ai.usage.prompt_tokens"] in (10, 11)
            assert json.loads(tags["gen_ai.request.stop"]) == ["a", "b"]
        assert _tags(session)["session.num_events"] == 2

    errored = [s for t in data for s in t["spans"] if _tags(s).get("otel.status_code") == "ERROR"]
    assert len(errored) == 1
    assert _tags(errored[0])["error"] is True
    assert _tags(errored[0])["otel.status_description"] == "boom"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_forked_child_spans_are_merged(tmp_path: Path) -> None:
    out = tmp_path / "traces.json"
    provider, exporter = _provider(out)
    _emit_session(provider, "parent")

    pid = os.fork()
    if pid == 0:
        try:
            _emit_session(provider, "child")
            provider.force_flush()
            # The child must not merge: only the creating process owns the output file.
            exporter.shutdown()
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    assert not out.exists()
    provider.shutdown()

    data = json.loads(out.read_text())["data"]
    names = sorted(_tags(s)["session.id"] for t in data for s in t["spans"] if s["operationName"].startswith("session."))
    assert names == ["child", "parent"]


def test_merge_skips_truncated_lines(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    parts.mkdir()
    span = {"traceID": "a" * 32, "spanID": "b" * 16, "operationName": "llm.x", "startTime": 1, "duration": 1, "tags": []}
    (parts / "1.jsonl").write_text(json.dumps(span) + "\n" + '{"traceID": "trunc')
    out = tmp_path / "out.json"
    assert merge_parts(parts, out) == 1
    assert json.loads(out.read_text())["data"][0]["spans"] == [span]


def test_new_exporter_clears_stale_parts(tmp_path: Path) -> None:
    out = tmp_path / "traces.json"
    stale = Path(f"{out}.parts")
    stale.mkdir()
    (stale / "999.jsonl").write_text("{}\n")
    JaegerJsonFileExporter(str(out))
    assert list(stale.iterdir()) == []
