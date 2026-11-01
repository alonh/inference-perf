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

"""
Span exporter that writes traces to a local file in Jaeger's JSON format.

The output is the same ``{"data": [trace, ...]}`` document that Jaeger's
``/api/traces`` endpoint and UI "Download JSON" produce, so a run can be analysed
without a Jaeger instance.

Spans are produced by the main process (session/stage spans) and by forked load
generator workers (LLM request spans). Each process appends its spans, one Jaeger
span per line, to its own part file under ``<path>.parts/``; the process that
created the exporter merges every part into ``<path>`` when it shuts down. If the
run dies before the merge the part files are left in place.
"""

import json
import logging
import os
import shutil
import threading
from collections import defaultdict
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

PROCESS_ID = "p1"


def _tag(key: str, value: Any) -> Dict[str, Any]:
    """One Jaeger key/type/value tag; sequences are stored as JSON strings, as Jaeger does."""
    if isinstance(value, bool):
        return {"key": key, "type": "bool", "value": value}
    if isinstance(value, int):
        return {"key": key, "type": "int64", "value": value}
    if isinstance(value, float):
        return {"key": key, "type": "float64", "value": value}
    if isinstance(value, str):
        return {"key": key, "type": "string", "value": value}
    return {"key": key, "type": "string", "value": json.dumps(list(value))}


def span_to_jaeger(span: ReadableSpan) -> Dict[str, Any]:
    """Convert a finished SDK span to a Jaeger JSON span (without the per-trace ``processes`` map)."""
    ctx = span.get_span_context()
    assert ctx is not None
    trace_id = format(ctx.trace_id, "032x")
    start_ns = span.start_time or 0
    end_ns = span.end_time or start_ns

    tags = [_tag(k, v) for k, v in (span.attributes or {}).items()]
    if span.instrumentation_scope is not None:
        tags.append(_tag("otel.scope.name", span.instrumentation_scope.name))
        if span.instrumentation_scope.version:
            tags.append(_tag("otel.scope.version", span.instrumentation_scope.version))
    tags.append(_tag("span.kind", span.kind.name.lower()))
    if span.status.status_code != StatusCode.UNSET:
        tags.append(_tag("otel.status_code", span.status.status_code.name))
        if span.status.description:
            tags.append(_tag("otel.status_description", span.status.description))
        if span.status.status_code == StatusCode.ERROR:
            tags.append(_tag("error", True))
    tags.sort(key=lambda t: str(t["key"]))

    references = []
    if span.parent is not None:
        references.append({"refType": "CHILD_OF", "traceID": trace_id, "spanID": format(span.parent.span_id, "016x")})
    for link in span.links:
        references.append(
            {
                "refType": "FOLLOWS_FROM",
                "traceID": format(link.context.trace_id, "032x"),
                "spanID": format(link.context.span_id, "016x"),
            }
        )

    logs = [
        {
            "timestamp": event.timestamp // 1000,
            "fields": [_tag("event", event.name)] + [_tag(k, v) for k, v in (event.attributes or {}).items()],
        }
        for event in span.events
    ]

    return {
        "traceID": trace_id,
        "spanID": format(ctx.span_id, "016x"),
        "operationName": span.name,
        "references": references,
        "startTime": start_ns // 1000,
        "duration": (end_ns - start_ns) // 1000,
        "tags": tags,
        "logs": logs,
        "processID": PROCESS_ID,
        "warnings": None,
    }


def _process_of(span: ReadableSpan) -> Dict[str, Any]:
    attributes = dict(span.resource.attributes) if span.resource is not None else {}
    service_name = str(attributes.pop("service.name", "unknown_service"))
    return {"serviceName": service_name, "tags": sorted((_tag(k, v) for k, v in attributes.items()), key=lambda t: t["key"])}


def merge_parts(parts_dir: Path, output_path: Path, process: Optional[Dict[str, Any]] = None) -> int:
    """Group every span line under ``parts_dir`` into Jaeger traces and write them to ``output_path``.

    Returns the number of traces written. Traces are ordered by their earliest span and
    spans within a trace by start time.
    """
    process = process or {"serviceName": "inference-perf", "tags": []}
    by_trace: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for part in sorted(parts_dir.glob("*.jsonl")):
        with open(part) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A process killed mid-write leaves a truncated last line.
                    logger.warning(f"Skipping malformed span line in {part}")
                    continue
                if "process" in record:
                    process = record["process"]
                    continue
                by_trace[record["traceID"]].append(record)

    traces: List[Dict[str, Any]] = []
    for trace_id, spans in by_trace.items():
        spans.sort(key=lambda s: (s["startTime"], s["spanID"]))
        traces.append({"traceID": trace_id, "spans": spans, "processes": {PROCESS_ID: process}, "warnings": None})
    traces.sort(key=lambda t: t["spans"][0]["startTime"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"data": traces}, f, ensure_ascii=False)
    os.replace(tmp_path, output_path)
    return len(traces)


class JaegerJsonFileExporter(SpanExporter):
    """Write spans to ``path`` as a Jaeger JSON trace export. Safe to share across forked processes."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.parts_dir = Path(f"{path}.parts")
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._file: Optional[IO[str]] = None
        self._file_pid: Optional[int] = None
        self._process_written = False
        self._shutdown = False
        if hasattr(os, "register_at_fork"):
            # The lock may be held by the parent's export thread at fork time.
            os.register_at_fork(after_in_child=self._reset_lock)
        # Leftovers from an earlier run would be merged into this one.
        shutil.rmtree(self.parts_dir, ignore_errors=True)
        self.parts_dir.mkdir(parents=True, exist_ok=True)

    def _reset_lock(self) -> None:
        self._lock = threading.Lock()

    def _part_file(self) -> IO[str]:
        # A forked worker inherits the parent's open handle; give each process its own part file.
        pid = os.getpid()
        if self._file is None or self._file_pid != pid:
            self._file = open(self.parts_dir / f"{pid}.jsonl", "a", encoding="utf-8")
            self._file_pid = pid
            self._process_written = False
        return self._file

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        try:
            with self._lock:
                f = self._part_file()
                if not self._process_written:
                    f.write(json.dumps({"process": _process_of(spans[0])}) + "\n")
                    self._process_written = True
                for span in spans:
                    f.write(json.dumps(span_to_jaeger(span), ensure_ascii=False) + "\n")
                f.flush()
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.warning(f"Failed to write spans to {self.parts_dir}: {e}")
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._lock:
            if self._file is not None and self._file_pid == os.getpid():
                self._file.flush()
        return True

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            if self._file is not None and self._file_pid == os.getpid():
                self._file.close()
            self._file = None
        # Only the creating process merges: by the time it shuts down the workers have exited.
        if os.getpid() != self._owner_pid:
            return
        try:
            count = merge_parts(self.parts_dir, self.path)
            shutil.rmtree(self.parts_dir, ignore_errors=True)
            logger.info(f"Wrote {count} OTEL traces (Jaeger JSON) to {self.path}")
        except Exception as e:
            logger.warning(f"Failed to merge OTEL span files in {self.parts_dir} into {self.path}: {e}")
