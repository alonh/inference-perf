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

"""Replay-report parsing: raw files and records in, plain data out.

Everything that touches a file, a raw request-response record, or a JSON string lives here.
This module is deliberately **general** — it knows the shape of a replay report and nothing
about consistency, similarity or any metric. It imports only the standard library, and in
particular imports nothing from `compare/`, so a notebook or a DataFrame script that only
wants to read a run can use it without pulling in the comparison library.

  file            find_run_dirs / load_records / load_run
  record          parse_response / parse_records        -> parsed response dict
  record          request_key / session_key / event_key -> grouping identity
  tool_calls      call_fields / extract_tool_names      -> names and raw arguments
  normalization   collapse_ws / strip_ws

What is deliberately NOT here: the canonical forms that exist only to be compared — the
tool/response signatures and the kv-set (`compare/signatures.py`) — and the trace profile that
`compare_profiles` consumes (`compare/traces_similarity.py`). Those are metric definitions:
changing them changes what a caller counts as "the same response". Splitting a record into
fields is not. `require_parsed`, the guard asserting a dict came from `parse_response`, lives
in `compare/signatures.py` too: it describes this module's output, but only the comparators
ever call it.

Callers should parse ONCE, up front, and pass the results around: parsing a record is
expensive relative to a comparison, and a pairwise analysis touches each record many times.
"""

import glob
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------ normalization

def collapse_ws(s: str) -> str:
    """Collapse every run of whitespace to a single space and trim ends.

    For free-form response *content* (prose): normalizes indentation / trailing spaces /
    line-wrap differences while preserving the single spaces between words, so genuinely
    different text stays different.
    """
    return " ".join((s or "").split())


def strip_ws(s: str) -> str:
    """Remove ALL whitespace.

    For *tool arguments* (JSON / structural), where inter-token spacing is pure formatting:
    makes `{ "content":` and `{"content":` compare equal. Valid JSON already round-trips
    whitespace-free through json.loads/dumps; this handles the unparseable-args fallback.
    """
    return "".join((s or "").split())


# ----------------------------------------------------------------------------- IO

def find_run_dirs(base: str) -> List[str]:
    """Every run_* subdirectory of `base`, sorted. One dir = one full replay."""
    dirs = sorted(glob.glob(os.path.join(base, "run_*")))
    return [d for d in dirs if os.path.isdir(d)]


def load_records(filepath: str) -> List[dict]:
    """Load the record list from one per_request_lifecycle_metrics.json.

    The report may wrap the list; accept either a bare list or {"contents": [...]} /
    {"records": [...]}. Raises if the file is missing or holds none of those shapes —
    use load_run() for the tolerant, returns-[] variant.
    """
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = data.get("contents", data.get("records"))
        if isinstance(records, list):
            return records
    raise ValueError(f"Unexpected data format in {filepath}")


def load_run(run_dir: str) -> List[dict]:
    """Load per_request_lifecycle_metrics.json from a run dir (list of records).

    Tolerant counterpart of load_records: a missing file or an unrecognized shape yields
    [] rather than raising, so a sweep over many run dirs skips incomplete runs.
    """
    path = os.path.join(run_dir, "per_request_lifecycle_metrics.json")
    if not os.path.exists(path):
        return []
    try:
        return load_records(path)
    except ValueError:
        return []


# ---------------------------------------------------------------- record identity

def request_key(record: dict) -> str:
    """Stable hash of the request payload. Identical inputs -> identical key.

    Used as a fallback grouping key when a record lacks an event_id, and to validate
    that every response in an event_id group really was fed identical input.
    """
    req = record.get("request")
    if req is None:
        return "no-request"
    try:
        obj = json.loads(req) if isinstance(req, str) else req
        # messages + tools + generation params define the input; drop nothing —
        # canonicalize with sorted keys so formatting can't fork the group.
        canon = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        canon = req if isinstance(req, str) else str(req)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def session_key(record: dict) -> str:
    """Per-trace identifier: the recorded session_id.

    Each source trace maps to one session_id; every call position within that trace
    shares it. Falls back to a request-payload-derived anchor only if session_id is
    absent (older metrics that never populated it).
    """
    sid = record.get("session_id") or (record.get("info") or {}).get("session_id")
    if sid:
        return str(sid)
    req = record.get("request")
    try:
        obj = json.loads(req) if isinstance(req, str) else req
        msgs = obj.get("messages") or []
        # First message content is the per-trace anchor (system/context prompt).
        anchor = (msgs[0].get("content") or "") if msgs else ""
        anchor = anchor if isinstance(anchor, str) else json.dumps(anchor, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, AttributeError):
        anchor = str(req)[:512]
    return "trace_" + hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:10]


def event_key(record: dict) -> str:
    """Per-call-position identifier used to group identical-input responses.

    Prefer the recorded event_id — it is unique within a run and byte-identical across
    runs for the same (trace, call-position), so it groups exactly the repeated outputs
    of one identical input. Falls back to (session, request-hash) if event_id is absent.
    """
    eid = record.get("event_id") or (record.get("info") or {}).get("event_id")
    if eid:
        return str(eid)
    return session_key(record) + "::" + request_key(record)[:12]


# ------------------------------------------------------- record -> parsed response

def parse_response(record: dict) -> Dict[str, Any]:
    """Extract the fields we compare from a per-request record.

    Returns dict with: ok(bool), has_output(bool), content(str), reasoning(str),
    tool_calls(list), finish_reason(str|None), completion_tokens(int|None),
    prompt_tokens(int|None), total_tokens(int|None), start_time(float|None),
    end_time(float|None), error(str|None).

    `ok` means the response parsed into a well-formed choice/message — NOT that the model
    produced anything usable. `has_output` is the stricter flag: True only when the response
    is ok AND carries actual content or tool calls. An ok response can have has_output=False
    — e.g. a turn that spends its whole token budget on reasoning, hits finish_reason=length,
    and emits an empty message. Such empty outputs would otherwise compare as vacuously
    identical (content_levenshtein("","")=1.0), so downstream metrics use has_output to treat
    them as their own category rather than as perfect agreement.

    Timing (start_time / end_time) is read from the top-level record and is populated even
    for errored responses — a request can fail yet still have taken time. These are the raw
    wall-clock bounds; any duration is derived downstream in the metric layer. Token counts
    come from the response body's `usage` block.

    The presence of the `ok` key is what marks a dict as parsed: the comparison layer
    checks for it and refuses raw records (see require_parsed).
    """
    out: Dict[str, Any] = {
        "ok": False,
        "has_output": False,
        "content": "",
        "reasoning": "",
        "tool_calls": [],
        "finish_reason": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "total_tokens": None,
        "start_time": None,
        "end_time": None,
        "error": None,
    }

    # Timing lives at the record level (not in the response body), so capture it before any
    # error short-circuit — an errored/empty response can still carry meaningful bounds.
    out["start_time"], out["end_time"] = record.get("start_time"), record.get("end_time")

    if record.get("error"):
        err = record["error"]
        out["error"] = err.get("error_type") if isinstance(err, dict) else str(err)
        return out

    resp = record.get("response")
    if not resp:
        out["error"] = "empty_response"
        return out
    try:
        body = json.loads(resp) if isinstance(resp, str) else resp
    except (json.JSONDecodeError, TypeError):
        # Non-JSON body (e.g. gateway HTML error). Treat as error but keep text.
        out["error"] = "non_json_response"
        out["content"] = resp if isinstance(resp, str) else str(resp)
        return out

    choices = body.get("choices") or []
    if not choices:
        out["error"] = "no_choices"
        return out
    msg = choices[0].get("message", {}) or {}
    out["content"] = (msg.get("content") or "").strip()
    # Both spellings, `reasoning` first — the same order the harness uses everywhere it reads
    # a chain-of-thought off the wire (openai_client.py, replay_graph_session_datagen.py).
    # vLLM/RITS return `reasoning`; `reasoning_content` is the OpenAI-style alias the harness
    # writes back out when it *re*-serializes a message (apis/chat.py sets both). Reading only
    # the alias silently yielded "" for every recorded response.
    out["reasoning"] = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    out["tool_calls"] = msg.get("tool_calls") or []
    out["finish_reason"] = choices[0].get("finish_reason")
    usage = body.get("usage") or {}
    out["completion_tokens"] = usage.get("completion_tokens")
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["total_tokens"] = usage.get("total_tokens")
    out["ok"] = True
    # Usable AND non-empty: distinguishes a real answer from an empty completion (e.g. a
    # length-truncated reasoning turn that emitted no content or tool calls).
    out["has_output"] = bool(out["content"] or out["tool_calls"])
    return out


def parse_records(records: List[dict]) -> List[Dict[str, Any]]:
    """Parse a whole run's records once, in order. The comparison layer's expected input."""
    return [parse_response(rec) for rec in records]


# ---------------------------------------------------------- tool-call extraction

def call_fields(tc: Any) -> Tuple[Optional[str], Any]:
    """Unpack one tool-call entry into (name, raw-arguments).

    Centralizes the defensive `tc["function"]["name"] / ["arguments"]` access repeated
    across every tool-call helper: a non-dict entry, a missing `function`, or a null
    `function` all yield (None, None) rather than raising.

    Accepts two shapes. The wire shape `{"function": {"name", "arguments"}}` is preferred;
    when no `function` block is present the flat shape `{"name", "arguments"}` is read
    instead, which is what the viewer's flattened tool summaries carry
    (export_viewer_data.version_from_record). A `function` block always wins, so a wire-shape
    entry can never be misread from stray top-level keys.

    Public (rather than the `_call_fields` it used to be) because the signature builders in
    compare/signatures.py read it too, and importing a private name across a module boundary
    would be worse than naming it.
    """
    if not isinstance(tc, dict):
        return None, None
    fn = tc.get("function")
    if fn is None and ("name" in tc or "arguments" in tc):
        fn = tc  # flat shape
    fn = fn or {}
    if not isinstance(fn, dict):
        return None, None
    return fn.get("name"), fn.get("arguments")


def extract_tool_names(tool_calls: List[dict]) -> Tuple[str, ...]:
    """Extract ordered tool names from tool_calls list."""
    names = []
    for tc in tool_calls:
        name, _ = call_fields(tc)
        if name:
            names.append(name)
    return tuple(names)
