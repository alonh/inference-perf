"""Parsing: the single module that turns raw data into comparison-ready inputs.

Everything that touches a file, a raw request-response record, or a JSON string lives here.
Nothing here computes a metric — that is the job of response_similarity.py (per-turn) and
traces_similarity.py (per-trace), which take the outputs of this module as their inputs and
never parse anything themselves.

The layers, in the order data flows through them:

  file            load_records / load_run / find_run_dirs
  record          parse_response / parse_records        -> parsed response dict
  record          request_key / session_key / event_key -> grouping identity
  tool_calls      extract_tool_names / tool_signature / tool_args_signature / tool_kv_set
  parsed response response_signature                    -> exact-match unit
  records         extract_profile                       -> trace profile

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
    out["reasoning"] = (msg.get("reasoning_content") or "").strip()
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


def require_parsed(response: dict, what: str = "response") -> dict:
    """Assert `response` came from parse_response; raise a pointed error if it did not.

    The comparison layer used to silently re-parse anything without an `ok` key, which hid
    both the cost (re-parsing the same record on every pairwise comparison) and the bug
    class where a caller passed a raw record and got metrics computed over the wrong shape.
    Parsing now happens once, in the run script; this is the guard that keeps it that way.
    """
    if not isinstance(response, dict) or "ok" not in response:
        raise ValueError(
            f"{what} is not a parsed response (no 'ok' key); "
            "call compare.parse_response(record) first"
        )
    return response


# ---------------------------------------------------------- tool-call extraction

def _call_fields(tc: Any) -> Tuple[Optional[str], Any]:
    """Unpack one tool-call entry into (name, raw-arguments).

    Centralizes the defensive `tc["function"]["name"] / ["arguments"]` access repeated
    across every tool-call helper: a non-dict entry, a missing `function`, or a null
    `function` all yield (None, None) rather than raising.

    Accepts two shapes. The wire shape `{"function": {"name", "arguments"}}` is preferred;
    when no `function` block is present the flat shape `{"name", "arguments"}` is read
    instead, which is what the viewer's flattened tool summaries carry
    (export_viewer_data.version_from_record). A `function` block always wins, so a wire-shape
    entry can never be misread from stray top-level keys.
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
        name, _ = _call_fields(tc)
        if name:
            names.append(name)
    return tuple(names)


def tool_signature(tool_calls: List[dict]) -> Tuple:
    """Comparable signature of tool calls: sequence of (name, sorted arg keys).

    Compares tool names and argument structure, but not argument values.
    """
    sig = []
    for tc in tool_calls:
        name, args_raw = _call_fields(tc)
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            keys = tuple(sorted(args.keys())) if isinstance(args, dict) else ("<non-dict-args>",)
        except (json.JSONDecodeError, TypeError):
            keys = ("<unparseable-args>",)
        sig.append((name, keys))
    return tuple(sig)


def tool_args_signature(tool_calls: List[dict]) -> Tuple:
    """Full signature including canonical arg VALUES.

    Whitespace-insensitive: valid JSON is re-serialized canonically (so key order and
    inter-token spacing don't matter), and the unparseable fallback is whitespace-collapsed.
    """
    sig = []
    for tc in tool_calls:
        name, args_raw = _call_fields(tc)
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            canon = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            canon = strip_ws(str(args_raw))
        sig.append((name, canon))
    return tuple(sig)


def tool_kv_set(tool_calls: List[dict]) -> frozenset:
    """Flatten a turn's tool calls to a {(tool.key, canonical-value)} set (Yagubyan AC unit).

    Keys are namespaced by tool name so identical arg names on different tools don't
    collide; values are canonical JSON so key order / spacing can't fork them. This is the
    precomputed unit both compare_tool_arguments and argument_consistency compare.
    """
    kv: set = set()
    for tc in tool_calls:
        name, args_raw = _call_fields(tc)
        name = name or "<unnamed>"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        if isinstance(args, dict):
            for k, v in args.items():
                kv.add((f"{name}.{k}", json.dumps(v, sort_keys=True, ensure_ascii=False)))
    return frozenset(kv)


def response_signature(response: dict) -> Optional[str]:
    """Exact-match unit: collapse_ws(content) + canonical tool-args.

    None if the response errored. This is the single definition of "the same response",
    shared by every caller (analyze_consistency's modal counts, consistency_statistics'
    Feature, export_viewer_data's cluster coloring, find_metric_witnesses' exact_match, and
    compare_profiles' exact_match), so a tool turn with identical empty text but different
    args is correctly counted as non-identical everywhere.

    Takes an already-parsed response (or any dict carrying ok / content / tool_calls, such
    as the viewer's flattened version dicts) — raises on a raw record.
    """
    require_parsed(response)
    if not response.get("ok"):
        return None
    content = response.get("content", "")
    calls = response.get("tool_calls") or []
    return collapse_ws(content) + "\x00" + json.dumps(
        tool_args_signature(calls), ensure_ascii=False
    )


# ------------------------------------------------------- records -> trace profile

def extract_profile(records: List[dict]) -> dict:
    """Extract a trace profile from a list of request-response records.

    Args:
        records: List of dicts from per_request_lifecycle_metrics.json

    Returns:
        dict with keys:
          - tool_sequence: Tuple[str, ...] — ordered tool NAMES across all requests
          - tool_call_sequence: Tuple[Tuple[str, str], ...] — ordered (name, canonical-args)
            tokens across all requests; the arg-aware counterpart of tool_sequence, used by
            the ordered-dedup trace metric (two calls match only if name AND args agree)
          - unique_tools: set — set of tool names used
          - num_requests: int — how many request-response pairs
          - num_tool_calls: int — total tool invocations
          - num_errors: int — how many requests errored
          - responses: List[dict] — parsed responses (ok, content, tool_calls, etc.)
          - records: List[dict] — original records

    This is the parsing step for a whole run: the resulting profile is what
    traces_similarity.compare_profiles consumes, and its `responses` are already parsed, so
    the comparison layer never re-reads a raw record.
    """
    all_tool_names: List[str] = []
    all_tool_calls: List[Tuple[str, str]] = []
    responses = []
    num_errors = 0

    for rec in records:
        parsed = parse_response(rec)
        responses.append(parsed)

        if not parsed["ok"]:
            num_errors += 1
            continue

        # Extract tool names in order.
        all_tool_names.extend(extract_tool_names(parsed["tool_calls"]))
        # Arg-aware tokens in order: (name, canonical-args) per call, spanning the whole trace.
        all_tool_calls.extend(tool_args_signature(parsed["tool_calls"]))

    tool_sequence = tuple(all_tool_names)
    unique_tools = set(all_tool_names)

    return {
        "tool_sequence": tool_sequence,
        "tool_call_sequence": tuple(all_tool_calls),
        "unique_tools": unique_tools,
        "num_requests": len(records),
        "num_tool_calls": len(all_tool_names),
        "num_errors": num_errors,
        "responses": responses,
        "records": records,
    }
