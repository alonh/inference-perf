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

  file            load_records / load_run
  record          parse_response / parse_records               -> parsed response dict
  record          request_key / session_key / event_key        -> grouping identity
  layout          find_sessions / load_session / iter_sessions -> Session (events x runs)
  driver          analyze                                      -> the two-stage reduction
  tool_calls      call_fields / extract_tool_names             -> names and raw arguments
  normalization   collapse_ws / strip_ws

Three words, and no more: a **session** consists of **events**; a **run** is one replay of a
whole session. There is deliberately no name for "one run of one session" — the directory
holding it is built from (base, session_id, run_id) inside `load_session` and never surfaces.

The layout read is `<base>/sessions/<session_id>/run_<i>/`, which is what `run_experiment.sh`
writes. The older run-major layout (`<base>/run_<i>/`, many sessions per directory) is NOT
supported: `find_run_dirs` was removed rather than made to straddle both, so pointing any
analysis at a run-major directory fails loudly instead of returning nothing.

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
import re
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

METRICS_FILE = "per_request_lifecycle_metrics.json"
SESSIONS_DIR = "sessions"


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
    path = os.path.join(run_dir, METRICS_FILE)
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


# ------------------------------------------------------------------------- layout

_TRACE_SLOT = re.compile(r"^trace\d+_")


class Session:
    """One session's replayed responses: the (event, run) table, plus both ordered axes.

    A session consists of EVENTS; a RUN is one replay of the whole session. The data for one
    session is therefore a table indexed by (event, run), and the two axes are the session's
    own structure rather than an artefact of how files were nested:

      events   every event key the session produced, in call order (the union across runs)
      runs     the run ids that produced at least one response, in numeric order

    The table is SPARSE on purpose. A run truncated part-way simply has no entry for the
    events it never reached, so ragged coverage stays visible: `events` is the union, and a
    hole is a missing (event, run) pair rather than a shorter axis. `missing()` lists them.

    Both orientations are transposes of ONE parse, so nothing is re-parsed to change view:
    `by_event()` gives "the same input, once per run" — what comparing repeated outputs needs —
    and `by_run()` gives "one run's path through the session" — what a per-run summary needs.
    """

    __slots__ = ("session_id", "events", "runs", "responses")

    def __init__(
        self,
        session_id: str,
        events: Sequence[str],
        runs: Sequence[str],
        responses: Dict[Tuple[str, str], Any],
    ) -> None:
        self.session_id = session_id
        self.events: Tuple[str, ...] = tuple(events)
        self.runs: Tuple[str, ...] = tuple(runs)
        self.responses = responses

    def by_event(self) -> Dict[str, Dict[str, Any]]:
        """event -> {run: payload}, both in axis order. Missing pairs are simply absent."""
        return {
            e: {r: self.responses[(e, r)] for r in self.runs if (e, r) in self.responses}
            for e in self.events
        }

    def by_run(self) -> Dict[str, Dict[str, Any]]:
        """run -> {event: payload}, both in axis order. Missing pairs are simply absent."""
        return {
            r: {e: self.responses[(e, r)] for e in self.events if (e, r) in self.responses}
            for r in self.runs
        }

    def missing(self) -> List[Tuple[str, str]]:
        """The (event, run) pairs with no recorded response — this session's ragged coverage."""
        return [(e, r) for e in self.events for r in self.runs
                if (e, r) not in self.responses]

    def __repr__(self) -> str:
        return (f"Session({self.session_id!r}, {len(self.events)} events x "
                f"{len(self.runs)} runs, {len(self.missing())} missing)")


def _run_index(run_dir: str) -> int:
    """Sort key for run_<i>: numeric, so run_10 does not land between run_1 and run_2."""
    suffix = os.path.basename(run_dir.rstrip("/")).partition("_")[2]
    return int(suffix) if suffix.isdigit() else 1 << 30


def _run_dirs(base: str, session_id: str) -> List[str]:
    """This session's run dirs that actually hold data, in run order.

    Requiring METRICS_FILE is the runner's own done-marker for a finished replay, so an
    interrupted experiment contributes the runs it completed instead of raising.
    """
    pattern = os.path.join(base, SESSIONS_DIR, session_id, "run_*")
    dirs = [d for d in glob.glob(pattern) if os.path.exists(os.path.join(d, METRICS_FILE))]
    return sorted(dirs, key=_run_index)


def check_session_matches(session_id: str, run_id: str, records: List[dict]) -> None:
    """Raise unless every record of this run belongs to `session_id`.

    A run directory under sessions/<id>/ is written by a process pinned to that one session
    (`num_sessions: 1` plus the single-session replay filter). More than one recorded session
    means the pin did not take; a different one means the filter named the wrong session.
    Either way the events would be attributed to a session that never produced them, which no
    downstream metric can detect — so this is a hard error rather than a warning.

    The recorded id carries the replay slot of the process that wrote it (`trace0_<dataset
    id>` for a session-major cell, `trace<k>_` in a corpus migrated from the run-major
    layout), so the slot is stripped before comparing: the directory name is the bare dataset
    id in both cases.
    """
    recorded = {_TRACE_SLOT.sub("", session_key(rec)) for rec in records}
    if recorded != {session_id}:
        raise ValueError(
            f"{run_id} of session {session_id!r} holds records for {sorted(recorded)!r} — "
            "expected only the session it is filed under"
        )


def find_sessions(base: str) -> List[str]:
    """The analyzable session ids under `base`, sorted. Empty if there are none.

    Session identity is the DIRECTORY name, which is the bare dataset session id. That is
    independent of the replay slot the records happen to carry, and it is known even for a
    session whose every response errored.

    "Analyzable" means at least one run wrote METRICS_FILE, so a partially-finished
    experiment yields the sessions it got to rather than failing.
    """
    root = os.path.join(base, SESSIONS_DIR)
    names = sorted(os.path.basename(d.rstrip("/"))
                   for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    return [name for name in names if _run_dirs(base, name)]


def load_session(base: str, session_id: str, transform: Optional[Callable[[dict], Any]] = None) -> Session:
    """Load one session — every run of it — as a (event, run) table.

    This is the whole filesystem entry point: one session at a time, so a notebook can work on
    a single session without reading the other nine.

    `transform` is applied ONCE per record — pass `parse_response`, or a metric layer's own
    per-record type — and both orientations of the table then re-key objects that already
    exist. It defaults to the raw record.

    Event order is `event_key` order. The recorded event_id embeds a zero-padded call index
    (`<session>:event_007_<call>`), so sorting the keys puts the session's events in call
    order; the request-hash fallback is not call-ordered but is stable across runs, which is
    what the axis has to guarantee. Within one run a repeated event key keeps the FIRST
    record: a run should meet each identical input once, and first-wins is order-stable.
    """
    run_dirs = _run_dirs(base, session_id)
    if not run_dirs:
        raise FileNotFoundError(
            f"no completed run under {os.path.join(base, SESSIONS_DIR, session_id)} "
            f"(a run dir holding {METRICS_FILE})"
        )

    responses: Dict[Tuple[str, str], Any] = {}
    runs: List[str] = []
    events: Set[str] = set()
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir.rstrip("/"))
        records = load_run(run_dir)
        if not records:
            continue  # marker present but unreadable/empty; load_run already tolerated it
        check_session_matches(session_id, run_id, records)
        runs.append(run_id)
        for rec in records:
            key = event_key(rec)
            if (key, run_id) in responses:
                continue
            responses[(key, run_id)] = transform(rec) if transform else rec
            events.add(key)

    return Session(session_id, sorted(events), runs, responses)


def iter_sessions(base: str, transform: Optional[Callable[[dict], Any]] = None) -> Iterator[Session]:
    """Every analyzable session under `base`, one at a time.

    A generator, so a reduction over sessions never holds more than one session's parse.
    Raises if `base` holds no analyzable session at all — an empty-but-successful analysis is
    the one outcome worse than a failed one.
    """
    session_ids = find_sessions(base)
    if not session_ids:
        raise FileNotFoundError(
            f"no analyzable session under {base!r}: expected "
            f"{os.path.join(base, SESSIONS_DIR, '<session_id>', 'run_<i>', METRICS_FILE)}"
        )
    for session_id in session_ids:
        yield load_session(base, session_id, transform)


class Analysis(NamedTuple):
    """What `analyze` returns: the cross-session conclusion, and the per-session ones."""

    combined: Any
    per_session: List[Any]


def analyze(
    base: str,
    transform: Optional[Callable[[dict], Any]],
    session_conclusion: Callable[[Session], Any],
    combine: Callable[[List[Any]], Any],
) -> Analysis:
    """The two-stage reduction every analysis in this experiment follows.

      stage 1   session_conclusion(Session) -> S    combines the RUNS of one session
      stage 2   combine([S, ...]) -> C              combines SESSIONS

    Event-level reduction belongs inside stage 1, because events belong to a session. A
    session is the unit of independence here — the U-statistic instance, the unit the CI's
    degrees of freedom count — so `combine` is the only place that may see more than one.

    `session_conclusion` takes a Session and nothing else: not `base`, not its neighbours. A
    per-session conclusion therefore cannot depend on which other sessions happened to be on
    disk, which is both what makes the statistics' independence claim true and what would let
    per-session results be cached later. Anything a stage needs that is NOT corpus data (a
    judge endpoint, an alpha) belongs in a closure over it.
    """
    per_session = [session_conclusion(s) for s in iter_sessions(base, transform)]
    return Analysis(combine(per_session), per_session)


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
