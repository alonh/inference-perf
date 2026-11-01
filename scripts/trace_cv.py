# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coefficient of variation (CV = stdev / mean) of run metrics from a trace file.

Two input formats are accepted, detected from the content:

* Jaeger JSON export (``{"data": [trace, ...]}``): each trace is one replayed session, a
  ``session.*`` root span with ``llm.*`` child spans.
* Wire-trace JSONL (one captured HTTP call per line, with ``response``/``chunks``/``total_ms``):
  calls are grouped into sessions by their ``x-litellm-session-id`` header. TTFT is recovered
  from ``chunks`` (``[ms since request start, bytes read]`` per network read) as the time of the
  read that delivered the first SSE event carrying a generated token -- reasoning, content or a
  tool call -- so role-only first deltas don't count.

The question answered is inter-replay consistency: how much the same task varies from one replay
(session) to the next. Each metric is computed once per session and its CV is taken across
sessions; the per-call-position table does the same for the Nth call of every session, which shows
which step of the task is the noisy one. Comparing against the original recording is out of scope.

With live substitution and sampling, replays legitimately write different amounts of output, so
the metrics are split in two groups:

* length-insensitive (expected to be consistent): TPOT, TTFT, output tokens per second of request
  busy time, and goodput when the SLOs are on TTFT/TPOT;
* length-sensitive (expected to vary with what the model chose to write): end-to-end latency,
  wall-clock throughput, request goodput and latency-SLO goodput.

TPOT comes from the TPOT attribute, else from ``(latency - TTFT) / (output_tokens - 1)``. When a
file has no TTFT at all, the per-request prefill+overhead time is estimated as the intercept of a
least-squares fit ``latency = a + b * output_tokens`` over every request, and TPOT is marked
estimated; per-call-position TPOT then skips calls with fewer than 200 output tokens.

Two structural metrics compare the replays' agent behaviour rather than server timing; both are
means over all pairs of replays, so 100% means every replay did the same thing:

* Workflow consistency: ``mean(1 - Lev(S_i, S_j) / max(|S_i|, |S_j|))`` where ``S_i`` is replay
  i's call-type sequence ``[user, assistant, tool, assistant, ...]``, built the same way as the
  workflow fidelity in ``trace_comparison.py``: every successful call is an ``assistant`` followed
  by one ``tool`` per tool it called, preceded by a ``user`` when it opens its conversation or the
  previous call of that conversation called no tools. Conversations (identified by their system
  prompt) are concatenated in order of first appearance, each as one block, so a side-call such as
  a title generator that ran at a different point doesn't count as an edit, while a conversation
  whose key changed mid-session (Claude Code's billing-header hash changes at compaction) stays in
  time order. It is reported twice: with call types only and with
  each ``assistant`` labelled by the tools it called.
* Dependency consistency: ``mean(|E_i & E_j| / |E_i | E_j|)`` where ``E_i`` is replay i's set of
  data-flow edges ``(m, n)``, calls identified by position: call n consumed call m's output, i.e.
  one of m's tool-call ids (else its text content, else its reasoning text) appears in n's
  input. A call whose output wasn't captured can't be traced; its outgoing edges are left out of
  every pair comparison involving that replay rather than counted as missing.

Goodput follows inference-perf's definition: a request is "good" when it succeeded and meets
every SLO given on the command line. With no SLOs, every successful request is good.

"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

TTFT_KEY = "gen_ai.response.time_to_first_token"
LATENCY_KEY = "gen_ai.response.total_latency"
TPOT_KEY = "gen_ai.response.time_per_output_token"
OUTPUT_TOKENS_KEY = "gen_ai.usage.completion_tokens"
PROMPT_TOKENS_KEY = "gen_ai.usage.prompt_tokens"
WIRE_SESSION_HEADER = "x-litellm-session-id"


@dataclass
class Request:
    start_us: int
    end_us: int
    ok: bool
    latency: float
    ttft: float | None
    tpot: float | None
    output_tokens: int
    prompt_tokens: int
    # Raw JSON of the input messages (searched for upstream outputs) and the assistant output message.
    input_raw: str = "[]"
    output_message: dict[str, Any] | None = None
    # Which conversation (main agent, title generator, subagent, ...) the call belongs to, from its system prompt.
    conversation: str = ""


@dataclass
class Session:
    name: str
    requests: list[Request] = field(default_factory=list)
    start_us: int | None = None
    end_us: int | None = None

    @property
    def duration_s(self) -> float:
        # Prefer the session span's own wall time; fall back to the first-start/last-end of its calls.
        start = self.start_us if self.start_us is not None else min(r.start_us for r in self.requests)
        end = self.end_us if self.end_us is not None else max(r.end_us for r in self.requests)
        return (end - start) / 1e6


def tags_of(span: dict[str, Any]) -> dict[str, Any]:
    return {t["key"]: t["value"] for t in span.get("tags", [])}


def load_sessions(path: str, llm_prefix: str) -> list[Session]:
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # More than one JSON value: a JSONL file.
        return load_wire_sessions([json.loads(line) for line in text.splitlines() if line.strip()])
    if isinstance(data, dict) and "data" not in data:
        return load_wire_sessions([data])  # a single-line JSONL file parses as one object
    return load_jaeger_sessions(data["data"] if isinstance(data, dict) else data, llm_prefix)


# Same conversation key as trace_comparison.py: the first KEY_CHARS characters of the whitespace-normalised system prompt.
KEY_CHARS = 200


def message_text(content: Any) -> str:
    """Text of a message content field, which may be a string or a list of text/content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text") or b.get("content") or "" for b in content or [] if isinstance(b, dict))


def conversation_key(first_text: str) -> str:
    return " ".join(first_text.split())[:KEY_CHARS]


def first_message_key(messages: Any) -> str:
    first = (messages or [{}])[0] if isinstance(messages, list) else {}
    return conversation_key(message_text(first.get("content") or first.get("parts"))) if isinstance(first, dict) else ""


def jaeger_output_message(tags: dict[str, Any]) -> dict[str, Any] | None:
    # gen_ai.output.message is only written when the output has tool calls; otherwise rebuild the
    # message from the text and reasoning attributes.
    if "gen_ai.output.message" in tags:
        try:
            msg = json.loads(tags["gen_ai.output.message"])
            return msg if isinstance(msg, dict) else None
        except json.JSONDecodeError:
            return None
    text = tags.get("gen_ai.output.text")
    reasoning = tags.get("gen_ai.output.reasoning")
    if not text and not reasoning:
        return None
    return {"role": "assistant", "content": text or "", "reasoning_content": reasoning or ""}


def load_jaeger_sessions(traces: list[dict[str, Any]], llm_prefix: str) -> list[Session]:
    sessions: list[Session] = []
    names: set[str] = set()
    for trace in traces:
        session = Session(name=trace["traceID"])
        for span in trace["spans"]:
            op = span["operationName"]
            tags = tags_of(span)
            if op.startswith("session."):
                session.name = str(tags.get("session.id", op.removeprefix("session.")))
                session.start_us = span["startTime"]
                session.end_us = span["startTime"] + span["duration"]
            elif op.startswith(llm_prefix):
                ttft = tags.get(TTFT_KEY)
                tpot = tags.get(TPOT_KEY)
                session.requests.append(
                    Request(
                        start_us=span["startTime"],
                        end_us=span["startTime"] + span["duration"],
                        ok=tags.get("otel.status_code", "OK") != "ERROR" and "error.message" not in tags,
                        latency=float(tags.get(LATENCY_KEY, span["duration"] / 1e6)),
                        ttft=float(ttft) if ttft is not None else None,
                        tpot=float(tpot) if tpot is not None else None,
                        output_tokens=int(tags.get(OUTPUT_TOKENS_KEY, 0)),
                        prompt_tokens=int(tags.get(PROMPT_TOKENS_KEY, 0)),
                        input_raw=str(tags.get("gen_ai.input.messages", "[]")),
                        output_message=jaeger_output_message(tags),
                        conversation=first_message_key(json.loads(str(tags.get("gen_ai.input.messages") or "[]"))),
                    )
                )
        if session.requests:
            session.requests.sort(key=lambda r: r.start_us)
            # session.id can repeat across traces (e.g. several replay exports merged into one file) and
            # sessions are keyed by name downstream, so disambiguate repeats rather than silently merging them.
            if session.name in names:
                session.name = f"{session.name}@{trace['traceID'][:8]}"
            names.add(session.name)
            sessions.append(session)
    return sessions


def _sse_events(body: str) -> list[tuple[int, dict[str, Any]]]:
    """Parse an SSE body into ``(end byte offset, data)`` pairs, skipping ``[DONE]`` and unparseable events."""
    events = []
    offset = 0
    for raw in body.split("\n\n"):
        offset += len(raw.encode()) + 2
        payload = "".join(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append((offset - 2, json.loads(payload)))
        except json.JSONDecodeError:
            continue
    return events


def _has_generated_token(data: dict[str, Any]) -> bool:
    choices = data.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    return bool(delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or delta.get("tool_calls"))


def wire_ttft_and_usage(record: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """Return ``(ttft seconds, usage)`` for one captured call; TTFT is None for a non-streamed response."""
    body = record.get("response") or ""
    try:
        return None, json.loads(body).get("usage") or {}
    except json.JSONDecodeError:
        pass
    events = _sse_events(body)
    usage: dict[str, Any] = {}
    for _, data in events:
        usage.update(data.get("usage") or {})
    first = next((end for end, data in events if _has_generated_token(data)), None)
    if first is None:
        return None, usage
    received = 0
    for ms, nbytes in record.get("chunks") or []:
        received += nbytes
        if received >= first:
            return float(ms) / 1000, usage
    return None, usage


def wire_output_message(record: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the assistant message (content and tool calls) from a captured response body."""
    body = record.get("response") or ""
    try:
        choices = json.loads(body).get("choices") or []
        unary = choices[0].get("message") if choices else None
        return unary if isinstance(unary, dict) else None
    except json.JSONDecodeError:
        pass
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    for _, data in _sse_events(body):
        delta = ((data.get("choices") or [{}])[0]).get("delta") or {}
        if delta.get("content"):
            content.append(str(delta["content"]))
        if delta.get("reasoning") or delta.get("reasoning_content"):
            reasoning.append(str(delta.get("reasoning") or delta.get("reasoning_content")))
        for chunk in delta.get("tool_calls") or []:
            call = tool_calls.setdefault(chunk.get("index", 0), {"id": "", "function": {"name": "", "arguments": ""}})
            if chunk.get("id"):
                call["id"] = chunk["id"]
            fn = chunk.get("function") or {}
            call["function"]["name"] += fn.get("name") or ""
            call["function"]["arguments"] += fn.get("arguments") or ""
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return msg


def wire_input_raw(record: dict[str, Any]) -> str:
    try:
        return json.dumps(json.loads(record.get("request") or "{}").get("messages") or [])
    except json.JSONDecodeError:
        return "[]"


def wire_conversation(record: dict[str, Any]) -> str:
    try:
        req = json.loads(record.get("request") or "{}")
    except json.JSONDecodeError:
        return ""
    # Anthropic requests carry the system prompt outside the messages.
    if req.get("system"):
        return conversation_key(message_text(req["system"]))
    return first_message_key(req.get("messages"))


def load_wire_sessions(records: list[dict[str, Any]]) -> list[Session]:
    by_session: dict[str, Session] = {}
    for record in records:
        headers = {k.lower(): v for k, v in (record.get("headers") or {}).items()}
        name = str(headers.get(WIRE_SESSION_HEADER, "session"))
        session = by_session.setdefault(name, Session(name=name))
        ttft, usage = wire_ttft_and_usage(record)
        latency = float(record.get("total_ms") or 0) / 1000
        output_tokens = int(usage.get("completion_tokens") or 0)
        start_us = int(float(record["start_unix"]) * 1e6)
        session.requests.append(
            Request(
                start_us=start_us,
                end_us=start_us + int(latency * 1e6),
                ok=200 <= int(record.get("status", 200)) < 300,
                latency=latency,
                ttft=ttft,
                # Same definition as the report: decode time spread over the tokens after the first.
                tpot=(latency - ttft) / (output_tokens - 1) if ttft is not None and output_tokens > 1 else None,
                output_tokens=output_tokens,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                input_raw=wire_input_raw(record),
                output_message=wire_output_message(record),
                conversation=wire_conversation(record),
            )
        )
    for session in by_session.values():
        session.requests.sort(key=lambda r: r.start_us)
    return list(by_session.values())


def called_tools(r: Request) -> list[str]:
    return [str((c.get("function") or {}).get("name", "")) for c in (r.output_message or {}).get("tool_calls") or []]


def workflow_sequence(s: Session, with_tools: bool) -> list[str]:
    """Call-type sequence S = [user, assistant, tool, assistant, ...] of a replay; see call_sequence in trace_comparison.py.

    Failed calls are left out: they generated nothing, and a retried call would otherwise count as an extra step.
    """
    by_key: dict[str, list[Request]] = {}
    for r in s.requests:
        if r.ok:
            by_key.setdefault(r.conversation, []).append(r)
    seq = []
    for key in by_key:  # first-appearance order
        prev_tools: list[str] | None = None
        for r in by_key[key]:
            tools = called_tools(r)
            if not prev_tools:
                seq.append("user")
            seq.append(f"assistant({','.join(tools)})" if with_tools and tools else "assistant")
            seq += ["tool"] * len(tools)
            prev_tools = tools
    return seq


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# A text-only output shorter than this is too generic to identify in a later prompt.
MIN_CONTENT_CHARS_FOR_EDGE = 20


def output_fingerprints(r: Request) -> list[str]:
    """Strings that identify r's output when it is fed into a later call's input."""
    msg = r.output_message or {}
    ids = [str(c["id"]) for c in msg.get("tool_calls") or [] if c.get("id")]
    if ids:
        return ids
    for key in ("content", "reasoning_content", "reasoning"):
        text = msg.get(key)
        if isinstance(text, str) and len(text) >= MIN_CONTENT_CHARS_FOR_EDGE:
            return [json.dumps(text)[1:-1]]  # as it appears escaped inside the input JSON
    return []


def dependency_edges(s: Session) -> tuple[set[tuple[int, int]], set[int]]:
    """Return ``(edges, untraceable)``: the data-flow edges and the call positions whose output can't be traced."""
    edges = set()
    untraceable = set()
    for m, upstream in enumerate(s.requests):
        prints = output_fingerprints(upstream)
        if not prints:
            untraceable.add(m)
            continue
        for n in range(m + 1, len(s.requests)):
            if any(fp in s.requests[n].input_raw for fp in prints):
                edges.add((m, n))
    return edges, untraceable


def jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def pairwise_mean(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"pairs": 0, "mean": None, "min": None}
    return {"pairs": len(values), "mean": statistics.fmean(values), "min": min(values)}


def structural_consistency(sessions: list[Session]) -> dict[str, dict[str, Any]]:
    pairs = [(i, j) for i in range(len(sessions)) for j in range(i + 1, len(sessions))]
    result = {}
    for name, with_tools in (("workflow_consistency_roles", False), ("workflow_consistency_tools", True)):
        seqs = [workflow_sequence(s, with_tools) for s in sessions]
        result[name] = pairwise_mean(
            [1 - levenshtein(seqs[i], seqs[j]) / max(len(seqs[i]), len(seqs[j]), 1) for i, j in pairs]
        )
    traced = [dependency_edges(s) for s in sessions]
    scores = []
    for i, j in pairs:
        blind = traced[i][1] | traced[j][1]
        scores.append(jaccard({e for e in traced[i][0] if e[0] not in blind}, {e for e in traced[j][0] if e[0] not in blind}))
    result["dependency_consistency"] = pairwise_mean(scores)
    result["dependency_consistency"]["edges_per_replay"] = [len(e) for e, _ in traced]
    result["dependency_consistency"]["untraceable_calls_per_replay"] = [sorted(u) for _, u in traced]
    return result


def is_good(r: Request, args: argparse.Namespace) -> bool:
    if not r.ok:
        return False
    if args.slo_latency is not None and r.latency > args.slo_latency:
        return False
    # A TTFT/TPOT SLO cannot be checked without the attribute; count the request as failing it.
    if args.slo_ttft is not None and (r.ttft is None or r.ttft > args.slo_ttft):
        return False
    if args.slo_tpot is not None and (r.tpot is None or r.tpot > args.slo_tpot):
        return False
    return True


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def cv(values: list[float | None]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return {"n": len(vals), "mean": vals[0] if vals else None, "stdev": None, "cv": None}
    mean = statistics.fmean(vals)
    stdev = statistics.stdev(vals)  # sample stdev: the sessions are a sample of repeated replays
    return {"n": len(vals), "mean": mean, "stdev": stdev, "cv": stdev / mean if mean else None}


def fit_prefill_estimate(requests: list[Request]) -> float | None:
    """Intercept ``a`` of the least-squares fit ``latency = a + b * output_tokens``; None if it can't be fitted."""
    xs = [float(r.output_tokens) for r in requests]
    ys = [r.latency for r in requests]
    if len(set(xs)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
    return max(0.0, my - slope * mx)


def decode_time(r: Request, prefill_est: float | None) -> float | None:
    """Time spent emitting tokens after the first one."""
    if r.tpot is not None:
        return r.tpot * (r.output_tokens - 1)
    if r.ttft is not None:
        return r.latency - r.ttft
    if prefill_est is not None:
        return max(0.0, r.latency - prefill_est)
    return None


# Below this many output tokens an estimated decode time is dominated by the error of the fitted prefill.
MIN_TOKENS_FOR_ESTIMATED_TPOT = 200


def request_tpot(r: Request, prefill_est: float | None) -> float | None:
    if r.tpot is None and r.ttft is None and r.output_tokens < MIN_TOKENS_FOR_ESTIMATED_TPOT:
        return None
    dt = decode_time(r, prefill_est)
    return dt / (r.output_tokens - 1) if dt is not None and r.output_tokens > 1 else None


def session_tpot(requests: list[Request], prefill_est: float | None) -> float | None:
    # Token-weighted, so a 20-token call doesn't count as much as a 2,000-token one.
    pairs = [(decode_time(r, prefill_est), r.output_tokens - 1) for r in requests if r.output_tokens > 1]
    known = [(dt, n) for dt, n in pairs if dt is not None]
    tokens = sum(n for _, n in known)
    return sum(dt for dt, _ in known) / tokens if tokens else None


STABLE_METRICS = ("tpot", "ttft_mean", "busy_output_throughput", "goodput_percentage")


def session_metrics(s: Session, args: argparse.Namespace, prefill_est: float | None) -> dict[str, float | None]:
    dur = s.duration_s
    ok = [r for r in s.requests if r.ok]
    good = [r for r in s.requests if is_good(r, args)]
    busy = sum(r.latency for r in ok)
    return {
        # length-insensitive
        "tpot": session_tpot(ok, prefill_est),
        "ttft_mean": mean_or_none([r.ttft for r in ok if r.ttft is not None]),
        "busy_output_throughput": sum(r.output_tokens for r in ok) / busy if busy > 0 else None,
        "goodput_percentage": 100 * len(good) / len(s.requests),
        # length-sensitive
        "latency_mean": mean_or_none([r.latency for r in ok]),
        "output_tokens_total": float(sum(r.output_tokens for r in ok)),
        "output_token_throughput": sum(r.output_tokens for r in ok) / dur if dur > 0 else None,
        "request_throughput": len(ok) / dur if dur > 0 else None,
        "request_goodput": len(good) / dur if dur > 0 else None,
        "token_goodput": sum(r.prompt_tokens + r.output_tokens for r in good) / dur if dur > 0 else None,
    }


def fmt(v: float | None, digits: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def pct(r: dict[str, Any]) -> str:
    return "n/a" if r["cv"] is None else f"{100 * r['cv']:.1f}%"


def print_cv_table(title: str, rows: dict[str, dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(f"  {'metric':<26}{'n':>4}{'mean':>14}{'stdev':>14}{'CV':>10}")
    for name, r in rows.items():
        print(f"  {name:<26}{r['n']:>4}{fmt(r['mean']):>14}{fmt(r['stdev']):>14}{pct(r):>10}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace_file", help="Jaeger JSON export or wire-trace JSONL")
    p.add_argument("--llm-prefix", default="llm.", help="operationName prefix of LLM request spans (default: llm.)")
    p.add_argument("--slo-latency", type=float, help="goodput SLO on request latency, seconds")
    p.add_argument("--slo-ttft", type=float, help="goodput SLO on TTFT, seconds")
    p.add_argument("--slo-tpot", type=float, help="goodput SLO on time per output token, seconds")
    p.add_argument("--json", dest="json_out", help="also write the results to this JSON file")
    args = p.parse_args()

    sessions = load_sessions(args.trace_file, args.llm_prefix)
    if not sessions:
        print(f"no requests found in {args.trace_file}", file=sys.stderr)
        return 1

    all_requests = [r for s in sessions for r in s.requests]
    ok_requests = [r for r in all_requests if r.ok]
    print(
        f"{args.trace_file}: {len(sessions)} sessions, {len(all_requests)} requests, "
        f"{sum(not r.ok for r in all_requests)} failed"
    )
    if len(sessions) < 2:
        print("note: inter-replay CVs need at least two sessions; every CV below is n/a")

    prefill_est = None
    if not any(r.ttft is not None or r.tpot is not None for r in ok_requests):
        prefill_est = fit_prefill_estimate(ok_requests)
        print(
            "warning: no request has a TTFT (no TTFT attribute, or no streamed token timing); TTFT CV is n/a and "
            f"TPOT is estimated with a fitted prefill+overhead of {fmt(prefill_est, 3)} s per request",
            file=sys.stderr,
        )
    if args.slo_ttft is None and args.slo_tpot is None and args.slo_latency is None:
        print("note: no SLOs given, so goodput counts every successful request")
    elif args.slo_latency is not None:
        print("note: a latency SLO makes goodput length-sensitive")

    per_session = {s.name: session_metrics(s, args, prefill_est) for s in sessions}
    metric_names = list(next(iter(per_session.values())).keys())
    across_sessions = {m: cv([v[m] for v in per_session.values()]) for m in metric_names}

    # Call position k = the k-th LLM call of every session. Prompt-token CV is how far the replays'
    # inputs diverged from one another by that step (live substitution feeds each replay its own outputs).
    max_calls = max(len(s.requests) for s in sessions)
    per_position: list[dict[str, Any]] = []
    for k in range(max_calls):
        reqs = [s.requests[k] for s in sessions if k < len(s.requests) and s.requests[k].ok]
        per_position.append(
            {
                "position": k,
                "prompt_tokens": cv([float(r.prompt_tokens) for r in reqs]),
                "output_tokens": cv([float(r.output_tokens) for r in reqs]),
                "tpot": cv([request_tpot(r, prefill_est) for r in reqs]),
                "ttft": cv([r.ttft for r in reqs]),
                "latency": cv([r.latency for r in reqs]),
            }
        )

    tpot_label = "tpot_ms(est)" if prefill_est is not None else "tpot_ms"
    print("\nPer-session values")
    print(f"  {'session':<48}{'reqs':>5}{tpot_label:>13}{'ttft_s':>9}{'busy_tok/s':>12}{'lat_s':>9}{'out_tok':>9}")
    for s in sessions:
        m = per_session[s.name]
        tpot_ms = m["tpot"] * 1000 if m["tpot"] is not None else None
        print(
            f"  {s.name:<48}{len(s.requests):>5}{fmt(tpot_ms, 2):>13}{fmt(m['ttft_mean'], 3):>9}"
            f"{fmt(m['busy_output_throughput'], 2):>12}{fmt(m['latency_mean'], 3):>9}{fmt(m['output_tokens_total'], 0):>9}"
        )

    print_cv_table(
        "Inter-replay CV, length-insensitive metrics (expected to be consistent)",
        {m: across_sessions[m] for m in STABLE_METRICS},
    )
    print_cv_table(
        "Inter-replay CV, length-sensitive metrics (vary with how much each replay generated)",
        {m: r for m, r in across_sessions.items() if m not in STABLE_METRICS},
    )

    print("\nInter-replay CV per call position")
    print(
        f"  {'pos':>4}{'n':>4}{'prompt_tok':>12}{'prompt_CV':>11}{'out_tok_CV':>12}{'tpot_CV':>9}{'ttft_CV':>9}{'lat_CV':>9}"
    )
    for row in per_position:
        print(
            f"  {row['position']:>4}{row['latency']['n']:>4}{fmt(row['prompt_tokens']['mean'], 0):>12}"
            f"{pct(row['prompt_tokens']):>11}{pct(row['output_tokens']):>12}{pct(row['tpot']):>9}"
            f"{pct(row['ttft']):>9}{pct(row['latency']):>9}"
        )

    structural = structural_consistency(sessions)
    print("\nInter-replay structural consistency (mean over replay pairs; 100% = identical)")
    print(f"  {'metric':<30}{'pairs':>6}{'mean':>10}{'min pair':>10}")
    for name, r in structural.items():
        mean = "n/a" if r["mean"] is None else f"{100 * r['mean']:.1f}%"
        low = "n/a" if r["min"] is None else f"{100 * r['min']:.1f}%"
        print(f"  {name:<30}{r['pairs']:>6}{mean:>10}{low:>10}")
    dc = structural["dependency_consistency"]
    print(f"  dependency edges per replay: {dc['edges_per_replay']}")
    untraced = {s.name: u for s, u in zip(sessions, dc["untraceable_calls_per_replay"], strict=True) if u}
    if untraced:
        print(f"  calls with an untraceable output (excluded from their pairs): {untraced}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "trace_file": args.trace_file,
                    "slo": {"latency": args.slo_latency, "ttft": args.slo_ttft, "tpot": args.slo_tpot},
                    "prefill_estimate_s": prefill_est,
                    "per_session": per_session,
                    "cv_across_sessions": across_sessions,
                    "cv_per_call_position": per_position,
                    "structural_consistency": structural,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
