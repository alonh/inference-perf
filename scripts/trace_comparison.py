import argparse
import html
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median, stdev

TOKENIZER = "zai-org/GLM-5.3"  # used to count reasoning tokens; neither side reports them separately


def turn_kind(tools, has_text):
    """Classify a turn by what the assistant produced."""
    if len(tools) > 1:
        kind = f"parallel_tools({len(tools)})"
    elif tools:
        kind = "tool_call"
    else:
        return "text" if has_text else "empty"
    return kind + "+text" if has_text else kind


def format_message(reasoning, text, tool_calls):
    """Render a turn's full output as plain text for a hover tooltip. tool_calls is a list of (name, arguments)."""
    parts = []
    if reasoning:
        parts.append(f"[reasoning]\n{reasoning}")
    if text:
        parts.append(f"[text]\n{text}")
    for name, args in tool_calls:
        try:
            args = json.dumps(json.loads(args), indent=2, ensure_ascii=False) if args else ""
        except (TypeError, ValueError):
            pass
        parts.append(f"[tool_call {name}]\n{args}")
    return "\n\n".join(parts) or "(empty)"


def anthropic_input_tokens(usage):
    """Full prompt size: Anthropic reports uncached, cache-read and cache-write input tokens separately."""
    keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    if not any(usage.get(k) is not None for k in keys):
        return None
    return sum(usage.get(k) or 0 for k in keys)


def parse_sse(sse_text):
    input_tokens = output_tokens = None
    tool_names = []
    has_text = False
    reasoning = []
    text = []
    tool_args = {}  # content block index -> accumulated input_json_delta
    for line in sse_text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:])
        except Exception:
            continue
        t = obj.get("type")
        if t in ("message_start", "message_delta"):
            # Some proxies send zeros in message_start and the real input counts in message_delta; keep the last non-zero.
            usage = (obj.get("message", {}) if t == "message_start" else obj).get("usage", {})
            it = anthropic_input_tokens(usage)
            if it or input_tokens is None:
                input_tokens = it
            ot = usage.get("output_tokens")
            if t == "message_delta" and ot is not None:
                output_tokens = ot
        elif t == "content_block_start":
            cb = obj.get("content_block", {})
            if cb.get("type") == "tool_use":
                tool_names.append(cb.get("name"))
                tool_args[obj.get("index")] = (cb.get("name"), [])
            elif cb.get("type") == "text":
                has_text = True
        elif t == "content_block_delta":
            delta = obj.get("delta", {})
            if delta.get("type") == "thinking_delta":
                reasoning.append(delta.get("thinking", ""))
            elif delta.get("type") == "text_delta":
                text.append(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta" and obj.get("index") in tool_args:
                tool_args[obj.get("index")][1].append(delta.get("partial_json", ""))
    reasoning = "".join(reasoning)
    message = format_message(reasoning, "".join(text), [(name, "".join(chunks)) for name, chunks in tool_args.values()])
    return input_tokens, output_tokens, tool_names, has_text, reasoning, message


def parse_openai_message(msg, usage):
    """Shared tail of the OpenAI parsers: msg holds content, reasoning and a list of (name, arguments) tool calls."""
    text = msg["content"]
    reasoning = msg["reasoning"]
    tool_calls = msg["tool_calls"]
    tools = [name for name, _ in tool_calls]
    out = (usage or {}).get("completion_tokens")
    return (usage or {}).get("prompt_tokens"), out, tools, bool(text), reasoning, format_message(reasoning, text, tool_calls)


def parse_openai_sse(sse_text):
    """Parse an OpenAI chat.completion.chunk stream (reasoning arrives as delta.reasoning_content / delta.reasoning)."""
    reasoning, text = [], []
    tool_args = {}  # tool call index -> [name, argument chunks]
    usage = None
    for line in sse_text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:])
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            text.append(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                entry = tool_args.setdefault(tc.get("index", 0), [None, []])
                fn = tc.get("function") or {}
                entry[0] = entry[0] or fn.get("name")
                entry[1].append(fn.get("arguments") or "")
    msg = {
        "content": "".join(text),
        "reasoning": "".join(reasoning),
        "tool_calls": [(name, "".join(chunks)) for name, chunks in (tool_args[i] for i in sorted(tool_args))],
    }
    return parse_openai_message(msg, usage)


def parse_openai_json(body):
    """Parse a non-streaming OpenAI chat.completion response."""
    obj = json.loads(body)
    message = (obj.get("choices") or [{}])[0].get("message") or {}
    msg = {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
        "tool_calls": [
            ((tc.get("function") or {}).get("name"), (tc.get("function") or {}).get("arguments"))
            for tc in message.get("tool_calls") or []
        ],
    }
    return parse_openai_message(msg, obj.get("usage"))


def parse_response(body):
    """Dispatch on the recorded response format: Anthropic SSE, OpenAI SSE, or OpenAI non-streaming JSON."""
    if body.lstrip().startswith("{"):
        return parse_openai_json(body)
    if '"chat.completion.chunk"' in body:
        return parse_openai_sse(body)
    return parse_sse(body)


def message_text(content):
    """Text of a message content field, which may be a string or a list of text/content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text") or b.get("content") or "" for b in content or [] if isinstance(b, dict))


def conversation_key(first_text):
    """Identify which conversation a call belongs to (main agent, title generator, subagent, ...) by its system prompt."""
    return " ".join(first_text.split())[:KEY_CHARS]


KEY_CHARS = 200


def recording_key(request):
    req = json.loads(request) if isinstance(request, str) else request or {}
    if req.get("system"):
        return conversation_key(message_text(req["system"]))
    msgs = req.get("messages") or [{}]
    return conversation_key(message_text(msgs[0].get("content")))


def load_recording(path):
    with open(path) as f:
        calls = [json.loads(l) for l in f]
    turns = []
    for c in calls:
        # Skip failed calls, empty-bodied probes (e.g. the bare "/anthropic" connectivity check) and token-counting
        # calls, which generate nothing and are never replayed.
        if c.get("status") != 200 or not c.get("response") or c.get("path", "").endswith("/count_tokens"):
            continue
        o_in, o_out, tools, has_text, reasoning, message = parse_response(c["response"])
        turns.append(
            {
                "tools": tools,
                "kind": turn_kind(tools, has_text),
                "key": recording_key(c.get("request")),
                "in": o_in,
                "out": o_out,
                "ms": c["total_ms"],
                "reasoning": reasoning,
                "message": message,
                "error": False,
            }
        )
    return turns, len(calls) - len(turns)


def recording_agent(path):
    """Name the agent that made the recorded calls: the x-title header, else the product token of the user-agent.

    Only successful calls count, since connectivity probes may come from a different client (e.g. a bare "node").
    """
    agents = Counter()
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            if c.get("status") != 200 or not c.get("response"):
                continue
            headers = {k.lower(): v for k, v in (c.get("headers") or {}).items()}
            agent = headers.get("x-title") or headers.get("user-agent", "").split("/", 1)[0]
            if agent:
                agents[agent] += 1
    return agents.most_common(1)[0][0] if agents else None


def get_tag(span, key):
    for t in span.get("tags", []):
        if t["key"] == key:
            return t["value"]
    return None


def fetch_jaeger(src):
    """Load a local Jaeger JSON export."""
    with open(src) as f:
        return json.load(f)


def iter_traces(sources):
    """Yield (src, trace) for every trace in every source; a Jaeger JSON export may hold many traces."""
    for src in sources:
        for trace in fetch_jaeger(src)["data"]:
            yield src, trace


def session_sort_key(trace):
    """Order traces by session.id, comparing the numeric part of a 'traceN_...' prefix numerically."""
    sid = (
        next(
            (get_tag(s, "session.id") for s in trace["spans"] if s["operationName"].startswith("session.")),
            None,
        )
        or trace["traceID"]
    )
    prefix = sid.split("_", 1)[0].removeprefix("trace")
    return (0, int(prefix), sid) if prefix.isdigit() else (1, 0, sid)


def load_jaeger(trace):
    spans = trace["spans"]
    session = next(
        ({t["key"]: t["value"] for t in s.get("tags", [])} for s in spans if s["operationName"].startswith("session.")),
        None,
    )
    llm_spans = sorted([s for s in spans if s["operationName"] == "llm.chat.completions"], key=lambda s: s["startTime"])
    turns = []
    for s in llm_spans:
        tools = []
        tool_calls = []
        text = get_tag(s, "gen_ai.output.text") or ""
        out_msg = get_tag(s, "gen_ai.output.message")
        if out_msg:
            obj = json.loads(out_msg)
            for tc in obj.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                tools.append(fn.get("name"))
                tool_calls.append((fn.get("name"), fn.get("arguments")))
            text = text or obj.get("content") or ""
        has_text = bool(text)
        reasoning = get_tag(s, "gen_ai.output.reasoning") or ""
        in_msgs = json.loads(get_tag(s, "gen_ai.input.messages") or "[]") or [{}]
        turns.append(
            {
                "tools": tools,
                "kind": turn_kind(tools, has_text),
                "key": conversation_key(message_text(in_msgs[0].get("content") or in_msgs[0].get("parts"))),
                "in": get_tag(s, "gen_ai.usage.prompt_tokens"),
                "out": get_tag(s, "gen_ai.usage.completion_tokens"),
                "ms": s["duration"] / 1000.0,
                "reasoning": reasoning,
                "message": format_message(reasoning, text, tool_calls),
                "error": get_tag(s, "otel.status_code") == "ERROR" or get_tag(s, "error") in (True, "true"),
            }
        )
    return session, turns


def signature(turns):
    return [tuple(t["tools"]) or (t["kind"],) for t in turns]


def fmt_tools(t):
    return ("+".join(t["tools"]) or "-") if t else "(missing)"


def pick_recording(session, jaeger, recordings):
    """Match via the session span's session.file_path; fall back to turn-signature similarity."""
    file_path = (session or {}).get("session.file_path")
    if file_path and Path(file_path).name in recordings:
        return Path(file_path).name, "session.file_path"
    scored = [
        (SequenceMatcher(None, signature(turns), signature(jaeger), autojunk=False).ratio(), name)
        for name, (turns, _) in recordings.items()
    ]
    score, name = max(scored)
    return name, f"similarity {score:.2f}" + (f" (session.file_path {file_path!r} not found)" if file_path else "")


STATUSES = ["match", "kind-mismatch", "tool-mismatch", "missing-in-jaeger", "extra-in-jaeger"]


def make_token_counter(name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


def content_tokens(out, think):
    """out − think; None when out is missing or smaller than think (e.g. a stream cut off before its usage chunk)."""
    if out is None or think is None or out < think:
        return None
    return out - think


def align(orig, jaeger):
    """Pair recorded and replayed turns as (orig index | None, jaeger index | None).

    The k-th recorded turn of each conversation (system prompt) pairs with the k-th replayed turn of the same
    conversation, so a side-call such as a title generator that runs at a different point in the replay, or a
    dropped turn in another conversation, doesn't shift every later turn. Rows follow recording order; replayed
    turns with no recorded counterpart are appended.
    """
    by_key = {}
    for ji, t in enumerate(jaeger):
        by_key.setdefault(t["key"], []).append(ji)
    seen = {}
    pairs = []
    for oi, t in enumerate(orig):
        k = seen.get(t["key"], 0)
        seen[t["key"]] = k + 1
        js = by_key.get(t["key"], [])
        pairs.append((oi, js[k] if k < len(js) else None))
    for key, js in by_key.items():
        pairs += [(None, ji) for ji in js[seen.get(key, 0) :]]
    return pairs


def levenshtein(a, b):
    """Edit distance between two sequences (unit-cost insert, delete, substitute)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def call_sequence(turns):
    """Call-type sequence S = [user, assistant, tool, assistant, ...] of a session.

    Each LLM turn is an assistant call, followed by one tool call per tool it invoked. The turn is preceded by a user
    message when it opens its conversation or the previous turn of that conversation called no tools (so the next input
    can only be a new user message). Conversations (system prompts) are concatenated in order of first appearance, each
    as one block, so a side-call that ran at a different point in the replay doesn't count as an edit. First appearance
    rather than sorted key keeps a conversation whose key changed mid-session (Claude Code's billing-header hash changes
    at compaction) in time order.
    """
    by_key = {}
    for t in turns:
        by_key.setdefault(t["key"], []).append(t)
    seq = []
    for key in by_key:  # first-appearance order
        prev = None
        for t in by_key[key]:
            if prev is None or not prev["tools"]:
                seq.append("user")
            seq.append("assistant")
            seq += ["tool"] * len(t["tools"])
            prev = t
    return seq


def workflow_fidelity(orig, jaeger):
    """WF = 1 − Lev(S_orig, S_replay) / max(|S_orig|, |S_replay|); None when both sequences are empty."""
    a, b = call_sequence(orig), call_sequence(jaeger)
    longest = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / longest if longest else None


def tool_fidelity(o, j):
    """Share of a turn's tool calls reproduced: |T_orig ∩ T_replay| / max(|T_orig|, |T_replay|) over tool-name multisets.

    A missing turn counts as no tool calls; None when neither side called a tool (the turn says nothing about tools).
    """
    a = Counter(o["tools"] if o else [])
    b = Counter(j["tools"] if j else [])
    longest = max(sum(a.values()), sum(b.values()))
    return sum((a & b).values()) / longest if longest else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def std(vals):
    """Sample std (n − 1) of the non-None values; None with fewer than two, where the spread is undefined."""
    vals = [v for v in vals if v is not None]
    return stdev(vals) if len(vals) > 1 else None


def percentile(vals, q):
    """Linearly interpolated q-th percentile (0–100) of the non-None values; None when there are none."""
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def token_ratio(orig, replayed, key, agg=sum):
    """agg(replayed tokens) ÷ agg(recorded tokens) of `key` ("in" or "out") over every turn of each side of the session.

    With sum this is R_s; with max over "in" it is the peak context ratio. Turns are aggregated per side rather than per
    pair, so turns the replay dropped or added count toward the workload. None when the recording reports no such tokens.
    """
    o_vals = [t[key] for t in orig if t[key] is not None]
    j_vals = [t[key] for t in replayed if t[key] is not None]
    o_agg = agg(o_vals) if o_vals else 0
    j_agg = agg(j_vals) if j_vals else 0
    return j_agg / o_agg if o_agg else None


def compare(orig, jaeger, count_tokens, session_errored=False):
    """Align recorded and replayed turns by conversation; return per-turn rows and status counts.

    Error turns get no tool-call fidelity: replayed spans that errored, and, in an errored session, recorded turns the
    replay never reached because the session aborted.
    """
    rows = []
    counts = dict.fromkeys(STATUSES, 0)
    for oi, ji in align(orig, jaeger):
        o = orig[oi] if oi is not None else None
        j = jaeger[ji] if ji is not None else None
        if j is None:
            status = "missing-in-jaeger"
        elif o is None:
            status = "extra-in-jaeger"
        elif o["kind"] != j["kind"]:
            status = "kind-mismatch"
        elif o["tools"] != j["tools"]:
            status = "tool-mismatch"
        else:
            status = "match"
        counts[status] += 1
        o_in = o["in"] if o else None
        j_in = j["in"] if j else None
        o_out = o["out"] if o else None
        j_out = j["out"] if j else None
        o_ms = o["ms"] if o else None
        j_ms = j["ms"] if j else None
        o_think = count_tokens(o["reasoning"]) if o else None
        j_think = count_tokens(j["reasoning"]) if j else None
        o_content = content_tokens(o_out, o_think)
        j_content = content_tokens(j_out, j_think)
        error_turn = j["error"] if j else session_errored
        rows.append(
            {
                "i": f"{fmt(oi)} / {fmt(ji)}",
                "o_kind": o["kind"] if o else "(missing)",
                "j_kind": j["kind"] if j else "(missing)",
                "o_tools": fmt_tools(o),
                "j_tools": fmt_tools(j),
                "o_in": o_in,
                "j_in": j_in,
                "d_in": j_in - o_in if (j_in is not None and o_in is not None) else None,
                "d_in_pct": 100.0 * (j_in - o_in) / o_in if (j_in is not None and o_in) else None,
                "o_out": o_out,
                "j_out": j_out,
                "d_out": j_out - o_out if (j_out is not None and o_out is not None) else None,
                "o_think": o_think,
                "j_think": j_think,
                "o_content": o_content,
                "j_content": j_content,
                "d_content": j_content - o_content if (j_content is not None and o_content is not None) else None,
                "o_reasoning": o["reasoning"] if o else "",
                "j_reasoning": j["reasoning"] if j else "",
                "o_message": o["message"] if o else "",
                "j_message": j["message"] if j else "",
                "o_ms": o_ms,
                "j_ms": j_ms,
                "ratio": j_ms / o_ms if (o_ms and j_ms is not None) else None,
                "tf": None if error_turn else tool_fidelity(o, j),
                "status": status,
            }
        )
    return rows, counts


def fmt(v, spec=""):
    return "–" if v is None else format(v, spec)


MEAN_COLS = [
    "o_in",
    "j_in",
    "d_in",
    "d_in_pct",
    "o_out",
    "j_out",
    "d_out",
    "o_think",
    "j_think",
    "o_content",
    "j_content",
    "d_content",
    "o_ms",
    "j_ms",
    "ratio",
    "tf",
]


def column_means(rows):
    """Mean of each numeric column, skipping turns where the value is missing."""
    means = {}
    for col in MEAN_COLS:
        vals = [row[col] for row in rows if row[col] is not None]
        means[col] = sum(vals) / len(vals) if vals else None
    return means


def delta_stats(rows, o_col, j_col):
    """Mean and std of Rep − Rec, and Σ(Rep − Rec) ÷ ΣRec as a signed percentage, over turns where both are present.

    The total-based share weights each turn by its size, so a few tiny turns that grow a lot can't flip its sign
    against the mean. Std is the sample std (n − 1), so a single turn gives none. The medians are of per-turn Rep − Rec
    and of per-turn (Rep − Rec) ÷ Rec (skipping turns whose Rec is 0), showing the typical turn rather than the heavy tail.
    """
    pairs = [(row[o_col], row[j_col]) for row in rows if row[o_col] is not None and row[j_col] is not None]
    deltas = [j - o for o, j in pairs]
    total = sum(o for o, _ in pairs)
    return {
        "mean": mean(deltas),
        "std": std(deltas),
        "pct": 100.0 * sum(deltas) / total if total else None,
        "median": median(deltas) if deltas else None,
        "median_pct": median(pcts) if (pcts := [100.0 * (j - o) / o for o, j in pairs if o]) else None,
    }


def overall_summary(reports):
    """Aggregate across all sessions; mismatch percentages are over turns present on both sides."""
    rows = [row for r in reports for row in r["rows"]]
    paired = [row for row in rows if row["status"] in ("match", "kind-mismatch", "tool-mismatch")]
    d_in_pct = [abs(row["d_in_pct"]) for row in rows if row["d_in_pct"] is not None]

    def pct(status):
        return 100.0 * sum(row["status"] == status for row in paired) / len(paired) if paired else None

    return {
        "sessions": len(reports),
        "count_mismatch_sessions": sum(len(r["orig"]) != len(r["jaeger"]) for r in reports),
        "missing_turns": sum(r["counts"]["missing-in-jaeger"] for r in reports),
        "extra_turns": sum(r["counts"]["extra-in-jaeger"] for r in reports),
        "paired_turns": len(paired),
        "kind_mismatch_pct": pct("kind-mismatch"),
        "tool_mismatch_pct": pct("tool-mismatch"),
        "d_in": delta_stats(rows, "o_in", "j_in"),
        "in_within_tol": sum(p <= IN_TOLERANCE_PCT for p in d_in_pct),
        "in_compared": len(d_in_pct),
        "d_out": delta_stats(rows, "o_out", "j_out"),
        "d_content": delta_stats(rows, "o_content", "j_content"),
        "d_ms": delta_stats(rows, "o_ms", "j_ms"),
        "avg_wf": mean(r["wf"] for r in reports),
        "std_wf": std(r["wf"] for r in reports),
        "wf_sessions": sum(r["wf"] is not None for r in reports),
        "tf_turns": mean(row["tf"] for row in rows),
        "tf_turns_std": std(row["tf"] for row in rows),
        "tf_turn_count": sum(row["tf"] is not None for row in rows),
        "tf_sessions": mean(r["tf"] for r in reports),
        "tf_sessions_std": std(r["tf"] for r in reports),
        "tf_session_count": sum(r["tf"] is not None for r in reports),
        "d_content_turn_count": sum(row["d_content"] is not None for row in rows),
        "d_content_sessions": mean(r["d_content"] for r in reports),
        "d_content_session_count": sum(r["d_content"] is not None for r in reports),
        "in_ratio_median": median(ratios)
        if (ratios := [r["in_ratio"] for r in reports if r["in_ratio"] is not None])
        else None,
        "in_ratio_p95": percentile(ratios, 95),
        "in_ratio_sessions": len(ratios),
        "out_ratio_median": (
            median(out_ratios) if (out_ratios := [r["out_ratio"] for r in reports if r["out_ratio"] is not None]) else None
        ),
        "out_ratio_p95": percentile(out_ratios, 95),
        "out_ratio_sessions": len(out_ratios),
        "peak_ratio_median": (
            median(peak_ratios) if (peak_ratios := [r["peak_ratio"] for r in reports if r["peak_ratio"] is not None]) else None
        ),
        "peak_ratio_p95": percentile(peak_ratios, 95),
        "peak_ratio_sessions": len(peak_ratios),
    }


IN_TOLERANCE_PCT = 5.0  # |Δ in| at or below this share of Rec in counts as the input load being preserved


def fmt_ratio(v):
    return fmt(v) if v is None else f"{v:.2f}x"


def fmt_delta(d, unit):
    return (
        f"{fmt(d['mean'], '+.1f')} ± {fmt(d['std'], '.1f')} {unit} (total Δ {fmt(d['pct'], '+.1f')}%); "
        f"median {fmt(d['median'], '+.1f')} {unit} ({fmt(d['median_pct'], '+.1f')}%)"
    )


OVERALL_ITEMS = [
    (
        "LLM event count mismatches",
        "Sessions whose replayed llm.chat.completions span count differs from the recorded turn count",
        lambda s: (
            f"{s['count_mismatch_sessions']} of {s['sessions']} sessions "
            f"({s['missing_turns']} turns missing in Jaeger, {s['extra_turns']} extra)"
        ),
    ),
    (
        "Kind mismatches",
        "kind-mismatch turns as a share of turns present on both sides",
        lambda s: f"{fmt(s['kind_mismatch_pct'], '.1f')}% of {s['paired_turns']} paired turns",
    ),
    (
        "Tool mismatches",
        "tool-mismatch turns as a share of turns present on both sides",
        lambda s: f"{fmt(s['tool_mismatch_pct'], '.1f')}% of {s['paired_turns']} paired turns",
    ),
    (
        "Median Input Ratio",
        "Median over sessions of R_s = Σ Rep in ÷ Σ Rec in, summed over every turn of each side of the session; "
        "how much larger (> 1) or smaller (< 1) the replayed input workload was than the recorded session",
        lambda s: f"{fmt_ratio(s['in_ratio_median'])} over {s['in_ratio_sessions']} sessions",
    ),
    (
        "P95 Input Ratio",
        "95th percentile (linear interpolation) over sessions of R_s = Σ Rep in ÷ Σ Rec in",
        lambda s: fmt_ratio(s["in_ratio_p95"]),
    ),
    (
        "Median Output Ratio",
        "Median over sessions of R_s = Σ Rep out ÷ Σ Rec out, summed over every turn of each side of the session; "
        "how much larger (> 1) or smaller (< 1) the replayed output workload was than the recorded session",
        lambda s: f"{fmt_ratio(s['out_ratio_median'])} over {s['out_ratio_sessions']} sessions",
    ),
    (
        "P95 Output Ratio",
        "95th percentile (linear interpolation) over sessions of R_s = Σ Rep out ÷ Σ Rec out",
        lambda s: fmt_ratio(s["out_ratio_p95"]),
    ),
    (
        "Median Peak Context Ratio",
        "Median over sessions of max Rep in ÷ max Rec in, the largest prompt of each side of the session; "
        "how much larger (> 1) or smaller (< 1) the replay's peak context was than the recorded session's",
        lambda s: f"{fmt_ratio(s['peak_ratio_median'])} over {s['peak_ratio_sessions']} sessions",
    ),
    (
        "P95 Peak Context Ratio",
        "95th percentile (linear interpolation) over sessions of max Rep in ÷ max Rec in",
        lambda s: fmt_ratio(s["peak_ratio_p95"]),
    ),
    (
        "Avg. Δ in",
        "Mean ± std of Rep in − Rec in over all turns where both are present, in tokens; "
        "Σ(Rep in − Rec in) ÷ ΣRec in in parentheses; then the median per-turn Δ in and median per-turn Δ in ÷ Rec in",
        lambda s: fmt_delta(s["d_in"], "tokens"),
    ),
    (
        "Input load preserved",
        f"Turns whose |Rep in − Rec in| is at most {IN_TOLERANCE_PCT:g}% of Rec in",
        lambda s: f"{s['in_within_tol']} of {s['in_compared']} turns within ±{IN_TOLERANCE_PCT:g}%",
    ),
    (
        "Avg. Δ out",
        "Mean ± std of Rep out − Rec out over all turns where both are present, in tokens; "
        "Σ(Rep out − Rec out) ÷ ΣRec out in parentheses; then the median per-turn Δ out and median per-turn Δ out ÷ Rec out",
        lambda s: fmt_delta(s["d_out"], "tokens"),
    ),
    (
        "Avg. Δ content",
        "Mean ± std of Rep content − Rec content (out − think) over all turns where both are present, in tokens; "
        "Σ(Rep content − Rec content) ÷ ΣRec content in parentheses; then the median per-turn Δ content and median per-turn Δ content ÷ Rec content; then the mean of per-session means",
        lambda s: (
            f"{fmt_delta(s['d_content'], 'tokens')} over {s['d_content_turn_count']} turns, "
            f"{fmt(s['d_content_sessions'], '+.1f')} mean of {s['d_content_session_count']} sessions"
        ),
    ),
    (
        "Avg. Δ latency",
        "Mean ± std of Rep ms − Rec ms over all turns where both are present; Σ(Rep ms − Rec ms) ÷ ΣRec ms in parentheses; then the median per-turn Δ ms and median per-turn Δ ms ÷ Rec ms",
        lambda s: fmt_delta(s["d_ms"], "ms"),
    ),
    (
        "Workflow fidelity",
        "WF = 1 − Lev(S_orig, S_replay) ÷ max(|S_orig|, |S_replay|), where S is the session's call-type sequence "
        "[user, assistant, tool, assistant, …]; mean ± std over sessions, excluding sessions whose session span errored",
        lambda s: f"{fmt(s['avg_wf'], '.3f')} ± {fmt(s['std_wf'], '.3f')} (mean ± std of {s['wf_sessions']} sessions)",
    ),
    (
        "Tool-call fidelity",
        "Per turn: |T_orig ∩ T_replay| ÷ max(|T_orig|, |T_replay|) over tool-name multisets, for turns where either "
        "side called a tool (a missing turn has no tools), excluding error turns (errored replayed spans, and turns "
        "an errored session never reached); "
        "mean ± std over all such turns, and mean ± std of per-session means",
        lambda s: (
            f"{fmt(s['tf_turns'], '.3f')} ± {fmt(s['tf_turns_std'], '.3f')} over {s['tf_turn_count']} turns, "
            f"{fmt(s['tf_sessions'], '.3f')} ± {fmt(s['tf_sessions_std'], '.3f')} mean of {s['tf_session_count']} sessions"
        ),
    ),
]


CSS = """
:root { --bg:#fff; --fg:#1f2328; --muted:#656d76; --border:#d0d7de; --head:#f6f8fa;
        --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; --warn-bg:#fff8c5; --bad-bg:#ffebe9; --miss-bg:#eef1f4; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --border:#30363d; --head:#161b22;
          --ok:#3fb950; --warn:#d29922; --bad:#f85149; --warn-bg:#2d2410; --bad-bg:#3a1618; --miss-bg:#1c2128; }
}
body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       margin:0; padding:24px 16px; }
main { max-width:1400px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 6px; }
.muted { color:var(--muted); } code { font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }
.scroll { overflow-x:auto; border:1px solid var(--border); border-radius:6px; margin-top:8px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { padding:5px 10px; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }
th { background:var(--head); font-weight:600; position:sticky; top:0; }
td.num, th.num { text-align:right; }
tr:last-child td { border-bottom:none; }
tr.kind-mismatch, tr.tool-mismatch { background:var(--warn-bg); }
tr.missing-in-jaeger, tr.extra-in-jaeger { background:var(--miss-bg); }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600; }
.s-match { color:var(--ok); } .s-kind-mismatch, .s-tool-mismatch { color:var(--warn); }
.s-missing-in-jaeger, .s-extra-in-jaeger { color:var(--muted); }
.st-OK { color:var(--ok); } .st-ERROR { color:var(--bad); font-weight:600; }
.meta { display:grid; grid-template-columns:max-content 1fr; gap:2px 14px; margin-top:6px; }
.filter { margin:12px 0 0; } .filter label { margin-right:14px; cursor:pointer; }
body.only-diff tr.match { display:none; }
td.in-off { color:var(--bad); font-weight:600; }
tr.mean td { font-weight:600; background:var(--head); border-top:2px solid var(--border); }
th[title], .meta span[title], .pill[title] { cursor:help; text-decoration:underline dotted var(--muted); }
"""

STATUS_TIPS = {
    "match": "Same turn kind and same tool names (in order) on both sides",
    "kind-mismatch": "Turn kind differs, e.g. text vs. tool_call, or a different number of parallel tool calls",
    "tool-mismatch": "Same turn kind, but different tool names were called",
    "missing-in-jaeger": "The recording has this turn but the replay produced no matching LLM span",
    "extra-in-jaeger": "The replay produced an LLM span beyond the end of the recording",
}

OVERVIEW_COLS = [
    ("Session", "session.id from the trace's session span; click to jump to its per-turn table", False),
    ("Status", "otel.status_code of the session span", False),
    ("Recording", "Recorded .jsonl file this trace was compared against", False),
    ("Recorded", "Successful (HTTP 200, non-empty response) LLM calls in the recording", True),
    ("Replayed", "llm.chat.completions spans in the Jaeger trace", True),
    (
        "WF",
        "Workflow fidelity: 1 − Lev(S_orig, S_replay) ÷ max(|S_orig|, |S_replay|) over call-type sequences; "
        "– for errored sessions",
        True,
    ),
    (
        "TF",
        "Tool-call fidelity: mean per-turn tool-call fidelity over turns where either side called a tool, "
        "excluding error turns",
        True,
    ),
    ("Δ content", "Mean Rep content − Rec content over turns where both are present, in tokens", True),
    ("In ratio", "R_s = Σ Rep in ÷ Σ Rec in over every turn of each side of the session", True),
    ("Out ratio", "R_s = Σ Rep out ÷ Σ Rec out over every turn of each side of the session", True),
    ("Peak ratio", "Peak context ratio: max Rep in ÷ max Rec in over every turn of each side of the session", True),
] + [(s, f"Turns with status {s}: {STATUS_TIPS[s]}", True) for s in STATUSES]

TURN_COLS = [
    (
        "#",
        "Recorded / replayed turn index; turns are paired within the same conversation (system prompt), "
        "so a side-call that moved in the replay doesn't shift later turns",
        True,
    ),
    (
        "Recorded kind",
        "What the recorded assistant produced: text, tool_call, parallel_tools(N), +text, or empty; "
        "hover a cell for the full output message",
        False,
    ),
    (
        "Replayed kind",
        "What the replayed assistant produced, classified the same way; hover a cell for the full output message",
        False,
    ),
    ("Recorded tools", "Tool names called in the recorded turn, in order; hover a cell for the full output message", False),
    (
        "Replayed tools",
        "Tool names in gen_ai.output.message tool_calls of the replayed span, in order; hover a cell for the full output message",
        False,
    ),
    (
        "Rec in",
        "Recorded input tokens: prompt_tokens (OpenAI), or input + cache_read + cache_creation input tokens (Anthropic)",
        True,
    ),
    ("Rep in", "Replayed input tokens (gen_ai.usage.prompt_tokens)", True),
    ("Δ in", "Rep in − Rec in, in tokens", True),
    ("Δ in %", f"(Rep in − Rec in) ÷ Rec in; highlighted when beyond ±{IN_TOLERANCE_PCT:g}%", True),
    ("Rec out", "Recorded output tokens (message_delta usage.output_tokens); includes reasoning, text and tool calls", True),
    ("Rep out", "Replayed output tokens (gen_ai.usage.completion_tokens); includes reasoning, text and tool calls", True),
    ("Δ out", "Rep out − Rec out, in tokens", True),
    (
        "Rec think",
        "Recorded reasoning tokens (joined thinking_delta text, tokenized locally); hover a cell for the text",
        True,
    ),
    ("Rep think", "Replayed reasoning tokens (gen_ai.output.reasoning, tokenized locally); hover a cell for the text", True),
    (
        "Rec content",
        "Rec out − Rec think: recorded tokens spent on text and tool calls (plus any special tokens); "
        "– when out < think, i.e. the reported out is unreliable",
        True,
    ),
    (
        "Rep content",
        "Rep out − Rep think: replayed tokens spent on text and tool calls (plus any special tokens); "
        "– when out < think, e.g. a stream cut off before its usage chunk",
        True,
    ),
    ("Δ content", "Rep content − Rec content, in tokens", True),
    ("Rec ms", "Recorded call latency (total_ms), in milliseconds", True),
    ("Rep ms", "Replayed LLM span duration, in milliseconds", True),
    ("Ratio", "Rep ms ÷ Rec ms; below 1 means the replay was faster", True),
    (
        "TF",
        "Tool-call fidelity: |T_orig ∩ T_replay| ÷ max(|T_orig|, |T_replay|) over tool-name multisets; "
        "– when neither side called a tool, the replayed span errored, or an errored session never reached the turn",
        True,
    ),
    ("Status", "How the two turns compare; hover a pill for its meaning", False),
]


def render_html(reports, tokenizer_name, recordings_dir, replayed):
    def e(v):
        return html.escape(str(v))

    def header_row(cols):
        return (
            "<tr>"
            + "".join(f'<th class="{"num" if num else ""}" title="{e(tip)}">{e(label)}</th>' for label, tip, num in cols)
            + "</tr>"
        )

    def in_off(pct):
        return " in-off" if pct is not None and abs(pct) > IN_TOLERANCE_PCT else ""

    def pill(s):
        return f'<span class="pill s-{s}" title="{e(STATUS_TIPS[s])}">{s}</span>'

    out = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Trace Comparison</title><style>{CSS}</style></head><body><main>",
        "<h1>Trace comparison</h1>",
        f'<div class="muted">Recorded calls in <code>{e(str(recordings_dir))}</code> vs. {e(replayed)}. '
        "Turns are paired by order within the same conversation (system prompt). All token columns: <em>in</em> is the server-reported prompt size; <em>out</em> is the server-reported count (includes reasoning); "
        f"<em>think</em> is the reasoning text tokenized locally with <code>{e(tokenizer_name)}</code>. "
        "Hover headers, labels and statuses for details.</div>",
        '<div class="filter"><label title="Hide rows whose status is match"><input type="checkbox" '
        "onchange=\"document.body.classList.toggle('only-diff', this.checked)\"> Show only non-matching turns</label></div>",
    ]
    summary = overall_summary(reports)
    out.append('<h2>Overall</h2><div class="meta">')
    for label, tip, render in OVERALL_ITEMS:
        out.append(f'<span class="muted" title="{e(tip)}">{e(label)}</span><span>{e(render(summary))}</span>')
    out.append("</div>")
    out.append('<h2>Overview</h2><div class="scroll"><table>' + header_row(OVERVIEW_COLS))
    for r in reports:
        st = r["session"].get("otel.status_code", "?")
        out.append(
            f'<tr><td><a href="#{e(r["anchor"])}">{e(r["session"].get("session.id", r["src"]))}</a></td>'
            f'<td class="st-{e(st)}">{e(st)}</td><td><code>{e(r["recording"])}</code></td>'
            f'<td class="num">{len(r["orig"])}</td><td class="num">{len(r["jaeger"])}</td>'
            f'<td class="num">{fmt(r["wf"], ".3f")}</td><td class="num">{fmt(r["tf"], ".3f")}</td>'
            f'<td class="num">{fmt(r["d_content"], "+.1f")}</td>'
            f'<td class="num">{fmt(r["in_ratio"], ".2f")}</td>'
            f'<td class="num">{fmt(r["out_ratio"], ".2f")}</td>'
            f'<td class="num">{fmt(r["peak_ratio"], ".2f")}</td>'
            + "".join(f'<td class="num s-{s}">{r["counts"][s] or ""}</td>' for s in STATUSES)
            + "</tr>"
        )
    out.append("</table></div>")

    for r in reports:
        sess = r["session"]
        st = sess.get("otel.status_code", "?")
        desc = f" ({sess['otel.status_description']})" if "otel.status_description" in sess else ""
        skipped = f", {r['n_failed']} non-200, empty or count_tokens calls skipped" if r["n_failed"] else ""
        out += [
            f'<h2 id="{e(r["anchor"])}">{e(sess.get("session.id", r["src"]))}</h2><div class="meta">',
            '<span class="muted" title="Open this trace in the Jaeger UI, or the replayed recording file">Replayed from</span>'
            f'<span><a href="{e(r["url"])}">{e(r["url"])}</a></span>',
            '<span class="muted" title="otel.status_code (and otel.status_description) of the session span">Status</span>'
            f'<span class="st-{e(st)}">{e(st + desc)}</span>',
            '<span class="muted" title="x-title header of the recorded calls, else the product in their user-agent">Agent</span>'
            f"<span>{e(r['agent'] or '?')}</span>",
            '<span class="muted" title="Matched via the session span\'s session.file_path, '
            'else by the most similar sequence of turn tool calls">Recording</span>'
            f"<span><code>{e(r['recording'])}</code> "
            f"({len(r['orig'])} turns{e(skipped)}) — matched by {e(r['how'])}</span>",
            '<span class="muted" title="llm.chat.completions spans in the trace, out of all session events">Replayed</span>'
            f"<span>{len(r['jaeger'])} LLM turns"
            + (f" of {e(sess['session.num_events'])} session events" if "session.num_events" in sess else "")
            + "</span>",
            '<span class="muted" title="1 − Lev(S_orig, S_replay) ÷ max(|S_orig|, |S_replay|) over call-type sequences '
            '[user, assistant, tool, assistant, …]">Workflow fidelity</span>'
            + (
                "<span>– (excluded: session errored)</span>"
                if r["wf"] is None and st == "ERROR"
                else f"<span>{fmt(r['wf'], '.3f')} ({len(r['seq_orig'])} recorded vs. {len(r['seq_jaeger'])} replayed calls)</span>"
            ),
            '<span class="muted" title="Mean per-turn tool-call fidelity over turns where either side called a tool">'
            f"Tool-call fidelity</span><span>{fmt(r['tf'], '.3f')}</span>",
            '<span class="muted" title="Mean Rep content − Rec content over turns where both are present">'
            f"Avg. Δ content</span><span>{fmt(r['d_content'], '+.1f')} tokens</span>",
            '<span class="muted" title="Σ Rep in ÷ Σ Rec in over every turn of each side of the session">'
            f"Input ratio</span><span>{fmt_ratio(r['in_ratio'])}</span>",
            '<span class="muted" title="Σ Rep out ÷ Σ Rec out over every turn of each side of the session">'
            f"Output ratio</span><span>{fmt_ratio(r['out_ratio'])}</span>",
            '<span class="muted" title="max Rep in ÷ max Rec in over every turn of each side of the session">'
            f"Peak context ratio</span><span>{fmt_ratio(r['peak_ratio'])}</span></div>",
            '<div class="scroll"><table>' + header_row(TURN_COLS),
        ]
        for row in r["rows"]:
            s = row["status"]
            out.append(
                f'<tr class="{s}"><td class="num">{row["i"]}</td>'
                f'<td title="{e(row["o_message"])}">{e(row["o_kind"])}</td>'
                f'<td title="{e(row["j_message"])}">{e(row["j_kind"])}</td>'
                f'<td title="{e(row["o_message"])}">{e(row["o_tools"])}</td>'
                f'<td title="{e(row["j_message"])}">{e(row["j_tools"])}</td>'
                f'<td class="num">{fmt(row["o_in"])}</td><td class="num">{fmt(row["j_in"])}</td>'
                f'<td class="num">{fmt(row["d_in"], "+d")}</td>'
                f'<td class="num{in_off(row["d_in_pct"])}">{fmt(row["d_in_pct"], "+.1f")}</td>'
                f'<td class="num">{fmt(row["o_out"])}</td><td class="num">{fmt(row["j_out"])}</td>'
                f'<td class="num">{fmt(row["d_out"], "+d")}</td>'
                f'<td class="num" title="{e(row["o_reasoning"])}">{fmt(row["o_think"])}</td>'
                f'<td class="num" title="{e(row["j_reasoning"])}">{fmt(row["j_think"])}</td>'
                f'<td class="num">{fmt(row["o_content"])}</td><td class="num">{fmt(row["j_content"])}</td>'
                f'<td class="num">{fmt(row["d_content"], "+d")}</td>'
                f'<td class="num">{fmt(row["o_ms"], ".1f")}</td><td class="num">{fmt(row["j_ms"], ".1f")}</td>'
                f'<td class="num">{fmt(row["ratio"], ".2f")}</td><td class="num">{fmt(row["tf"], ".2f")}</td>'
                f"<td>{pill(s)}</td></tr>"
            )
        m = column_means(r["rows"])
        out.append(
            '<tr class="mean" title="Mean over turns where the value is present (missing turns are skipped per column)">'
            '<td colspan="5">Mean</td>'
            f'<td class="num">{fmt(m["o_in"], ".1f")}</td><td class="num">{fmt(m["j_in"], ".1f")}</td>'
            f'<td class="num">{fmt(m["d_in"], "+.1f")}</td><td class="num">{fmt(m["d_in_pct"], "+.1f")}</td>'
            f'<td class="num">{fmt(m["o_out"], ".1f")}</td><td class="num">{fmt(m["j_out"], ".1f")}</td>'
            f'<td class="num">{fmt(m["d_out"], "+.1f")}</td>'
            f'<td class="num">{fmt(m["o_think"], ".1f")}</td><td class="num">{fmt(m["j_think"], ".1f")}</td>'
            f'<td class="num">{fmt(m["o_content"], ".1f")}</td><td class="num">{fmt(m["j_content"], ".1f")}</td>'
            f'<td class="num">{fmt(m["d_content"], "+.1f")}</td>'
            f'<td class="num">{fmt(m["o_ms"], ".1f")}</td><td class="num">{fmt(m["j_ms"], ".1f")}</td>'
            f'<td class="num">{fmt(m["ratio"], ".2f")}</td><td class="num">{fmt(m["tf"], ".3f")}</td><td></td></tr>'
        )
        out.append("</table></div>")
    out.append("</main></body></html>")
    return "\n".join(out)


def build_reports(sources, recordings_dir, tokenizer):
    """Load every replayed trace, pair it with its recording, and compare them turn by turn."""
    recordings_dir = Path(recordings_dir)
    count_tokens = make_token_counter(tokenizer)
    recordings = {p.name: load_recording(p) for p in sorted(recordings_dir.glob("*.jsonl"))}
    agents = {p.name: recording_agent(p) for p in sorted(recordings_dir.glob("*.jsonl"))}
    reports = []
    for src, trace in sorted(iter_traces(sources), key=lambda st: session_sort_key(st[1])):
        session, jaeger = load_jaeger(trace)
        name, how = pick_recording(session, jaeger, recordings)
        orig, n_failed = recordings[name]
        errored = (session or {}).get("otel.status_code") == "ERROR"
        rows, counts = compare(orig, jaeger, count_tokens, errored)
        trace_id = trace["traceID"]
        reports.append(
            {
                "src": src,
                "url": Path(src).resolve().as_uri(),
                "anchor": f"t-{trace_id}",
                "session": session or {},
                "recording": name,
                "agent": agents[name],
                "how": how,
                "orig": orig,
                "n_failed": n_failed,
                "jaeger": jaeger,
                "rows": rows,
                "counts": counts,
                # An errored session stopped early, so its call sequence isn't comparable to the full recording.
                "wf": None if errored else workflow_fidelity(orig, jaeger),
                "seq_orig": call_sequence(orig),
                "seq_jaeger": call_sequence(jaeger),
                "tf": mean(row["tf"] for row in rows),
                "d_content": mean(row["d_content"] for row in rows),
                "in_ratio": token_ratio(orig, jaeger, "in"),
                "out_ratio": token_ratio(orig, jaeger, "out"),
                "peak_ratio": token_ratio(orig, jaeger, "in", max),
            }
        )

    return reports


def build_recording_reports(recordings_dir, replayed_dir, tokenizer):
    """Compare two recorded runs: each recording with the same-named one in replayed_dir, treated as the replay."""
    count_tokens = make_token_counter(tokenizer)
    reports = []
    for path in sorted(Path(recordings_dir).glob("*.jsonl")):
        other = Path(replayed_dir) / path.name
        if not other.is_file():
            print(f"  skipping {path.name}: not in {replayed_dir}")
            continue
        orig, n_failed = load_recording(path)
        replayed, _ = load_recording(other)
        rows, counts = compare(orig, replayed, count_tokens)
        reports.append(
            {
                "src": str(other),
                "url": other.resolve().as_uri(),
                "anchor": f"t-{path.stem}",
                "session": {"session.id": path.stem, "otel.status_code": "OK"},
                "recording": path.name,
                "agent": recording_agent(path),
                "how": "file name",
                "orig": orig,
                "n_failed": n_failed,
                "jaeger": replayed,
                "rows": rows,
                "counts": counts,
                "wf": workflow_fidelity(orig, replayed),
                "seq_orig": call_sequence(orig),
                "seq_jaeger": call_sequence(replayed),
                "tf": mean(row["tf"] for row in rows),
                "d_content": mean(row["d_content"] for row in rows),
                "in_ratio": token_ratio(orig, replayed, "in"),
                "out_ratio": token_ratio(orig, replayed, "out"),
                "peak_ratio": token_ratio(orig, replayed, "in", max),
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description="Compare recorded LLM calls against replayed Jaeger traces.")
    parser.add_argument(
        "sources", nargs="*", help="Local Jaeger JSON export files (required unless --replayed-recordings is given)"
    )
    parser.add_argument("-o", "--output", default="trace_comparison.html", help="HTML report path")
    parser.add_argument("--tokenizer", default=TOKENIZER, help="Hugging Face tokenizer used to count reasoning tokens")
    parser.add_argument("--recordings", required=True, help="Directory containing .jsonl recording files")
    parser.add_argument(
        "--replayed-recordings",
        help="Directory of .jsonl recordings to compare against --recordings (paired by file name) instead of Jaeger traces",
    )
    args = parser.parse_args()
    if not args.replayed_recordings and not args.sources:
        parser.error("provide at least one Jaeger JSON source, or --replayed-recordings")

    if args.replayed_recordings:
        reports = build_recording_reports(args.recordings, args.replayed_recordings, args.tokenizer)
        replayed = f"recorded calls in {args.replayed_recordings}"
    else:
        reports = build_reports(args.sources, args.recordings, args.tokenizer)
        replayed = "replayed Jaeger traces"
    Path(args.output).write_text(render_html(reports, args.tokenizer, args.recordings, replayed))
    print(f"Wrote {args.output}")
    for r in reports:
        print(
            f"  {r['session'].get('session.id', r['src'])}: {r['recording']}  "
            + ", ".join(f"{k}={v}" for k, v in r["counts"].items())
            + f", WF={fmt(r['wf'], '.3f')}, TF={fmt(r['tf'], '.3f')}, Δcontent={fmt(r['d_content'], '+.1f')}"
            + f", R_in={fmt_ratio(r['in_ratio'])}, R_out={fmt_ratio(r['out_ratio'])}, R_peak={fmt_ratio(r['peak_ratio'])}"
        )
    summary = overall_summary(reports)
    print("Overall:")
    for label, _, render in OVERALL_ITEMS:
        print(f"  {label}: {render(summary)}")


if __name__ == "__main__":
    main()
