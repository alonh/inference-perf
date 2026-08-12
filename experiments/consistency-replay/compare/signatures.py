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

"""The canonical forms that equality and Jaccard are taken over.

These look like parsing — they read tool-call dicts and re-serialize them — but they are
**metric definitions**, which is why they live in the comparison library rather than in
replay_parsing.py. Each one answers a question of the form "are these two the same?", and
each fixes a policy about what a difference is:

  tool_signature       names + argument KEYS      -> same call structure, values ignored
  tool_args_signature  names + argument VALUES    -> same call, canonically compared
  tool_kv_set          {(tool.argname, value)}    -> membership, for a Jaccard
  response_signature   content + tool args        -> "the same response", exactly

Change one of these and every caller's numbers change: response_signature is the single
definition of response identity shared by analyze_consistency's modal counts,
consistency_statistics' Feature, export_viewer_data's cluster coloring,
find_metric_witnesses' exact_match, and compare_profiles' exact_match. Splitting a record
into fields (replay_parsing.py) carries no such policy; this does.

Whitespace policy, applied consistently: prose content is collapsed to single spaces
(collapse_ws), while tool arguments are canonicalized as JSON with sorted keys — and when the
arguments will not parse, whitespace-stripped (strip_ws). Key order and indentation therefore
never fork a group.
"""

import json
from typing import Optional, Tuple

from replay_parsing import call_fields, collapse_ws, strip_ws


def require_parsed(response: dict, what: str = "response") -> dict:
    """Assert `response` came from parse_response; raise a pointed error if it did not.

    The comparison layer used to silently re-parse anything without an `ok` key, which hid
    both the cost (re-parsing the same record on every pairwise comparison) and the bug
    class where a caller passed a raw record and got metrics computed over the wrong shape.
    Parsing now happens once, in the run script; this is the guard that keeps it that way.

    It describes replay_parsing.parse_response's output but lives here because only the
    comparators call it — it exists to protect this library's contract, not that one's.
    """
    if not isinstance(response, dict) or "ok" not in response:
        raise ValueError(
            f"{what} is not a parsed response (no 'ok' key); "
            "call replay_parsing.parse_response(record) first"
        )
    return response


def tool_signature(tool_calls: list) -> Tuple:
    """Comparable signature of tool calls: sequence of (name, sorted arg keys).

    Compares tool names and argument structure, but not argument values.
    """
    sig = []
    for tc in tool_calls:
        name, args_raw = call_fields(tc)
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            keys = tuple(sorted(args.keys())) if isinstance(args, dict) else ("<non-dict-args>",)
        except (json.JSONDecodeError, TypeError):
            keys = ("<unparseable-args>",)
        sig.append((name, keys))
    return tuple(sig)


def tool_args_signature(tool_calls: list) -> Tuple:
    """Full signature including canonical arg VALUES.

    Whitespace-insensitive: valid JSON is re-serialized canonically (so key order and
    inter-token spacing don't matter), and the unparseable fallback is whitespace-collapsed.
    """
    sig = []
    for tc in tool_calls:
        name, args_raw = call_fields(tc)
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            canon = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            canon = strip_ws(str(args_raw))
        sig.append((name, canon))
    return tuple(sig)


def tool_kv_set(tool_calls: list) -> frozenset:
    """Flatten a turn's tool calls to a {(tool.key, canonical-value)} set (Yagubyan AC unit).

    Keys are namespaced by tool name so identical arg names on different tools don't
    collide; values are canonical JSON so key order / spacing can't fork them. This is the
    precomputed unit both compare_tool_arguments and argument_consistency compare.
    """
    kv: set = set()
    for tc in tool_calls:
        name, args_raw = call_fields(tc)
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
