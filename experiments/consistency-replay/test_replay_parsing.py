"""Unit tests for replay_parsing — the general layer: files, raw records, grouping keys.

Nothing here compares anything. Tests that assert a *metric* (what counts as "the same
response") live in compare/test_compare.py, including the one cross-boundary case where a
flat tool-call shape has to survive all the way into a signature.
"""

import json
import pytest
from replay_parsing import (
    call_fields,
    collapse_ws,
    event_key,
    extract_tool_names,
    load_records,
    load_run,
    parse_records,
    parse_response,
    request_key,
    session_key,
    strip_ws,
)


class TestWhitespaceNormalization:
    """collapse_ws is for prose (keeps word boundaries); strip_ws is for JSON (drops them)."""

    def test_collapse_ws_folds_runs_and_trims(self):
        assert collapse_ws("  a \n\t b  ") == "a b"

    def test_collapse_ws_keeps_words_separate(self):
        # The whole point of collapse-vs-strip: "a b" must not become "ab".
        assert collapse_ws("a    b") == "a b"

    def test_collapse_ws_empty_and_none(self):
        assert collapse_ws("") == ""
        assert collapse_ws(None) == ""

    def test_strip_ws_removes_everything(self):
        assert strip_ws('{ "content": 1 }') == '{"content":1}'

    def test_strip_ws_joins_words(self):
        # Deliberate: on structural text there are no words to keep apart.
        assert strip_ws("a b") == "ab"

    def test_strip_ws_empty_and_none(self):
        assert strip_ws("") == ""
        assert strip_ws(None) == ""


class TestParseResponse:
    """Test response parsing."""

    def test_parse_response_ok(self):
        record = {
            "response": json.dumps({
                "choices": [{
                    "message": {
                        "content": "test",
                        "tool_calls": []
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"completion_tokens": 10}
            })
        }
        parsed = parse_response(record)
        assert parsed["ok"] is True
        assert parsed["content"] == "test"
        assert parsed["completion_tokens"] == 10

    def test_parse_response_error(self):
        record = {"error": {"error_type": "timeout"}}
        parsed = parse_response(record)
        assert parsed["ok"] is False
        assert parsed["error"] == "timeout"

    def test_parse_response_empty_response(self):
        record = {"response": None}
        parsed = parse_response(record)
        assert parsed["ok"] is False
        assert parsed["error"] == "empty_response"

    def test_timing_survives_an_error(self):
        # start/end live on the record, not the body: an errored request still took time.
        parsed = parse_response(
            {"error": {"error_type": "timeout"}, "start_time": 1.0, "end_time": 3.5}
        )
        assert (parsed["start_time"], parsed["end_time"]) == (1.0, 3.5)

    def test_ok_but_no_output(self):
        # A length-truncated reasoning turn: well-formed, but nothing usable came out.
        record = {"response": json.dumps({
            "choices": [{"message": {"content": "", "reasoning_content": "thinking..."},
                         "finish_reason": "length"}]
        })}
        parsed = parse_response(record)
        assert parsed["ok"] is True
        assert parsed["has_output"] is False
        assert parsed["reasoning"] == "thinking..."

    def test_reasoning_reads_the_wire_spelling(self):
        # Servers (vLLM/RITS) return `reasoning`; `reasoning_content` is only the alias the
        # harness writes when it re-serializes. Reading the alias alone made `reasoning` ""
        # for every recorded response in every corpus, so cover the wire spelling too.
        record = {"response": json.dumps({
            "choices": [{"message": {"content": "hi", "reasoning": "step 1..."},
                         "finish_reason": "stop"}]
        })}
        assert parse_response(record)["reasoning"] == "step 1..."

    def test_reasoning_prefers_the_wire_spelling(self):
        # When a message carries both (apis/chat.py sets both), they agree — but pin the
        # precedence so the two readers can't drift apart.
        record = {"response": json.dumps({
            "choices": [{"message": {"content": "", "reasoning": "wire",
                                     "reasoning_content": "alias"}}]
        })}
        assert parse_response(record)["reasoning"] == "wire"

    def test_parse_records_is_elementwise(self):
        records = [{"response": None}, {"error": {"error_type": "timeout"}}]
        assert parse_records(records) == [parse_response(r) for r in records]

    def test_parse_records_empty(self):
        assert parse_records([]) == []


class TestCallFields:
    """Both tool-call wire shapes are readable; junk degrades to (None, None)."""

    def test_nested_function_shape(self):
        assert call_fields({"function": {"name": "grep", "arguments": "{}"}}) == ("grep", "{}")

    def test_flat_shape(self):
        # export_viewer_data's flattened summaries carry name/arguments at top level.
        assert call_fields({"name": "grep", "arguments": "{}"}) == ("grep", "{}")

    def test_function_block_wins_over_top_level(self):
        tc = {"function": {"name": "inner", "arguments": "{}"}, "name": "outer"}
        assert call_fields(tc) == ("inner", "{}")

    def test_junk(self):
        assert call_fields(None) == (None, None)
        assert call_fields("grep") == (None, None)
        assert call_fields({}) == (None, None)
        assert call_fields({"function": "not-a-dict"}) == (None, None)
        assert call_fields({"id": "call_1"}) == (None, None)  # no name/arguments => not flat


class TestExtractToolNames:
    """Ordered names off a tool_calls list; unnamed entries are dropped, not blanked."""

    def test_extract_tool_names(self):
        tool_calls = [
            {"function": {"name": "find", "arguments": "{}"}},
            {"function": {"name": "grep", "arguments": "{}"}},
        ]
        names = extract_tool_names(tool_calls)
        assert names == ("find", "grep")

    def test_extract_tool_names_empty(self):
        names = extract_tool_names([])
        assert names == ()

    def test_unnamed_entries_are_skipped(self):
        assert extract_tool_names([{"id": "call_1"}, {"name": "grep"}]) == ("grep",)


class TestRecordIdentity:
    """The grouping keys: request_key hashes input, session_key names the trace, event_key
    names the call position within it."""

    @staticmethod
    def _rec(**kw):
        return kw

    def test_request_key_is_formatting_insensitive(self):
        # Same payload, different key order and spacing -> same group.
        a = self._rec(request='{"model": "m", "messages": []}')
        b = self._rec(request='{"messages":[],"model":"m"}')
        assert request_key(a) == request_key(b)

    def test_request_key_separates_different_inputs(self):
        a = self._rec(request='{"messages": [{"content": "x"}]}')
        b = self._rec(request='{"messages": [{"content": "y"}]}')
        assert request_key(a) != request_key(b)

    def test_request_key_no_request(self):
        assert request_key({}) == "no-request"

    def test_session_key_prefers_recorded_session_id(self):
        assert session_key(self._rec(session_id="trace3_abc")) == "trace3_abc"

    def test_session_key_reads_the_info_block(self):
        assert session_key(self._rec(info={"session_id": "s9"})) == "s9"

    def test_session_key_falls_back_to_the_first_message(self):
        # Older metrics with no session_id: the system prompt anchors the trace.
        a = self._rec(request='{"messages": [{"content": "system prompt A"}]}')
        b = self._rec(request='{"messages": [{"content": "system prompt A"}, {"content": "t2"}]}')
        c = self._rec(request='{"messages": [{"content": "system prompt B"}]}')
        assert session_key(a) == session_key(b) != session_key(c)
        assert session_key(a).startswith("trace_")

    def test_event_key_prefers_recorded_event_id(self):
        assert event_key(self._rec(event_id="e7", session_id="s1")) == "e7"

    def test_event_key_falls_back_to_session_plus_request(self):
        # Same trace, different call position -> different key.
        a = self._rec(session_id="s1", request='{"messages": [{"content": "x"}]}')
        b = self._rec(session_id="s1", request='{"messages": [{"content": "y"}]}')
        assert event_key(a) != event_key(b)
        assert event_key(a).startswith("s1::")

    def test_event_key_groups_identical_input_across_runs(self):
        # The property the whole experiment rests on: identical input at the same position in
        # the same trace lands in one group even with no event_id recorded.
        rec = self._rec(session_id="s1", request='{"messages": [{"content": "x"}]}')
        assert event_key(rec) == event_key(dict(rec))


class TestLoadRecords:
    """The three on-disk shapes load_records accepts, plus the one it rejects."""

    REC = {"event_id": "e1", "response": None}

    def _write(self, tmp_path, obj, name="metrics.json"):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return str(p)

    def test_bare_list(self, tmp_path):
        assert load_records(self._write(tmp_path, [self.REC])) == [self.REC]

    def test_contents_key(self, tmp_path):
        assert load_records(self._write(tmp_path, {"contents": [self.REC]})) == [self.REC]

    def test_records_key(self, tmp_path):
        assert load_records(self._write(tmp_path, {"records": [self.REC]})) == [self.REC]

    def test_unknown_dict_raises(self, tmp_path):
        path = self._write(tmp_path, {"something_else": [self.REC]})
        with pytest.raises(ValueError, match="Unexpected data format"):
            load_records(path)


class TestLoadRun:
    """load_run is the tolerant variant: a sweep over many run dirs must not die on one."""

    def test_reads_the_expected_filename(self, tmp_path):
        (tmp_path / "per_request_lifecycle_metrics.json").write_text(json.dumps([{"a": 1}]))
        assert load_run(str(tmp_path)) == [{"a": 1}]

    def test_missing_file_yields_empty(self, tmp_path):
        assert load_run(str(tmp_path)) == []

    def test_bad_shape_yields_empty_instead_of_raising(self, tmp_path):
        (tmp_path / "per_request_lifecycle_metrics.json").write_text(json.dumps({"nope": 1}))
        assert load_run(str(tmp_path)) == []
