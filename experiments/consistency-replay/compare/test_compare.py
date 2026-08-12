"""Unit tests for the compare module — metrics, canonical units, profile comparison.

parse_response / parse_records are imported from replay_parsing purely as fixtures: these
tests need realistic parsed input, but the parser's own behaviour is covered next door in
test_replay_parsing.py.
"""

import json
import pytest
from replay_parsing import extract_tool_names, parse_response, parse_records
from compare import (
    compare_responses,
    compare_profiles,
    extract_profile,
    normalized_levenshtein,
    fast_content_ratio,
    jaccard,
    compare_tool_calls,
    compare_tool_calls_ordered_dedup,
    collapse_adjacent,
    tool_sequence_lcs,
    tss,
    compare_tool_set_overlap,
    response_signature,
    js_kernel,
    global_alignment_kernel,
    action_histogram,
)


def _call(name, **args):
    """Shorthand: build a tool_call dict with JSON-encoded arguments."""
    return {"function": {"name": name, "arguments": json.dumps(args)}}


class TestStringMetrics:
    """Test low-level string similarity metrics."""

    def test_normalized_levenshtein_identical(self):
        assert normalized_levenshtein("hello", "hello") == 1.0

    def test_normalized_levenshtein_completely_different(self):
        assert normalized_levenshtein("abc", "xyz") == 0.0

    def test_normalized_levenshtein_partial(self):
        sim = normalized_levenshtein("kitten", "sitting")
        assert 0.0 < sim < 1.0

    def test_normalized_levenshtein_empty(self):
        assert normalized_levenshtein("", "") == 1.0
        assert normalized_levenshtein("a", "") == 0.0
        assert normalized_levenshtein("", "b") == 0.0

    def test_jaccard_identical(self):
        assert jaccard("hello world", "hello world") == 1.0

    def test_jaccard_no_overlap(self):
        assert jaccard("abc def", "xyz uvw") == 0.0

    def test_jaccard_partial(self):
        sim = jaccard("hello world foo", "hello world bar")
        assert 0.0 < sim < 1.0
        # 2 words overlap (hello, world) out of 5 unique words total


class TestParsingContract:
    """replay_parsing owns parsing; the comparators refuse to do it for you."""

    RAW = {
        "response": json.dumps({
            "choices": [{"message": {"content": "hi", "tool_calls": []},
                         "finish_reason": "stop"}]
        })
    }

    def test_compare_responses_rejects_raw_record(self):
        parsed = parse_response(self.RAW)
        with pytest.raises(ValueError, match="not a parsed response"):
            compare_responses(self.RAW, parsed)
        with pytest.raises(ValueError, match="not a parsed response"):
            compare_responses(parsed, self.RAW)
        # Parsed on both sides works.
        assert compare_responses(parsed, parsed)["content_levenshtein"] == 1.0

    def test_response_signature_rejects_raw_record(self):
        with pytest.raises(ValueError, match="not a parsed response"):
            response_signature(self.RAW)
        assert response_signature(parse_response(self.RAW)) is not None

    def test_error_message_names_the_offending_side(self):
        parsed = parse_response(self.RAW)
        with pytest.raises(ValueError, match="response_b"):
            compare_responses(parsed, self.RAW)

    def test_parse_records_is_elementwise(self):
        records = [self.RAW, {"error": {"error_type": "timeout"}}]
        parsed = parse_records(records)
        assert [p["ok"] for p in parsed] == [True, False]
        assert parsed == [parse_response(r) for r in records]

    def test_parse_records_empty(self):
        assert parse_records([]) == []


class TestWireShapeReachesTheMetrics:
    """The flat tool-call shape has to survive replay_parsing.call_fields all the way into a
    signature — the one assertion that spans the module boundary. call_fields' own edge cases
    are covered in test_replay_parsing.py."""

    def test_flat_shape_reaches_the_signature(self):
        flat = {"ok": True, "content": "x", "tool_calls": [{"name": "g", "arguments": '{"p": 1}'}]}
        nested = {"ok": True, "content": "x",
                  "tool_calls": [{"function": {"name": "g", "arguments": '{"p": 1}'}}]}
        assert response_signature(flat) == response_signature(nested)
        assert extract_tool_names(flat["tool_calls"]) == ("g",)


class TestToolMetrics:
    """Test tool-related metrics."""

    def test_compare_tool_calls_identical(self):
        calls = [{"function": {"name": "grep", "arguments": '{"pattern": "foo"}'}}]
        sim = compare_tool_calls(calls, calls)
        assert sim == 1.0

    def test_compare_tool_calls_different(self):
        calls_a = [{"function": {"name": "grep", "arguments": '{"pattern": "foo"}'}}]
        calls_b = [{"function": {"name": "grep", "arguments": '{"pattern": "bar"}'}}]
        sim = compare_tool_calls(calls_a, calls_b)
        assert sim == 0.0

    def test_tool_sequence_lcs_identical(self):
        seq_a = ("find", "grep", "sort")
        seq_b = ("find", "grep", "sort")
        sim = tool_sequence_lcs(seq_a, seq_b)
        assert sim == 1.0

    def test_tool_sequence_lcs_partial(self):
        seq_a = ("find", "grep", "sort")
        seq_b = ("find", "sort")  # grep missing
        sim = tool_sequence_lcs(seq_a, seq_b)
        # LCS is (find, sort) = 2 out of max(3, 2) = 3
        assert sim == 2.0 / 3.0

    def test_compare_tool_set_overlap_full(self):
        sim = compare_tool_set_overlap({"find", "grep"}, {"find", "grep"})
        assert sim == 1.0

    def test_compare_tool_set_overlap_partial(self):
        sim = compare_tool_set_overlap({"find", "grep"}, {"grep", "sort"})
        # Intersection: {grep}, Union: {find, grep, sort}
        assert sim == 1.0 / 3.0

    def test_compare_tool_set_overlap_none(self):
        sim = compare_tool_set_overlap({"find"}, {"grep"})
        assert sim == 0.0


class TestCompareResponses:
    """Test Level 1: response comparison."""

    def test_compare_responses_identical(self):
        resp = {
            "ok": True,
            "content": "hello",
            "tool_calls": [],
            "finish_reason": "stop"
        }
        result = compare_responses(resp, resp)
        assert "overall_similarity" not in result  # composite roll-up dropped
        assert result["content_levenshtein"] == 1.0
        assert result["tool_calls_exact"] == 1.0

    def test_compare_responses_different_content(self):
        resp_a = {"ok": True, "content": "hello", "tool_calls": [], "finish_reason": "stop"}
        resp_b = {"ok": True, "content": "goodbye", "tool_calls": [], "finish_reason": "stop"}
        result = compare_responses(resp_a, resp_b)
        assert result["content_levenshtein"] < 1.0

    def test_compare_responses_one_error(self):
        resp_ok = {"ok": True, "content": "test", "tool_calls": [], "finish_reason": "stop"}
        resp_error = {"ok": False, "error": "timeout"}
        result = compare_responses(resp_ok, resp_error)
        # Errored side => every metric 0, and no composite key.
        assert "overall_similarity" not in result
        assert result["content_levenshtein"] == 0.0
        assert result["tool_calls_exact"] == 0.0

    def test_compare_responses_with_tools(self):
        resp_a = {
            "ok": True,
            "content": "",
            "tool_calls": [{"function": {"name": "grep", "arguments": '{"pattern": "test"}'}}],
            "finish_reason": "tool_calls"
        }
        resp_b = {
            "ok": True,
            "content": "",
            "tool_calls": [{"function": {"name": "grep", "arguments": '{"pattern": "test"}'}}],
            "finish_reason": "tool_calls"
        }
        result = compare_responses(resp_a, resp_b)
        assert "overall_similarity" not in result
        assert result["tool_calls_exact"] == 1.0

    def test_compare_responses_different_tools(self):
        resp_a = {
            "ok": True,
            "content": "",
            "tool_calls": [{"function": {"name": "find", "arguments": '{}'}}],
            "finish_reason": "tool_calls"
        }
        resp_b = {
            "ok": True,
            "content": "",
            "tool_calls": [{"function": {"name": "grep", "arguments": '{}'}}],
            "finish_reason": "tool_calls"
        }
        result = compare_responses(resp_a, resp_b)
        assert result["tool_calls_exact"] == 0.0
        assert result["tool_sequence_similarity"] == 0.0


class TestOrderedDedup:
    """Order-preserving, arg-aware tool-call metric that ignores back-to-back repeats."""

    def test_collapse_adjacent_only_consecutive(self):
        # Consecutive dups collapse; a non-adjacent revisit is preserved.
        assert collapse_adjacent(("A", "A", "B")) == ("A", "B")
        assert collapse_adjacent(("A", "B", "A")) == ("A", "B", "A")
        assert collapse_adjacent(("A", "A", "A")) == ("A",)
        assert collapse_adjacent(()) == ()

    def test_adjacent_repeat_is_forgiven(self):
        # Same tool+args emitted twice back-to-back scores 1.0 (a stutter, not a new plan)...
        a = [_call("find", q="x"), _call("find", q="x"), _call("book", id="1")]
        b = [_call("find", q="x"), _call("book", id="1")]
        assert compare_tool_calls_ordered_dedup(a, b) == 1.0
        # ...while exact match punishes the extra call.
        assert compare_tool_calls(a, b) == 0.0

    def test_non_adjacent_revisit_is_kept(self):
        # A, B, A collapses to A, B, A — comparing to A, B is a genuine difference.
        a = [_call("find", q="x"), _call("book", id="1"), _call("find", q="x")]
        b = [_call("find", q="x"), _call("book", id="1")]
        # deduped seqs: (A,B,A) vs (A,B) -> LCS 2 / max 3
        assert compare_tool_calls_ordered_dedup(a, b) == pytest.approx(2 / 3)

    def test_arg_aware_same_name_different_args(self):
        # Same tool name back-to-back but DIFFERENT args must NOT collapse.
        a = [_call("book", id="1"), _call("book", id="2")]
        b = [_call("book", id="1")]
        # tokens (book,{id:1}) != (book,{id:2}) -> not adjacent-equal -> kept -> 1/2
        assert compare_tool_calls_ordered_dedup(a, b) == pytest.approx(0.5)

    def test_identical_is_one_and_empty_is_one(self):
        a = [_call("find", q="x"), _call("book", id="1")]
        assert compare_tool_calls_ordered_dedup(a, a) == 1.0
        assert compare_tool_calls_ordered_dedup([], []) == 1.0

    def test_present_in_compare_responses_both_paths(self):
        ok = {"ok": True, "content": "", "tool_calls": [_call("f", a=1)], "finish_reason": "tool_calls"}
        err = {"ok": False, "error": "boom"}
        assert "tool_calls_ordered_dedup" in compare_responses(ok, ok)
        assert compare_responses(ok, err)["tool_calls_ordered_dedup"] == 0.0


class TestExtractProfile:
    """Test profile extraction."""

    def test_extract_profile_basic(self):
        records = [
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {
                            "content": "searching",
                            "tool_calls": [{"function": {"name": "find", "arguments": "{}"}}]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
            },
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {
                            "content": "filtering",
                            "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
            },
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {
                            "content": "done",
                            "tool_calls": []
                        },
                        "finish_reason": "stop"
                    }]
                })
            }
        ]

        profile = extract_profile(records)
        assert profile["num_requests"] == 3
        assert profile["tool_sequence"] == ("find", "grep")
        assert profile["unique_tools"] == {"find", "grep"}
        assert profile["num_tool_calls"] == 2
        assert profile["num_errors"] == 0

    def test_extract_profile_with_error(self):
        records = [
            {"error": {"error_type": "timeout"}},
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {"content": "ok", "tool_calls": []},
                        "finish_reason": "stop"
                    }]
                })
            }
        ]
        profile = extract_profile(records)
        assert profile["num_requests"] == 2
        assert profile["num_errors"] == 1

    def test_extract_profile_empty(self):
        profile = extract_profile([])
        assert profile["num_requests"] == 0
        assert profile["tool_sequence"] == ()
        assert profile["unique_tools"] == set()


class TestCompareProfiles:
    """Test Level 2: profile comparison."""

    def test_compare_profiles_identical(self):
        records = [
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {"content": "test", "tool_calls": []},
                        "finish_reason": "stop"
                    }]
                })
            }
        ]
        profile = extract_profile(records)
        result = compare_profiles(profile, profile)
        assert "overall_similarity" not in result  # composite roll-up dropped
        assert result["exact_match"] == 1.0

    def test_compare_profiles_different_tool_sequences(self):
        records_a = [
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {
                            "content": "a",
                            "tool_calls": [{"function": {"name": "find", "arguments": "{}"}}]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
            }
        ]
        records_b = [
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {
                            "content": "b",
                            "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
            }
        ]
        profile_a = extract_profile(records_a)
        profile_b = extract_profile(records_b)
        result = compare_profiles(profile_a, profile_b)
        assert result["tool_sequence_similarity"] == 0.0
        assert result["exact_match"] == 0.0

    def test_compare_profiles_different_depths(self):
        records_a = [
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {"content": "a", "tool_calls": []},
                        "finish_reason": "stop"
                    }]
                })
            }
        ]
        records_b = [
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {"content": "a", "tool_calls": []},
                        "finish_reason": "stop"
                    }]
                })
            },
            {
                "response": json.dumps({
                    "choices": [{
                        "message": {"content": "b", "tool_calls": []},
                        "finish_reason": "stop"
                    }]
                })
            }
        ]
        profile_a = extract_profile(records_a)
        profile_b = extract_profile(records_b)
        result = compare_profiles(profile_a, profile_b)
        assert result["session_depth_agreement"] < 1.0
        # 1 vs 2: diff = 1, max = 2, agreement = 1 - (1/2) = 0.5
        assert result["session_depth_agreement"] == 0.5

    def test_ordered_dedup_forgives_consecutive_turns(self):
        # A trace that calls the same tool+args on two consecutive turns should score 1.0
        # against a trace that called it once — a stutter across turns, not a new step.
        def turn(name, **args):
            return {"response": json.dumps({"choices": [{
                "message": {"content": "", "tool_calls": [
                    {"function": {"name": name, "arguments": json.dumps(args)}}]},
                "finish_reason": "tool_calls"}]})}

        prof_stutter = extract_profile([turn("find", q="x"), turn("find", q="x"), turn("book", id="1")])
        prof_clean = extract_profile([turn("find", q="x"), turn("book", id="1")])
        # arg-aware token stream spans turns; adjacent dup collapses -> identical
        assert prof_stutter["tool_call_sequence"] == (
            ("find", '{"q": "x"}'), ("find", '{"q": "x"}'), ("book", '{"id": "1"}'))
        r = compare_profiles(prof_stutter, prof_clean)
        assert r["tool_calls_ordered_dedup"] == 1.0
        # name-only LCS keeps the repeat: LCS(("find","find","book"),("find","book"))=2/3
        assert r["tool_sequence_similarity"] == pytest.approx(2 / 3)

    def test_ordered_dedup_keeps_nonadjacent_revisit(self):
        def turn(name):
            return {"response": json.dumps({"choices": [{
                "message": {"content": "", "tool_calls": [
                    {"function": {"name": name, "arguments": "{}"}}]},
                "finish_reason": "tool_calls"}]})}

        prof_revisit = extract_profile([turn("find"), turn("book"), turn("find")])
        prof_clean = extract_profile([turn("find"), turn("book")])
        r = compare_profiles(prof_revisit, prof_clean)
        # (find,book,find) has no adjacent dups -> preserved -> LCS 2/3, not 1.0
        assert r["tool_calls_ordered_dedup"] == pytest.approx(2 / 3)


class TestPaperPrimitives:
    """TSS (edit-distance) vs LCS, kernels, fast ratio, and the exact-match signature."""

    def test_tss_vs_lcs_diverge_on_reorder(self):
        # (a,b,c) vs (a,c,b): LCS keeps the subsequence (a,b) or (a,c) -> 2/3;
        # TSS counts the substitution -> 1 edit / 3 -> 1 - 1/3? No: 2 swaps => dist 2/3.
        a, b = ("a", "b", "c"), ("a", "c", "b")
        assert tool_sequence_lcs(a, b) == pytest.approx(2.0 / 3.0)
        assert tss(a, b) == pytest.approx(1.0 / 3.0)  # 2 edits over max len 3

    def test_tss_identical_and_empty(self):
        assert tss(("x", "y"), ("x", "y")) == 1.0
        assert tss((), ()) == 1.0
        assert tss(("x",), ()) == 0.0

    def test_fast_ratio_endpoints(self):
        assert fast_content_ratio("hello", "hello") == 1.0
        assert fast_content_ratio("abc", "") == 0.0
        assert 0.0 < fast_content_ratio("kitten", "sitting") < 1.0

    def test_response_signature_content_and_args(self):
        # Same content, different args => different signatures.
        ra = {"ok": True, "content": "x",
              "tool_calls": [{"function": {"name": "g", "arguments": '{"p": 1}'}}]}
        rb = {"ok": True, "content": "x",
              "tool_calls": [{"function": {"name": "g", "arguments": '{"p": 2}'}}]}
        assert response_signature(ra) != response_signature(rb)
        # Whitespace-only content difference => same signature.
        rc = {"ok": True, "content": "a  b", "tool_calls": []}
        rd = {"ok": True, "content": "a b", "tool_calls": []}
        assert response_signature(rc) == response_signature(rd)
        # Errored => None.
        assert response_signature({"ok": False, "error": "x"}) is None

    def test_js_kernel(self):
        assert js_kernel({}, {}) == 1.0  # both no-tool turns
        h = action_histogram(("g", "g", "f"))
        assert js_kernel(h, h) == 1.0
        assert 0.0 < js_kernel(action_histogram(("g",)), action_histogram(("f",))) < 1.0

    def test_global_alignment_kernel(self):
        assert global_alignment_kernel(("a", "b"), ("a", "b")) == 1.0
        assert global_alignment_kernel((), ()) == 1.0
        v = global_alignment_kernel(("a", "b", "c"), ("a", "x", "c"))
        assert 0.0 <= v < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
