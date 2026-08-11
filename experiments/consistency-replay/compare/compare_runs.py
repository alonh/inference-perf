#!/usr/bin/env python3
"""Compare two trace runs using the compare library.

This is the run script: it owns the whole parsing process (load_records -> extract_profile /
parse_records, all from compare.parsing) and hands only comparison-ready inputs to
compare_profiles / compare_responses, which do no parsing of their own.

Usage:
    python compare_runs.py <path_to_run_1> <path_to_run_2>

Example:
    python compare_runs.py \
        reports-consistency/tau2_airline/qwen.../20260803-004827/run_1/per_request_lifecycle_metrics.json \
        reports-consistency/tau2_airline/qwen.../20260803-004827/run_2/per_request_lifecycle_metrics.json
"""

import sys
import argparse
from pathlib import Path

# Put the directory CONTAINING the compare package on the path (this file lives inside
# the package, so its own parent is the package dir, not the importable root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare import (
    # Parsing (this script is the only place the library parses).
    load_records,
    extract_profile,
    response_signature,
    extract_tool_names,
    # Comparison.
    compare_profiles,
    compare_responses,
)


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"{text:^70s}")
    print(f"{'=' * 70}")


def compare_runs(run_1_path, run_2_path, verbose=False):
    """Compare two trace runs."""
    
    print(f"Loading run 1: {run_1_path}")
    records_1 = load_records(run_1_path)
    print(f"  ✓ Loaded {len(records_1)} records")
    
    print(f"Loading run 2: {run_2_path}")
    records_2 = load_records(run_2_path)
    print(f"  ✓ Loaded {len(records_2)} records")
    
    # Parse: this is the one parsing step. extract_profile parses every record once and
    # stores the results in profile["responses"], so every comparison below (including the
    # verbose per-turn pass) reads parsed data rather than re-parsing.
    print("\nExtracting profiles...")
    profile_1 = extract_profile(records_1)
    profile_2 = extract_profile(records_2)
    
    # Print profile info
    print_header("PROFILE INFORMATION")
    print(f"\nRun 1:")
    print(f"  • Requests: {profile_1['num_requests']}")
    print(f"  • Tool calls: {profile_1['num_tool_calls']}")
    print(f"  • Unique tools: {len(profile_1['unique_tools'])}")
    print(f"  • Errors: {profile_1['num_errors']}")
    
    print(f"\nRun 2:")
    print(f"  • Requests: {profile_2['num_requests']}")
    print(f"  • Tool calls: {profile_2['num_tool_calls']}")
    print(f"  • Unique tools: {len(profile_2['unique_tools'])}")
    print(f"  • Errors: {profile_2['num_errors']}")
    
    # Tool set analysis
    tools_1 = profile_1['unique_tools']
    tools_2 = profile_2['unique_tools']
    common = tools_1 & tools_2
    only_in_1 = tools_1 - tools_2
    only_in_2 = tools_2 - tools_1
    
    print(f"\nTools in common: {len(common)}")
    if only_in_1:
        print(f"  Only in run 1: {only_in_1}")
    if only_in_2:
        print(f"  Only in run 2: {only_in_2}")
    
    # Compare profiles
    print_header("COMPARISON RESULTS")
    
    result = compare_profiles(profile_1, profile_2)
    
    print(f"\nSimilarity Metrics:")
    for metric, score in sorted(result.items()):
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {metric:30s} {score:6.3f} [{bar}]")

    print(f"\n{'=' * 70}")
    # No composite roll-up: for a single grounded consistency figure (theta + CI), run
    # consistency_statistics.py over the run set. Here we report the components.
    print("For a single grounded consistency number (U-statistic theta + CI),")
    print("run: consistency_statistics.py --condition <name>=<run_base_dir>")
    print(f"{'=' * 70}")

    # Interpretation
    print("\nInterpretation:")
    if result['exact_match'] == 1.0:
        print("  ✓ Perfectly identical runs!")
    if result['tool_sequence_similarity'] == 1.0:
        print("  ✓ Tool sequences are identical")
    else:
        print(f"  ⚠ Tool sequences differ: {result['tool_sequence_similarity']:.1%} match")

    if result['session_depth_agreement'] == 1.0:
        print("  ✓ Same number of turns")
    else:
        print(f"  ⚠ Different depths: {result['session_depth_agreement']:.1%} agreement")

    print(f"  Mean response similarity: {result['response_similarity']:.1%}")

    # Verbose: per-turn analysis. "Identical" uses the exact-match signature (content +
    # canonical tool args); divergence severity is reported via content Levenshtein.
    if verbose:
        print_header("PER-TURN ANALYSIS")

        # Already parsed by extract_profile above — reuse, don't re-parse.
        responses_1 = profile_1["responses"]
        responses_2 = profile_2["responses"]

        divergences = 0
        identical = 0
        for i in range(min(len(responses_1), len(responses_2))):
            sig_1 = response_signature(responses_1[i])
            sig_2 = response_signature(responses_2[i])
            if sig_1 is not None and sig_1 == sig_2:
                identical += 1
                continue

            divergences += 1
            if divergences <= 10:  # Show first 10 divergences
                comparison = compare_responses(responses_1[i], responses_2[i])
                lev = comparison['content_levenshtein']
                tools_1 = extract_tool_names(responses_1[i].get('tool_calls', []))
                tools_2 = extract_tool_names(responses_2[i].get('tool_calls', []))
                print(f"\nTurn {i+1}: content Levenshtein {lev:.1%}")
                print(f"  Run 1: {tools_1 if tools_1 else '(no tools)'}")
                print(f"  Run 2: {tools_2 if tools_2 else '(no tools)'}")

        print(f"\nSummary:")
        print(f"  • Total turns: {len(responses_1)}")
        print(f"  • Identical turns (exact signature): {identical}")
        print(f"  • Divergent turns: {divergences}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare two trace runs using the compare library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("run_1", help="Path to first run's per_request_lifecycle_metrics.json")
    parser.add_argument("run_2", help="Path to second run's per_request_lifecycle_metrics.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-turn analysis")
    
    args = parser.parse_args()
    
    try:
        result = compare_runs(args.run_1, args.run_2, verbose=args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
