"""Plot replay fidelity (WF/TF) and recorded-vs-replayed input/output/latency from the trace comparison.

Takes the same arguments as trace_comparison.py, except that sources and --recordings are comma-separated lists
paired by position (source i is compared against recordings dir i; a single recordings dir applies to every source).
Each run gets its own bars, side by side, labelled by --labels. Writes two figures:
  <prefix>_fidelity.<ext>  - median WF and TF over sessions, one bar per run,
                             error bars = 25th-75th percentile over sessions
  <prefix>_load.<ext>      - mean input tokens, output tokens and latency per turn, recorded vs. replayed per run,
                             over turns present on both sides, error bars = sample std over those turns
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402

from trace_comparison import TOKENIZER, build_reports  # noqa: E402

RECORDED = "#2a78d6"
REPLAYED = "#eb6834"
RUN_COLORS = ["#2a78d6", "#1f9e89", "#8a5cd1", "#d6a32a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def human(v):
    return f"{v / 1000:.3g}k" if v >= 1000 else f"{v:.3g}"


def bar_with_err(ax, x, vals, color):
    """Median bar with a 25th-75th percentile error bar; returns the top of the error bar."""
    lo, m, hi = np.percentile(vals, [25, 50, 75])
    ax.bar(x, m, width=0.6, color=color, edgecolor="white", linewidth=1)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, linewidth=0.8)
    ax.text(x, hi, f"{human(m)}\n", ha="center", va="bottom", color=INK, fontsize=7, linespacing=0.6)
    return hi


def bar_mean_std(ax, x, vals, color):
    """Mean bar with a sample-std error bar, clipped at 0; returns the top of the error bar."""
    m, sd = np.mean(vals), np.std(vals, ddof=1)
    ax.bar(x, m, width=0.6, color=color, edgecolor="white", linewidth=1)
    ax.errorbar(x, m, yerr=[[min(sd, m)], [sd]], color=INK, capsize=3, linewidth=0.8)
    ax.text(x, m + sd, f"{human(m)}\n", ha="center", va="bottom", color=INK, fontsize=7, linespacing=0.6)
    return m + sd


def plot_fidelity(runs, path):
    """runs is a list of (label, reports); each metric gets one bar per run."""
    fig, ax = plt.subplots(figsize=(3.25, 1.9))
    group = len(runs) + 1
    for g, (key, name) in enumerate((("wf", "Workflow fidelity"), ("tf", "Tool-call fidelity"))):
        for i, (_, reports) in enumerate(runs):
            vals = [r[key] for r in reports if r[key] is not None]
            bar_with_err(ax, g * group + i, vals, RUN_COLORS[i % len(RUN_COLORS)])
            ax.text(g * group + i, 0, f"$n$={len(vals)}", ha="center", va="bottom", color="white", fontsize=6)
        ax.text(
            g * group + (len(runs) - 1) / 2,
            -0.12,
            name,
            ha="center",
            va="top",
            color=INK,
            fontsize=8,
            transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
        )
    ax.set_xticks([])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fidelity (1 = identical)", color=INK_2)
    handles = [
        matplotlib.patches.Patch(color=RUN_COLORS[i % len(RUN_COLORS)], label=label) for i, (label, _) in enumerate(runs)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(runs), frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired(reports, o_col, j_col):
    rows = [row for r in reports for row in r["rows"] if row[o_col] is not None and row[j_col] is not None]
    return [row[o_col] for row in rows], [row[j_col] for row in rows]


def plot_load(runs, path):
    """runs is a list of (label, reports); each panel gets a Rec./Rep. pair per run, left to right."""
    panels = [
        ("o_in", "j_in", "Input tokens", 1),
        ("o_out", "j_out", "Output tokens", 1),
        ("o_ms", "j_ms", "Latency (s)", 1e-3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(3.25 * len(runs), 1.9))
    for ax, (o_col, j_col, title, scale) in zip(axes, panels, strict=True):
        tops, ticks = [], []
        for i, (label, reports) in enumerate(runs):
            rec, rep = paired(reports, o_col, j_col)
            x0 = 3 * i
            tops += [bar_mean_std(ax, x, np.array(v) * scale, c) for x, v, c in ((x0, rec, RECORDED), (x0 + 1, rep, REPLAYED))]
            ticks += [x0, x0 + 1]
            ax.text(
                x0 + 0.5,
                -0.2,
                label.replace(" (", "\n("),
                ha="center",
                va="top",
                color=INK,
                fontsize=7,
                transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
            )
        ax.set_xticks(ticks, ["Rec.", "Rep."] * len(runs))
        ax.set_ylim(0, max(tops) * 1.2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: human(v) if v else "0"))
        ax.set_title(title, color=INK, fontsize=8)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", help="Comma-separated local Jaeger JSON export files")
    parser.add_argument(
        "--recordings",
        required=True,
        help="Comma-separated directories of .jsonl recording files, one per source (or one for all)",
    )
    parser.add_argument("--labels", help="Comma-separated run names, one per source (default: source file names)")
    parser.add_argument("--tokenizer", default=TOKENIZER, help="Hugging Face tokenizer used to count reasoning tokens")
    parser.add_argument("-o", "--output-prefix", default="trace_comparison", help="Prefix for the figure files")
    parser.add_argument("--format", default="png", help="Figure file format, e.g. png or pdf")
    args = parser.parse_args()

    sources = [src for arg in args.sources for src in arg.split(",") if src]
    recordings = [d for d in args.recordings.split(",") if d]
    if len(recordings) == 1:
        recordings *= len(sources)
    if len(recordings) != len(sources):
        parser.error(f"got {len(sources)} sources but {len(recordings)} recordings dirs")

    labels = args.labels.split(",") if args.labels else [Path(src).stem for src in sources]
    if len(labels) != len(sources):
        parser.error(f"got {len(sources)} sources but {len(labels)} labels")

    runs = [
        (label, build_reports([src], rec, args.tokenizer)) for label, src, rec in zip(labels, sources, recordings, strict=True)
    ]
    for name, plot in (("fidelity", plot_fidelity), ("load", plot_load)):
        path = Path(f"{args.output_prefix}_{name}.{args.format}")
        plot(runs, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
