#!/usr/bin/env python3
"""Build the self-contained consistency viewer HTML.

Runs the data export, then injects the JSON into viewer_template.html to produce a
single double-clickable HTML file. This is the last step of the experiment pipeline.

Usage:
  build_viewer.py <reports_base_dir> [--analysis analysis.json] [--out consistency_viewer.html]

Example:
  build_viewer.py reports-consistency \
      --analysis reports-consistency/analysis.json \
      --out reports-consistency/consistency_viewer.html
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "viewer_template.html")
EXPORTER = os.path.join(HERE, "export_viewer_data.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir", help="Directory containing run_* subdirectories")
    ap.add_argument("--analysis", default=None, help="analysis.json (for semantic clusters)")
    ap.add_argument("--out", default=None, help="Output HTML path")
    ap.add_argument("--data", default=None, help="Intermediate viewer_data.json path")
    args = ap.parse_args()

    data_path = args.data or os.path.join(args.base_dir, "viewer_data.json")
    out_path = args.out or os.path.join(args.base_dir, "consistency_viewer.html")

    # 1. Export per-group data (reuses analyzer parsing).
    cmd = [sys.executable, EXPORTER, args.base_dir, "--out", data_path]
    if args.analysis:
        cmd += ["--analysis", args.analysis]
    print("→ exporting viewer data (this can take a few minutes for long traces)...")
    rc = subprocess.call(cmd)
    if rc != 0:
        print("export failed", file=sys.stderr)
        return rc

    # 2. Inject JSON into the template.
    with open(TEMPLATE) as f:
        tpl = f.read()
    with open(data_path) as f:
        data = f.read()
    # Escape any "</" so the embedded JSON can't terminate the <script> tag early.
    # JSON.parse (and Python json) both read "\/" as "/", so this is lossless.
    data = data.replace("</", "<\\/")
    if "__DATA__" not in tpl:
        print("template missing __DATA__ placeholder", file=sys.stderr)
        return 1
    html = tpl.replace("__DATA__", data)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"✓ wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
    print(f"  open it: open {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
