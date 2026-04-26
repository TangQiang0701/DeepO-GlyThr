from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepoglythr import build_summary_dataframe, explain_window, parse_input_text, predict_windows


def main():
    parser = argparse.ArgumentParser(description="DeepO-GlyThr CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="FASTA / TXT input file")
    group.add_argument("--sequence", help="Single raw sequence")
    parser.add_argument("--output-csv", default="DeepO-GlyThr-results.csv", help="Summary CSV output")
    parser.add_argument("--output-json", default="DeepO-GlyThr-details.json", help="Per-window explanation JSON output")
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = args.sequence

    windows = parse_input_text(text)
    rows = predict_windows(windows)
    summary_df = build_summary_dataframe(rows)
    summary_df.to_csv(args.output_csv, index=False)

    details = []
    for row in rows:
        details.append(
            {
                "summary": row,
                "explanation": explain_window(row),
            }
        )
    Path(args.output_json).write_text(json.dumps(details, indent=2), encoding="utf-8")

    print(f"Saved summary CSV to {args.output_csv}")
    print(f"Saved detailed JSON to {args.output_json}")


if __name__ == "__main__":
    main()
