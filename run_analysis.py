"""
run_analysis.py — Analysis orchestrator. FIXED — agent cannot edit.

Loads data and runs the agent's analysis code. Output goes to findings.md.
"""

import sys
import traceback

from prepare import load_data, load_feature_names, get_expanding_windows


def main():
    # Import agent-editable module (fresh import each run)
    import importlib
    import analysis
    importlib.reload(analysis)

    print("Loading data...")
    df = load_data()
    base_features = load_feature_names()
    windows = get_expanding_windows(df)
    print(f"Loaded {len(df)} rows, {len(base_features)} base features, {len(windows)} windows")

    print("Running analysis...")
    analysis.run_analysis(df, base_features, windows)
    print("Analysis complete. Results written to findings.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
