"""
CLI: generate an Evidently drift report comparing training data vs live predictions.

Exit codes:
  0 — no drift detected
  1 — error (missing data, import failure, etc.)
  2 — drift detected (triggers GitHub issue creation in CI)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

from src.monitoring.drift import generate_drift_report, PREDICTIONS_LOG

REFERENCE_PATH = Path("data/raw/voters.parquet")
OUTPUT_PATH = Path("reports/drift_report.html")


def _make_synthetic_current() -> None:
    """Write synthetic current data to PREDICTIONS_LOG if the log is missing."""
    from src.data.generate_voters import generate_voters

    print("No prediction log found — generating synthetic current data ...")
    synthetic = generate_voters(n_voters=500, random_seed=99)
    parties = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"]
    rng = np.random.default_rng(99)
    synthetic["prediction"] = rng.choice(parties, size=len(synthetic))
    synthetic["probability"] = rng.uniform(0.3, 0.9, size=len(synthetic))

    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LOG, "w", encoding="utf-8") as f:
        for _, row in synthetic.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    print(f"  Synthetic log written to {PREDICTIONS_LOG} ({len(synthetic)} rows)")


def main() -> None:
    if not REFERENCE_PATH.exists():
        print(
            f"Error: reference data not found at {REFERENCE_PATH}.\n"
            "Run: python scripts/generate_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    if not PREDICTIONS_LOG.exists() or PREDICTIONS_LOG.stat().st_size == 0:
        _make_synthetic_current()

    print(f"Generating drift report ...")
    print(f"  reference : {REFERENCE_PATH}")
    print(f"  current   : {PREDICTIONS_LOG}")
    print(f"  output    : {OUTPUT_PATH}")

    result = generate_drift_report(REFERENCE_PATH, PREDICTIONS_LOG, OUTPUT_PATH)

    print(f"\nDrift report summary")
    print(f"  Features drifted : {result['n_features_drifted']} / {result['total_features']}")
    print(f"  Drift detected   : {result['drift_detected']}")
    print(f"  Report saved to  : {result['report_path']}")

    if result["drift_detected"]:
        print("\nWARNING: Significant data drift detected — consider retraining.")
        sys.exit(2)
    else:
        print("\nNo significant drift detected.")


if __name__ == "__main__":
    main()
