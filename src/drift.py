"""
Data Drift Detection with Evidently
====================================

Compares a *reference* dataset (pre-SMOTE training data) against a *current*
dataset (test data or fresh production data) to detect distribution shifts
that could degrade model performance.

The script produces:
  * An interactive HTML report (``data_drift_report.html``)
  * A machine-readable JSON summary (``drift_summary.json``)
  * A human-readable console summary
  * Exit code 1 when overall drift is detected (for CI/CD gating)

Usage::

    python src/drift.py \\
        --reference data/processed/X_train_ref.csv \\
        --current data/processed/X_test.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Evidently 0.7.x imports (API changed significantly from 0.4.x)
from evidently import Report
from evidently.presets import DataDriftPreset

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("drift")


# Data loading


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV dataset and log basic statistics.

    Parameters
    ----------
    path : Path
        Path to a CSV file.
    label : str
        Human-readable name for log messages (e.g. "reference", "current").

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label.capitalize()} dataset not found: {path}")
    logger.info("Loading %s data from %s", label, path)
    df = pd.read_csv(path)
    logger.info("%s data: %d rows × %d columns", label.capitalize(), *df.shape)
    return df


# Drift analysis


def run_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> Any:
    """Run Evidently's ``DataDriftPreset`` report.

    The preset applies a battery of per-feature statistical tests (e.g.
    Kolmogorov-Smirnov for numerical features, chi-squared for categorical)
    and aggregates them into a dataset-level drift flag.

    Parameters
    ----------
    reference : pd.DataFrame
        Baseline distribution (typically pre-SMOTE training data).
    current : pd.DataFrame
        New / production distribution to compare.

    Returns
    -------
    Snapshot
        Executed Evidently snapshot object (Evidently 0.7.x).
    """
    logger.info("Running Evidently DataDriftPreset analysis …")
    report = Report([DataDriftPreset()])
    # In Evidently 0.7.x, run() returns a Snapshot object
    snapshot = report.run(reference_data=reference, current_data=current)
    logger.info("Drift analysis complete")
    return snapshot


def extract_drift_summary(snapshot: Any) -> dict[str, Any]:
    """Parse the Evidently snapshot into a concise, serialisable summary.

    Parameters
    ----------
    snapshot : Snapshot
        An already-executed Evidently snapshot.

    Returns
    -------
    dict
        Keys: ``overall_drift``, ``drifted_features``, ``drift_scores``,
        ``num_features``, ``num_drifted``, ``share_drifted``, ``timestamp``.
    """
    report_dict = snapshot.dump_dict()
    metric_results = report_dict.get("metric_results", {})

    # Parse through the metric results to find drift information
    num_drifted = 0
    num_features = 0
    share_drifted = 0.0
    drifted_features: list[str] = []
    drift_scores: dict[str, float] = {}

    for _key, result in metric_results.items():
        result.get("display_name", "")
        metric_info = result.get("metric_value_location", {}).get("metric", {})
        metric_params = metric_info.get("params", {})
        metric_type = metric_params.get("type", "")

        # Extract the counter/value from the widget data
        widgets = result.get("widget", [])

        if "DriftedColumnsCount" in metric_type:
            for w in widgets:
                params = w.get("params", {})
                counters = params.get("counters", [])
                for c in counters:
                    val = c.get("value", "")
                    try:
                        num_drifted = int(val)
                    except (ValueError, TypeError):
                        pass

        elif "TotalColumnsCount" in metric_type or "NumberOfColumns" in metric_type:
            for w in widgets:
                params = w.get("params", {})
                counters = params.get("counters", [])
                for c in counters:
                    val = c.get("value", "")
                    try:
                        num_features = int(val)
                    except (ValueError, TypeError):
                        pass

        elif "ColumnDriftMetric" in metric_type:
            col_name = metric_params.get("column_name", "")
            if col_name:
                # Check if this column drifted from the widget data
                for w in widgets:
                    params = w.get("params", {})
                    counters = params.get("counters", [])
                    for c in counters:
                        val = c.get("value", "")
                        label = c.get("label", "")
                        if "score" in label.lower() or "p-value" in label.lower():
                            try:
                                score = float(val)
                                drift_scores[col_name] = round(score, 6)
                            except (ValueError, TypeError):
                                pass
                        elif "drift" in label.lower() and val in ("True", "true", True):
                            drifted_features.append(col_name)

    if num_features > 0:
        share_drifted = num_drifted / num_features
    overall_drift = num_drifted > (num_features * 0.5) if num_features > 0 else False

    summary: dict[str, Any] = {
        "overall_drift": overall_drift,
        "drifted_features": sorted(set(drifted_features)),
        "drift_scores": dict(sorted(drift_scores.items())),
        "num_features": num_features,
        "num_drifted": num_drifted,
        "share_drifted": round(share_drifted, 4),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    return summary


# Output helpers


def save_html_report(snapshot: Any, path: Path) -> None:
    """Save the interactive Evidently HTML report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    logger.info("HTML drift report saved to %s", path)


def save_json_summary(summary: dict[str, Any], path: Path) -> None:
    """Persist the JSON drift summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("JSON drift summary saved to %s", path)


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable drift summary to the console."""
    print("\n" + "=" * 60)
    print("DATA DRIFT SUMMARY")
    print("=" * 60)
    status = "⚠  DRIFT DETECTED" if summary["overall_drift"] else "✓  No drift detected"
    print(f"  Status           : {status}")
    print(f"  Features checked : {summary['num_features']}")
    print(f"  Features drifted : {summary['num_drifted']} ({summary['share_drifted']:.2%})")

    if summary["drifted_features"]:
        print(f"  Drifted features : {', '.join(summary['drifted_features'])}")
    print(f"  Timestamp        : {summary['timestamp']}")
    print()

    # Show per-feature drift scores sorted by magnitude (descending)
    if summary["drift_scores"]:
        print("  Per-feature drift scores (top 10):")
        sorted_scores = sorted(summary["drift_scores"].items(), key=lambda kv: kv[1], reverse=True)
        for name, score in sorted_scores[:10]:
            marker = " *" if name in summary["drifted_features"] else ""
            print(f"    {name:<30s}  {score:.6f}{marker}")

        if len(sorted_scores) > 10:
            print(f"    … and {len(sorted_scores) - 10} more features")
    print("=" * 60 + "\n")


# CLI entry-point


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect data drift between reference and current datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/processed/X_train_ref.csv"),
        help="Path to the reference (baseline) CSV dataset.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("data/processed/X_test.csv"),
        help="Path to the current (production / test) CSV dataset.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where reports are written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the end-to-end drift detection pipeline."""
    args = parse_args(argv)

    try:
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        reference = load_dataset(args.reference, "reference")
        current = load_dataset(args.current, "current")

        # Ensure both datasets share the same columns.  Mismatched schemas
        # would produce misleading drift results.
        common_cols = sorted(set(reference.columns) & set(current.columns))
        if set(common_cols) != set(reference.columns):
            dropped = set(reference.columns) - set(common_cols)
            logger.warning("Columns in reference but not current (dropped): %s", dropped)
        if set(common_cols) != set(current.columns):
            extra = set(current.columns) - set(common_cols)
            logger.warning("Columns in current but not reference (dropped): %s", extra)
        reference = reference[common_cols]
        current = current[common_cols]

        # Run drift analysis
        snapshot = run_drift_analysis(reference, current)

        # Save outputs
        save_html_report(snapshot, args.reports_dir / "data_drift_report.html")

        summary = extract_drift_summary(snapshot)
        save_json_summary(summary, args.reports_dir / "drift_summary.json")
        print_summary(summary)

        # Exit code for CI/CD
        # Non-zero exit signals that a human (or automated retrain) should
        # investigate the distributional shift.
        if summary["overall_drift"]:
            logger.warning("Overall drift detected — exiting with code 1")
            sys.exit(1)

        logger.info("Drift check complete ✓")

    except SystemExit:
        raise  # let sys.exit(1) propagate
    except Exception:
        logger.exception("Drift detection failed")
        sys.exit(2)  # code 2 = unexpected error (distinct from drift=1)


if __name__ == "__main__":
    main()
