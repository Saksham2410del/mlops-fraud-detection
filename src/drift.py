import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def load_data(reference_path, current_path):
    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)
    return ref_df, curr_df


def run_drift_report(ref_df, curr_df):
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=curr_df)
    return snapshot


def get_drift_summary(snapshot):
    result = snapshot.dump_dict() if hasattr(snapshot, "dump_dict") else {}

    num_drifted = 0
    share_drifted = 0.0
    drifted_features = []

    metric_results = result.get("metric_results", {})
    for res in metric_results.values():
        display = res.get("display_name", "")
        if "Count of Drifted Columns" in display:
            count_info = res.get("count", {})
            if isinstance(count_info, dict) and "value" in count_info:
                num_drifted = int(count_info["value"])
            share_info = res.get("share", {})
            if isinstance(share_info, dict) and "value" in share_info:
                share_drifted = float(share_info["value"])
        elif display.startswith("Value drift for "):
            col_name = display.replace("Value drift for ", "").strip()
            val = res.get("value")
            metric_params = res.get("metric_value_location", {}).get("metric", {}).get("params", {})
            thresh = metric_params.get("threshold", 0.05)
            if val is not None and val < thresh:
                drifted_features.append(col_name)

    drift_detected = share_drifted >= 0.5

    return {
        "drift_detected": bool(drift_detected),
        "num_drifted": num_drifted,
        "share_drifted": float(share_drifted),
        "drifted_features": drifted_features,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Data drift detection with Evidently")
    parser.add_argument("--reference", default="data/processed/X_train_ref.csv", help="Path to reference data")
    parser.add_argument("--current", default="data/processed/X_test.csv", help="Path to current data")
    return parser.parse_args()


def main():
    args = parse_args()

    ref_path = Path(args.reference)
    curr_path = Path(args.current)

    if not ref_path.exists():
        print(f"Error: Reference file not found at {ref_path}")
        sys.exit(2)
    if not curr_path.exists():
        print(f"Error: Current file not found at {curr_path}")
        sys.exit(2)

    print(f"Loading reference: {ref_path}")
    print(f"Loading current: {curr_path}")
    ref_df, curr_df = load_data(ref_path, curr_path)

    print("Running drift analysis...")
    snapshot = run_drift_report(ref_df, curr_df)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    html_path = reports_dir / "data_drift_report.html"
    json_path = reports_dir / "drift_summary.json"

    snapshot.save_html(str(html_path))
    print(f"HTML report saved to {html_path}")

    summary = get_drift_summary(snapshot)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to {json_path}")

    print("\n" + "=" * 40)
    print("DRIFT SUMMARY")
    print("=" * 40)
    print(f"Drift detected: {summary['drift_detected']}")
    print(f"Share of drifted features: {summary['share_drifted']:.2%}")
    if summary["drifted_features"]:
        print(f"Drifted features: {', '.join(summary['drifted_features'])}")
    else:
        print("No drifted features found.")
    print("=" * 40)

    if summary["drift_detected"]:
        print("\nDrift detected! Exiting with code 1.")
        sys.exit(1)

    print("\nNo drift detected.")


if __name__ == "__main__":
    main()
