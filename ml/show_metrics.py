"""
show_metrics.py
---------------
Print a summary table of all logged training runs from metrics_log.json.
"""

import json
import os
import sys

METRICS_LOG = os.path.join(os.path.dirname(__file__), "data", "metrics_log.json")


def main():
    if not os.path.exists(METRICS_LOG):
        print("No metrics log found. Run a training script first.")
        sys.exit(0)

    with open(METRICS_LOG) as f:
        log = json.load(f)

    if not log:
        print("Metrics log is empty.")
        sys.exit(0)

    header = f"{'#':<4} {'Timestamp':<22} {'Pipeline':<8} {'Features':<10} {'Windows':<9} {'Patients':<10} {'Accuracy':<10} {'ROC-AUC':<9} {'Mean F1'}"
    print("=" * len(header))
    print("  Smart Stethoscope — Model Metrics History")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, e in enumerate(log, 1):
        threshold = f"  (thr={e['threshold']})" if "threshold" in e else ""
        print(
            f"{i:<4} {e['timestamp']:<22} {e['pipeline']:<8} "
            f"{e['n_features']:<10} {e['n_windows']:<9} {e['n_patients']:<10} "
            f"{e['accuracy']:<10.1%} {e['roc_auc']:<9.3f} {e['mean_f1']:.3f}"
            f"{threshold}"
        )

    print("=" * len(header))
    print(f"Total runs: {len(log)}")


if __name__ == "__main__":
    main()
