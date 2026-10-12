"""
Standalone Biometric Evaluation Script for Offline Research Paper Metrics.

This script reads real biometric recognition logs from the SQLite database
(`recognition_logs` table) and joins them with human-annotated ground-truth labels
from `ground_truth.csv`.

It computes:
- Accuracy
- False Accept Rate (FAR)
- False Reject Rate (FRR)
- Equal Error Rate (EER) across a sweep of distance thresholds
- Breakdown of Recognition Rate & Mean Quality score by degradation condition
- Generates an ROC-style curve saved as `evaluation_roc_curve.png`.

Note:
This script ONLY operates on real logged data and human ground-truth labels.
It does NOT generate synthetic or simulated data.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI compatibility
import matplotlib.pyplot as plt
import database
import config

GROUND_TRUTH_FILE = os.path.join(config.BASE_DIR, "ground_truth.csv")
OUTPUT_PLOT_FILE = os.path.join(config.BASE_DIR, "evaluation_roc_curve.png")

def generate_ground_truth_template():
    """Generates ground_truth.csv template prepopulated from existing recognition logs."""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, session_id, timestamp, matched_name, computed_distance, decision, liveness_passed FROM recognition_logs ORDER BY id ASC;")
    logs = cursor.fetchall()
    conn.close()

    if not logs:
        print("No recognition logs found in database. Run real webcam recognition sessions first.")
        return False

    with open(GROUND_TRUTH_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["log_id", "session_id", "timestamp", "matched_name", "distance", "decision", "liveness_passed", "ground_truth_label"])
        for r in logs:
            writer.writerow([r["id"], r["session_id"], r["timestamp"], r["matched_name"], r["computed_distance"], r["decision"], r["liveness_passed"], "correct_match"])
    
    print(f"Created ground-truth template at {GROUND_TRUTH_FILE} with {len(logs)} entries.")
    print("Labels to use in ground_truth_label column:")
    print("  - correct_match : True genuine user present")
    print("  - wrong_match   : Incorrect student matched (Impostor)")
    print("  - correctly_rejected_spoof : Spoof attack properly blocked")
    print("  - missed_spoof  : Spoof attack incorrectly accepted")
    return True

def load_ground_truth_and_logs():
    """Reads ground_truth.csv and merges with recognition logs."""
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Ground truth file {GROUND_TRUTH_FILE} not found. Creating template...")
        created = generate_ground_truth_template()
        if not created:
            return None
        print("Please review and annotate ground_truth.csv before re-running evaluate.py.")
        return None

    gt_labels = {}
    with open(GROUND_TRUTH_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = int(row["log_id"])
            label = row["ground_truth_label"].strip()
            if label:
                gt_labels[lid] = label

    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.id, r.computed_distance, r.liveness_passed, r.composite_quality,
               r.dominant_degradation, s.test_condition
        FROM recognition_logs r
        LEFT JOIN sessions s ON r.session_id = s.id
        WHERE r.computed_distance IS NOT NULL;
    """)
    logs = cursor.fetchall()
    conn.close()

    dataset = []
    for r in logs:
        lid = r["id"]
        if lid in gt_labels:
            dataset.append({
                "log_id": lid,
                "distance": float(r["computed_distance"]),
                "liveness_passed": bool(r["liveness_passed"]),
                "label": gt_labels[lid],
                "composite_quality": r["composite_quality"],
                "dominant_degradation": r["dominant_degradation"],
                "test_condition": r["test_condition"] or "normal",
            })

    return dataset

def evaluate_by_condition():
    """
    Reports recognition rate broken down by degradation condition (normal,
    low_illumination, pose_variation, occlusion, low_resolution, motion_blur),
    as tagged per-session via database.set_session_condition(). This is the
    direct, real-data answer to "recognition rate under each research-gap
    condition" -- addresses low illumination, pose variation, occlusion, low
    resolution, and motion blur individually rather than as one pooled number.
    Also reports the mean composite_quality score per condition, so the
    quality-aware module's own diagnosis of degradation severity can be
    cross-checked against the human-assigned condition label.
    """
    dataset = load_ground_truth_and_logs()
    if not dataset:
        print("Per-condition evaluation aborted: insufficient labeled data.")
        return None

    by_condition = {}
    for item in dataset:
        cond = item["test_condition"]
        by_condition.setdefault(cond, []).append(item)

    print("\n" + "=" * 72)
    print("RECOGNITION RATE BY DEGRADATION CONDITION (real logged + labeled data)")
    print("=" * 72)
    header = f"{'Condition':<28}{'N':>6}{'Accepted':>10}{'Rate':>10}{'Mean Q':>10}"
    print(header)
    print("-" * 72)

    results = {}
    for cond, items in sorted(by_condition.items()):
        genuine = [it for it in items if it["label"] == "correct_match"]
        n = len(genuine)
        accepted = sum(
            1 for it in genuine
            if it["distance"] <= config.MATCH_THRESHOLD and it["liveness_passed"]
        )
        rate = (accepted / n * 100.0) if n > 0 else float("nan")
        q_vals = [it["composite_quality"] for it in genuine if it["composite_quality"] is not None]
        mean_q = (sum(q_vals) / len(q_vals)) if q_vals else float("nan")
        results[cond] = {"n": n, "accepted": accepted, "rate_pct": rate, "mean_quality": mean_q}
        print(f"{cond:<28}{n:>6}{accepted:>10}{rate:>9.1f}%{mean_q:>10.3f}")

    print("=" * 72)
    print("Note: a condition only appears here after you have (a) tagged the")
    print("session with database.set_session_condition(session_id, 'low_illumination'")
    print("| 'pose_variation' | 'occlusion' | 'low_resolution' | 'motion_blur' | 'normal'),")
    print("and (b) labeled its logged attempts in ground_truth.csv.")
    return results

def evaluate_metrics():
    dataset = load_ground_truth_and_logs()
    if not dataset:
        print("Evaluation aborted: Insufficient labeled data.")
        return

    print(f"\nEvaluating performance across {len(dataset)} labeled real recognition events...\n")

    thresholds = np.linspace(0.20, 0.85, 66)
    far_list = []
    frr_list = []
    acc_list = []

    best_eer = 1.0
    best_eer_threshold = 0.60
    eer_val = 0.0

    for T in thresholds:
        ta, fr, fa, tr = 0, 0, 0, 0

        for item in dataset:
            dist = item["distance"]
            liveness = item["liveness_passed"]
            label = item["label"]

            # Decision rule at threshold T
            system_accept = (dist <= T) and liveness

            is_genuine = (label == "correct_match")

            if is_genuine:
                if system_accept:
                    ta += 1
                else:
                    fr += 1
            else:
                if system_accept:
                    fa += 1
                else:
                    tr += 1

        total_genuine = ta + fr
        total_impostor = fa + tr

        far = (fa / total_impostor) if total_impostor > 0 else 0.0
        frr = (fr / total_genuine) if total_genuine > 0 else 0.0
        acc = ((ta + tr) / (total_genuine + total_impostor)) if (total_genuine + total_impostor) > 0 else 0.0

        far_list.append(far)
        frr_list.append(frr)
        acc_list.append(acc)

        diff = abs(far - frr)
        if diff < best_eer:
            best_eer = diff
            eer_val = (far + frr) / 2.0
            best_eer_threshold = T

    # Plot ROC-style FAR vs FRR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_list, label='False Accept Rate (FAR)', color='red', linewidth=2)
    plt.plot(thresholds, frr_list, label='False Reject Rate (FRR)', color='blue', linewidth=2)
    plt.plot(thresholds, acc_list, label='Accuracy', color='green', linestyle='--', linewidth=2)
    plt.axvline(x=best_eer_threshold, color='gray', linestyle=':', label=f'EER Threshold ({best_eer_threshold:.2f})')

    plt.title('Biometric Recognition Performance Analysis', fontsize=14)
    plt.xlabel('Distance Matching Threshold', fontsize=12)
    plt.ylabel('Rate / Percentage', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    plt.close()

    print("=" * 60)
    print("BIOMETRIC EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Logged Events Evaluated : {len(dataset)}")
    print(f"Optimal EER Threshold        : {best_eer_threshold:.3f}")
    print(f"Equal Error Rate (EER)        : {eer_val * 100:.2f}%")
    print(f"Max System Accuracy          : {max(acc_list) * 100:.2f}%")
    print(f"ROC Curve Plot Saved To       : {OUTPUT_PLOT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_metrics()
    evaluate_by_condition()
