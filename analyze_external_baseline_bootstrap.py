import argparse
import csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_offline_external_adaptation_baselines as ext
import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
N_BOOT = 5000
BOOT_SEED = 20260608

COMPARISONS = [
    ("Full gate", "full_risk_gate", "LLP feedback", "llp_feedback_projection"),
    ("Constrained LLP", "llp_constrained_sparse_gate", "Full gate", "full_risk_gate"),
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_predictions(ids, y, seed, args):
    report = ext.run_split(ids, y, seed, args.test_fraction, args.budget, args)
    rng = np.random.default_rng(seed)
    order = np.asarray(ids).copy()
    rng.shuffle(order)
    n_test = max(8, int(round(len(ids) * args.test_fraction)))
    test_ids = set(map(int, order[:n_test]))
    train_mask = np.asarray([int(pid) not in test_ids for pid in ids], dtype=bool)
    test_mask = ~train_mask

    x = sim.load_features(ids, train_mask)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    prob = sim.predict_probabilities(x_train, y_train, x_test)
    initial_pred = (prob >= 0.5).astype(int)

    feedback_mask, eval_mask = sim.make_feedback_eval_masks(
        len(y_test), rng, args.feedback_fraction, args.eval_scope
    )
    row_metrics = {row["method"]: row for row in report["rows"]}

    method_preds = {
        "initial": initial_pred.copy(),
        "llp_prevalence_projection": ext.prevalence_calibration(prob, y_train.mean(axis=0)),
    }
    method_rng = np.random.default_rng(seed + 1777)
    method_preds["llp_feedback_projection"], _ = ext.run_llp_feedback_projection(
        y_test,
        initial_pred,
        prob,
        y_train.mean(axis=0),
        args.budget,
        method_rng,
        feedback_mask,
        eval_mask,
        feedback_noise=args.feedback_noise,
        flip_feedback_prob=args.flip_feedback_prob,
    )
    method_rng = np.random.default_rng(seed + 2111)
    method_preds["llp_constrained_sparse_gate"], _ = ext.run_llp_constrained_sparse_gate(
        y_test,
        initial_pred,
        prob,
        y_train.mean(axis=0),
        args.budget,
        method_rng,
        feedback_mask,
        eval_mask,
        feedback_noise=args.feedback_noise,
        flip_feedback_prob=args.flip_feedback_prob,
    )
    method_rng = np.random.default_rng(seed + 2029)
    method_preds["aggregate_threshold_feedback"], _ = ext.run_aggregate_threshold_feedback(
        y_test,
        initial_pred,
        prob,
        args.budget,
        method_rng,
        feedback_mask,
        eval_mask,
        feedback_noise=args.feedback_noise,
        flip_feedback_prob=args.flip_feedback_prob,
    )
    method_rng = np.random.default_rng(seed + args.budget * 101 + sim.METHODS.index("full_risk_gate") * 1009)
    method_preds["full_risk_gate"], _ = sim.run_feedback_method(
        y_test,
        initial_pred,
        prob,
        "full_risk_gate",
        args.budget,
        method_rng,
        feedback_noise=args.feedback_noise,
        flip_feedback_prob=args.flip_feedback_prob,
        feedback_mask=feedback_mask,
        eval_mask=eval_mask,
    )

    validation_rows = []
    for method, pred in method_preds.items():
        metric = sim.metric_row(y_test[eval_mask], pred[eval_mask])
        ref = row_metrics.get(method)
        if ref is None:
            continue
        validation_rows.append(
            {
                "seed": seed,
                "method": method,
                "score_abs_error": abs(metric["score"] - float(ref["score"])),
                "item_accuracy_abs_error": abs(metric["item_accuracy"] - float(ref["item_accuracy"])),
                "total_rmse_abs_error": abs(metric["total_rmse"] - float(ref["total_rmse"])),
            }
        )

    records = []
    label_ids_ordered = ids[test_mask].astype(int)
    eval_indices = np.flatnonzero(eval_mask)
    for method, pred in method_preds.items():
        for idx in eval_indices:
            y_i = y_test[idx]
            pred_i = pred[idx]
            total_error = int(pred_i.sum()) - int(y_i.sum())
            records.append(
                {
                    "seed": seed,
                    "patient_id": int(label_ids_ordered[idx]),
                    "method": method,
                    "item_correct": int((y_i == pred_i).sum()),
                    "n_items": int(len(sim.TRACK1_COLS)),
                    "total_sqerr": float(total_error * total_error),
                    "changed_labels": int((pred_i != initial_pred[idx]).sum()),
                }
            )
    return records, validation_rows


def metric_from_records(records):
    item_total = sum(int(r["item_correct"]) for r in records)
    n_items = sum(int(r["n_items"]) for r in records)
    total_sse = sum(float(r["total_sqerr"]) for r in records)
    n_rows = len(records)
    if n_rows == 0 or n_items == 0:
        return {"score": np.nan, "item_accuracy": np.nan, "total_rmse": np.nan, "changed_labels": np.nan}
    item_acc = item_total / float(n_items)
    rmse = float(np.sqrt(total_sse / float(n_rows)))
    return {
        "score": 0.5 * (item_acc + 1.0 - rmse / 34.0),
        "item_accuracy": item_acc,
        "total_rmse": rmse,
        "changed_labels": float(np.mean([float(r["changed_labels"]) for r in records])),
    }


def cluster_bootstrap(records, setting):
    by_method_patient = defaultdict(lambda: defaultdict(list))
    patient_ids = sorted({int(row["patient_id"]) for row in records})
    for row in records:
        by_method_patient[row["method"]][int(row["patient_id"])].append(row)

    rng = np.random.default_rng(BOOT_SEED)
    summary_rows = []
    for lhs_label, lhs_method, rhs_label, rhs_method in COMPARISONS:
        observed = {}
        for method in [lhs_method, rhs_method]:
            method_records = [row for pid in patient_ids for row in by_method_patient[method][pid]]
            observed[method] = metric_from_records(method_records)

        boot_diffs = defaultdict(list)
        for _ in range(N_BOOT):
            sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            metrics = {}
            for method in [lhs_method, rhs_method]:
                method_records = []
                for pid in sampled:
                    method_records.extend(by_method_patient[method][int(pid)])
                metrics[method] = metric_from_records(method_records)
            for metric_name in ["score", "item_accuracy", "total_rmse", "changed_labels"]:
                boot_diffs[metric_name].append(metrics[lhs_method][metric_name] - metrics[rhs_method][metric_name])

        for metric_name in ["score", "item_accuracy", "total_rmse", "changed_labels"]:
            diffs = np.asarray(boot_diffs[metric_name], dtype=np.float64)
            summary_rows.append(
                {
                    "setting": setting,
                    "comparison": f"{lhs_label} - {rhs_label}",
                    "lhs_method": lhs_method,
                    "rhs_method": rhs_method,
                    "metric": metric_name,
                    "observed_diff": observed[lhs_method][metric_name] - observed[rhs_method][metric_name],
                    "bootstrap_mean_diff": float(np.mean(diffs)),
                    "ci_low": float(np.quantile(diffs, 0.025)),
                    "ci_high": float(np.quantile(diffs, 0.975)),
                    "prob_positive": float(np.mean(diffs > 0.0)),
                    "n_boot": N_BOOT,
                    "n_patient_clusters": len(patient_ids),
                }
            )
    return summary_rows


def signed(value, decimals=5):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def write_markdown(summary_rows, validation_rows, path):
    lines = [
        "# External Baseline Patient-Clustered Bootstrap",
        "",
        f"- Bootstrap resamples: `{N_BOOT}`",
        f"- Bootstrap seed: `{BOOT_SEED}`",
        "- Resampling unit: labeled patient ID with all repeated split appearances.",
        "",
        "| setting | comparison | metric | observed diff | 95% cluster bootstrap interval | Pr(diff > 0) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["metric"] not in {"score", "item_accuracy"}:
            continue
        lines.append(
            f"| {row['setting']} | {row['comparison']} | {row['metric']} | "
            f"{signed(float(row['observed_diff']))} | "
            f"[{signed(float(row['ci_low']))}, {signed(float(row['ci_high']))}] | "
            f"{float(row['prob_positive']):.3f} |"
        )

    lines.extend(["", "## Trace-Replay Validation", ""])
    by_method = defaultdict(list)
    for row in validation_rows:
        by_method[row["method"]].append(row)
    lines.append("| method | max score abs error | max item-acc abs error | max total-rmse abs error |")
    lines.append("|---|---:|---:|---:|")
    for method, rows in sorted(by_method.items()):
        lines.append(
            f"| {method} | "
            f"{max(float(r['score_abs_error']) for r in rows):.8f} | "
            f"{max(float(r['item_accuracy_abs_error']) for r in rows):.8f} | "
            f"{max(float(r['total_rmse_abs_error']) for r in rows):.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--feedback-fraction", type=float, default=1.0)
    parser.add_argument("--eval-scope", choices=["full", "private"], default="full")
    parser.add_argument("--feedback-noise", type=float, default=0.0)
    parser.add_argument("--flip-feedback-prob", type=float, default=0.0)
    parser.add_argument("--setting", default="Clean")
    parser.add_argument("--out-prefix", default="external_baseline_bootstrap_v1")
    args = parser.parse_args()

    ids, y = sim.load_track1()
    all_records = []
    validation_rows = []
    split_args = SimpleNamespace(**vars(args))
    for i in range(args.splits):
        seed = args.seed + i * 17
        records, validations = split_predictions(ids, y, seed, split_args)
        all_records.extend(records)
        validation_rows.extend(validations)
        print(f"completed split {i + 1}/{args.splits}", flush=True)

    summary_rows = cluster_bootstrap(all_records, args.setting)
    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    write_csv(csv_path, summary_rows)
    write_markdown(summary_rows, validation_rows, md_path)
    print({"csv": str(csv_path), "md": str(md_path)})


if __name__ == "__main__":
    main()
