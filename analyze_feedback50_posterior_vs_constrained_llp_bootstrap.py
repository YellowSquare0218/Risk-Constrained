import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_offline_external_adaptation_baselines as ext
import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
N_BOOT = 5000
BOOT_SEED = 20260608
COL_TO_IDX = {col: i for i, col in enumerate(sim.TRACK1_COLS)}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rebuild_initial_predictions(ids, y, test_ids):
    test_ids = set(int(pid) for pid in test_ids)
    train_mask = np.asarray([int(pid) not in test_ids for pid in ids], dtype=bool)
    test_mask = ~train_mask
    x = sim.load_features(ids, train_mask)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    prob = sim.predict_probabilities(x_train, y_train, x_test)
    initial_pred = (prob >= 0.5).astype(int)
    return ids[test_mask].astype(int), y_train, y_test, prob, initial_pred


def apply_saved_trace(initial_pred, trace):
    pred = initial_pred.copy()
    for step in trace:
        if not bool(step.get("accepted")):
            continue
        row_idx = int(step["row_idx"])
        col_idx = COL_TO_IDX[str(step["column"])]
        pred[row_idx, col_idx] = int(step["after"])
    return pred


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


def collect_records(posterior_json, setting, args):
    ids, y = sim.load_track1()
    data = json.loads(Path(posterior_json).read_text(encoding="utf-8"))
    records = []
    validation_rows = []
    for split in data["splits"]:
        seed = int(split["seed"])
        label_ids_ordered, y_train, y_test, prob, initial_pred = rebuild_initial_predictions(ids, y, split["test_ids"])
        feedback_mask = np.zeros(len(y_test), dtype=bool)
        feedback_mask[np.asarray(split["feedback_test_indices"], dtype=int)] = True
        eval_mask = np.zeros(len(y_test), dtype=bool)
        eval_mask[np.asarray(split["eval_test_indices"], dtype=int)] = True

        posterior_pred = apply_saved_trace(initial_pred, split["traces"]["sparse_state_posterior_gate_budget20"])
        method_rng = np.random.default_rng(seed + 2111)
        constrained_pred, _ = ext.run_llp_constrained_sparse_gate(
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

        predictions = {
            "sparse_state_posterior_gate": posterior_pred,
            "llp_constrained_sparse_gate": constrained_pred,
        }

        lookup = {(row["method"], int(row["budget"])): row for row in split["rows"]}
        for method, pred in predictions.items():
            metric = sim.metric_row(y_test[eval_mask], pred[eval_mask])
            if method == "sparse_state_posterior_gate":
                ref = lookup.get((method, args.budget))
                validation_rows.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "method": method,
                        "score_abs_error": abs(metric["score"] - float(ref["score"])),
                        "item_accuracy_abs_error": abs(metric["item_accuracy"] - float(ref["item_accuracy"])),
                        "total_rmse_abs_error": abs(metric["total_rmse"] - float(ref["total_rmse"])),
                    }
                )

            for idx in np.flatnonzero(eval_mask):
                y_i = y_test[idx]
                pred_i = pred[idx]
                total_error = int(pred_i.sum()) - int(y_i.sum())
                records.append(
                    {
                        "setting": setting,
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


def cluster_bootstrap(records, setting):
    by_method_patient = defaultdict(lambda: defaultdict(list))
    patient_ids = sorted({int(row["patient_id"]) for row in records})
    for row in records:
        by_method_patient[row["method"]][int(row["patient_id"])].append(row)

    lhs_method = "sparse_state_posterior_gate"
    rhs_method = "llp_constrained_sparse_gate"
    observed = {}
    for method in [lhs_method, rhs_method]:
        method_records = [row for pid in patient_ids for row in by_method_patient[method][pid]]
        observed[method] = metric_from_records(method_records)

    rng = np.random.default_rng(BOOT_SEED)
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

    rows = []
    for metric_name in ["score", "item_accuracy", "total_rmse", "changed_labels"]:
        diffs = np.asarray(boot_diffs[metric_name], dtype=np.float64)
        rows.append(
            {
                "setting": setting,
                "comparison": "Posterior gate - Constrained LLP",
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
    return rows


def signed(value, decimals=5):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def write_markdown(summary_rows, validation_rows, path):
    lines = [
        "# Feedback-50 Posterior vs Constrained LLP Bootstrap",
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
    lines.extend(["", "## Posterior Trace-Replay Validation", ""])
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
    parser.add_argument("--posterior-json", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--feedback-noise", type=float, default=0.0)
    parser.add_argument("--flip-feedback-prob", type=float, default=0.0)
    parser.add_argument("--out-prefix", default="feedback50_posterior_constrained_llp_bootstrap_v1")
    args = parser.parse_args()

    records, validation_rows = collect_records(args.posterior_json, args.setting, SimpleNamespace(**vars(args)))
    summary_rows = cluster_bootstrap(records, args.setting)
    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    write_csv(csv_path, summary_rows)
    write_markdown(summary_rows, validation_rows, md_path)
    print({"csv": str(csv_path), "md": str(md_path)})


if __name__ == "__main__":
    main()
