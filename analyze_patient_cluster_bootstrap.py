import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
N_BOOT = 5000
BOOT_SEED = 20260607
BUDGET = 20

SCENARIOS = [
    {
        "setting": "Clean",
        "json": OUT_DIR / "offline_sparse_state_posterior_v6_samesplit_24splits.json",
    },
    {
        "setting": "Noisy",
        "json": OUT_DIR / "offline_sparse_state_posterior_v6_samesplit_24splits_noisy.json",
    },
]

COMPARISONS = [
    ("Full gate", "full_risk_gate", "Feedback only", "feedback_only"),
    ("Full gate", "full_risk_gate", "Biomech prior", "physics_prior_only"),
    ("Sparse posterior", "sparse_state_posterior_gate", "Full gate", "full_risk_gate"),
]

METHODS = sorted({method for comp in COMPARISONS for method in (comp[1], comp[3])} | {"initial"})
COL_TO_IDX = {col: i for i, col in enumerate(sim.TRACK1_COLS)}


def rebuild_initial_predictions(ids, y, split):
    test_ids = set(int(pid) for pid in split["test_ids"])
    train_mask = np.asarray([int(pid) not in test_ids for pid in ids], dtype=bool)
    test_mask = ~train_mask
    x = sim.load_features(ids, train_mask)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    prob = sim.predict_probabilities(x_train, y_train, x_test)
    initial_pred = (prob >= 0.5).astype(int)
    label_ids_ordered = ids[test_mask].astype(int)
    simulator_test_ids = sorted(map(int, ids[test_mask]))
    return label_ids_ordered, simulator_test_ids, y_test, prob, initial_pred


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


def row_metric_lookup(split):
    return {(row["method"], int(row["budget"])): row for row in split["rows"]}


def collect_patient_records():
    label_patient_ids, y = sim.load_track1()
    records = []
    validation_rows = []
    for scenario in SCENARIOS:
        data = json.loads(scenario["json"].read_text(encoding="utf-8"))
        setting = scenario["setting"]
        for split in data["splits"]:
            label_ids_ordered, simulator_test_ids, y_test, prob, initial_pred = rebuild_initial_predictions(
                label_patient_ids, y, split
            )
            validation_rows.append(
                {
                    "setting": setting,
                    "seed": int(split["seed"]),
                    "method": "__split_ids__",
                    "score_abs_error": 0.0 if simulator_test_ids == sorted(map(int, split["test_ids"])) else 1.0,
                    "item_accuracy_abs_error": 0.0,
                    "total_rmse_abs_error": 0.0,
                }
            )
            eval_indices = split.get("eval_test_indices")
            if eval_indices is None:
                eval_indices = list(range(len(y_test)))
            eval_indices = np.asarray(eval_indices, dtype=int)

            predictions = {}
            predictions["initial"] = initial_pred.copy()
            predictions["physics_prior_only"] = sim.run_physics_only(initial_pred, prob, BUDGET)
            for method in ["feedback_only", "full_risk_gate", "sparse_state_posterior_gate"]:
                trace = split["traces"][f"{method}_budget{BUDGET}"]
                predictions[method] = apply_saved_trace(initial_pred, trace)

            lookup = row_metric_lookup(split)
            for method in METHODS:
                pred = predictions[method]
                y_eval = y_test[eval_indices]
                pred_eval = pred[eval_indices]
                metric = sim.metric_row(y_eval, pred_eval)
                ref = lookup.get((method, BUDGET))
                if ref is not None:
                    validation_rows.append(
                        {
                            "setting": setting,
                            "seed": int(split["seed"]),
                            "method": method,
                            "score_abs_error": abs(metric["score"] - float(ref["score"])),
                            "item_accuracy_abs_error": abs(metric["item_accuracy"] - float(ref["item_accuracy"])),
                            "total_rmse_abs_error": abs(metric["total_rmse"] - float(ref["total_rmse"])),
                        }
                    )

                for idx in eval_indices:
                    y_i = y_test[idx]
                    pred_i = pred[idx]
                    total_error = int(pred_i.sum()) - int(y_i.sum())
                    records.append(
                        {
                            "setting": setting,
                            "seed": int(split["seed"]),
                            "patient_id": int(label_ids_ordered[idx]),
                            "method": method,
                            "item_correct": int((y_i == pred_i).sum()),
                            "n_items": int(len(sim.TRACK1_COLS)),
                            "total_sqerr": float(total_error * total_error),
                            "changed_labels": int((pred_i != initial_pred[idx]).sum()),
                        }
                    )
    return records, validation_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_validation(validation_rows):
    by_setting_method = defaultdict(list)
    for row in validation_rows:
        by_setting_method[(row["setting"], row["method"])].append(row)
    out = []
    for (setting, method), rows in sorted(by_setting_method.items()):
        out.append(
            {
                "setting": setting,
                "method": method,
                "max_score_abs_error": max(float(r["score_abs_error"]) for r in rows),
                "max_item_accuracy_abs_error": max(float(r["item_accuracy_abs_error"]) for r in rows),
                "max_total_rmse_abs_error": max(float(r["total_rmse_abs_error"]) for r in rows),
            }
        )
    return out


def cluster_bootstrap(records):
    by_setting = defaultdict(list)
    for row in records:
        by_setting[row["setting"]].append(row)

    rng = np.random.default_rng(BOOT_SEED)
    summary_rows = []
    for setting, setting_records in sorted(by_setting.items()):
        by_method_patient = defaultdict(lambda: defaultdict(list))
        patient_ids = sorted({int(row["patient_id"]) for row in setting_records})
        for row in setting_records:
            by_method_patient[row["method"]][int(row["patient_id"])].append(row)

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
                        "n_split_appearances": len(by_method_patient[lhs_method][patient_ids[0]])
                        if patient_ids
                        else 0,
                    }
                )
    return summary_rows


def format_signed(value, decimals):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def write_markdown(summary_rows, validation_rows, path):
    focus = [
        row
        for row in summary_rows
        if row["metric"] in {"score", "item_accuracy"}
    ]
    lines = [
        "# Patient-Clustered Bootstrap Uncertainty",
        "",
        f"- Bootstrap resamples: `{N_BOOT}`",
        f"- Bootstrap seed: `{BOOT_SEED}`",
        "- Resampling unit: labeled patient ID. Each sampled patient contributes all of its repeated split appearances within a setting.",
        "- Interpretation: descriptive patient-clustered uncertainty over the repeated pseudo-test protocol, not independent split-level inference.",
        "",
        "## Main Comparisons",
        "",
        "| setting | comparison | metric | observed diff | 95% cluster bootstrap interval | Pr(diff > 0) | patient clusters |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in focus:
        decimals = 5 if row["metric"] == "score" else 5
        lines.append(
            f"| {row['setting']} | {row['comparison']} | {row['metric']} | "
            f"{format_signed(row['observed_diff'], decimals)} | "
            f"[{format_signed(row['ci_low'], decimals)}, {format_signed(row['ci_high'], decimals)}] | "
            f"{row['prob_positive']:.3f} | {row['n_patient_clusters']} |"
        )
    lines.extend(
        [
            "",
            "## Trace Replay Validation",
            "",
            "| setting | method | max score error | max item-acc error | max RMSE error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in validation_rows:
        lines.append(
            f"| {row['setting']} | {row['method']} | "
            f"{row['max_score_abs_error']:.3e} | {row['max_item_accuracy_abs_error']:.3e} | "
            f"{row['max_total_rmse_abs_error']:.3e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    records, validation_raw = collect_patient_records()
    validation_rows = summarize_validation(validation_raw)
    summary_rows = cluster_bootstrap(records)

    OUT_DIR.mkdir(exist_ok=True)
    write_csv(OUT_DIR / "patient_cluster_bootstrap_records.csv", records)
    write_csv(OUT_DIR / "patient_cluster_bootstrap_summary.csv", summary_rows)
    write_csv(OUT_DIR / "patient_cluster_bootstrap_trace_replay_validation.csv", validation_rows)
    (OUT_DIR / "patient_cluster_bootstrap_summary.json").write_text(
        json.dumps(
            {
                "n_boot": N_BOOT,
                "bootstrap_seed": BOOT_SEED,
                "scenarios": [{"setting": row["setting"], "json": str(row["json"])} for row in SCENARIOS],
                "summary": summary_rows,
                "trace_replay_validation": validation_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(summary_rows, validation_rows, OUT_DIR / "patient_cluster_bootstrap_uncertainty.md")
    print(
        json.dumps(
            {
                "records": str(OUT_DIR / "patient_cluster_bootstrap_records.csv"),
                "summary": str(OUT_DIR / "patient_cluster_bootstrap_summary.csv"),
                "markdown": str(OUT_DIR / "patient_cluster_bootstrap_uncertainty.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
