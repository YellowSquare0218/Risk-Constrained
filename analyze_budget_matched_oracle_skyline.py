import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
SOURCE_JSON = OUT_DIR / "offline_sparse_state_posterior_v6_samesplit_24splits.json"
BOOT_SEED = 20260608
N_BOOT = 5000
FULL_METHOD = "full_risk_gate"
POSTERIOR_METHOD = "sparse_state_posterior_gate"
BUDGET = 20


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rebuild_initial_predictions(ids, y, split):
    test_ids = set(int(pid) for pid in split["test_ids"])
    train_mask = np.asarray([int(pid) not in test_ids for pid in ids], dtype=bool)
    test_mask = ~train_mask
    x = sim.load_features(ids, train_mask)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    prob = sim.predict_probabilities(x_train, y_train, x_test)
    initial_pred = (prob >= 0.5).astype(int)
    return ids[test_mask].astype(int), y_test, initial_pred


def patient_options(y_i, pred_i, max_k):
    """All exact-b wrong-bit repair options for one patient."""
    pred_total = int(pred_i.sum())
    true_total = int(y_i.sum())
    initial_error = pred_total - true_total
    up = int(((pred_i == 0) & (y_i == 1)).sum())
    down = int(((pred_i == 1) & (y_i == 0)).sum())
    wrong = up + down
    init_correct = int((pred_i == y_i).sum())
    options = []
    for b in range(0, min(max_k, wrong) + 1):
        best = None
        for u in range(0, min(up, b) + 1):
            d = b - u
            if d < 0 or d > down:
                continue
            final_error = initial_error + u - d
            sqerr = float(final_error * final_error)
            candidate = (sqerr, final_error, u, d)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            sqerr, final_error, u, d = best
            options.append(
                {
                    "flips": b,
                    "item_correct": init_correct + b,
                    "total_sqerr": sqerr,
                    "final_error": int(final_error),
                    "up_flips": int(u),
                    "down_flips": int(d),
                }
            )
    return options


def budget_matched_oracle_records(patient_ids, y_true, initial_pred, k):
    """Hidden-label oracle that flips exactly k true-wrong bits, budget permitting.

    The oracle is budget-matched: it only repairs labels that are wrong under the
    fixed initial predictor and uses no more than the method-matched sparse
    update budget. Among item-equivalent repairs, it chooses the allocation that
    minimizes total-severity squared error.
    """
    n = len(y_true)
    total_wrong = int((y_true != initial_pred).sum())
    k_eff = min(int(k), total_wrong)
    options = [patient_options(y_true[i], initial_pred[i], k_eff) for i in range(n)]

    inf = 1e18
    dp = np.full((n + 1, k_eff + 1), inf, dtype=np.float64)
    back = [[None for _ in range(k_eff + 1)] for _ in range(n + 1)]
    dp[0, 0] = 0.0
    for i, opts in enumerate(options, 1):
        for used in range(k_eff + 1):
            if not np.isfinite(dp[i - 1, used]):
                continue
            for opt_idx, opt in enumerate(opts):
                nxt = used + int(opt["flips"])
                if nxt > k_eff:
                    continue
                val = dp[i - 1, used] + float(opt["total_sqerr"])
                if val < dp[i, nxt]:
                    dp[i, nxt] = val
                    back[i][nxt] = (used, opt_idx)

    records = []
    used = k_eff
    selected = [0] * n
    for i in range(n, 0, -1):
        prev_used, opt_idx = back[i][used]
        selected[i - 1] = opt_idx
        used = prev_used

    for i, opt_idx in enumerate(selected):
        opt = options[i][opt_idx]
        records.append(
            {
                "patient_id": int(patient_ids[i]),
                "item_correct": int(opt["item_correct"]),
                "n_items": int(len(sim.TRACK1_COLS)),
                "total_sqerr": float(opt["total_sqerr"]),
                "changed_labels": int(opt["flips"]),
            }
        )
    return records


def metric_from_records(records):
    item_total = sum(int(r["item_correct"]) for r in records)
    n_items = sum(int(r["n_items"]) for r in records)
    total_sse = sum(float(r["total_sqerr"]) for r in records)
    n_rows = len(records)
    item_acc = item_total / float(n_items)
    rmse = float(np.sqrt(total_sse / float(n_rows)))
    return {
        "score": 0.5 * (item_acc + 1.0 - rmse / 34.0),
        "item_accuracy": item_acc,
        "total_rmse": rmse,
        "changed_labels": float(np.mean([float(r["changed_labels"]) for r in records])),
    }


def cluster_bootstrap(records_by_k):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for label, records in records_by_k.items():
        by_patient = defaultdict(list)
        for row in records:
            by_patient[int(row["patient_id"])].append(row)
        patient_ids = sorted(by_patient)
        observed = metric_from_records(records)
        boot = defaultdict(list)
        for _ in range(N_BOOT):
            sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            sampled_records = []
            for pid in sampled:
                sampled_records.extend(by_patient[int(pid)])
            metric = metric_from_records(sampled_records)
            for metric_name in ["score", "item_accuracy", "total_rmse", "changed_labels"]:
                boot[metric_name].append(metric[metric_name])
        for metric_name, values in boot.items():
            arr = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "oracle": label,
                    "metric": metric_name,
                    "observed": observed[metric_name],
                    "bootstrap_mean": float(np.mean(arr)),
                    "ci_low": float(np.quantile(arr, 0.025)),
                    "ci_high": float(np.quantile(arr, 0.975)),
                    "n_boot": N_BOOT,
                    "n_patient_clusters": len(patient_ids),
                }
            )
    return rows


def fmt(value, decimals=5):
    return f"{value:.{decimals}f}"


def main():
    ids, y = sim.load_track1()
    data = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    split_rows = []
    records_by_k = defaultdict(list)

    full_changed = []
    posterior_changed = []
    initial_acc = []
    full_acc = []
    posterior_acc = []
    for split in data["splits"]:
        row_lookup = {(row["method"], int(row["budget"])): row for row in split["rows"]}
        full_changed.append(float(row_lookup[(FULL_METHOD, BUDGET)]["changed_labels"]))
        posterior_changed.append(float(row_lookup[(POSTERIOR_METHOD, BUDGET)]["changed_labels"]))
        initial_acc.append(float(row_lookup[("initial", BUDGET)]["item_accuracy"]))
        full_acc.append(float(row_lookup[(FULL_METHOD, BUDGET)]["item_accuracy"]))
        posterior_acc.append(float(row_lookup[(POSTERIOR_METHOD, BUDGET)]["item_accuracy"]))

    k_full = int(round(float(np.mean(full_changed))))
    k_post = int(round(float(np.mean(posterior_changed))))

    for split in data["splits"]:
        patient_ids, y_test, initial_pred = rebuild_initial_predictions(ids, y, split)
        for label, k in [(f"K={k_full}", k_full), (f"K={k_post}", k_post)]:
            records = budget_matched_oracle_records(patient_ids, y_test, initial_pred, k)
            metric = metric_from_records(records)
            split_rows.append(
                {
                    "seed": int(split["seed"]),
                    "oracle": label,
                    "k": k,
                    "score": metric["score"],
                    "item_accuracy": metric["item_accuracy"],
                    "total_rmse": metric["total_rmse"],
                    "changed_labels": sum(int(r["changed_labels"]) for r in records),
                }
            )
            for row in records:
                row = dict(row)
                row["seed"] = int(split["seed"])
                records_by_k[label].append(row)

    summary_rows = []
    for label in sorted(records_by_k):
        subset = [row for row in split_rows if row["oracle"] == label]
        summary_rows.append(
            {
                "oracle": label,
                "k": int(subset[0]["k"]),
                "score_mean": float(np.mean([r["score"] for r in subset])),
                "item_accuracy_mean": float(np.mean([r["item_accuracy"] for r in subset])),
                "total_rmse_mean": float(np.mean([r["total_rmse"] for r in subset])),
                "changed_labels_mean": float(np.mean([r["changed_labels"] for r in subset])),
            }
        )

    boot_rows = cluster_bootstrap(records_by_k)
    initial_acc_mean = float(np.mean(initial_acc))
    full_acc_mean = float(np.mean(full_acc))
    posterior_acc_mean = float(np.mean(posterior_acc))
    oracle_full_acc = next(r["item_accuracy_mean"] for r in summary_rows if r["oracle"] == f"K={k_full}")
    oracle_post_acc = next(r["item_accuracy_mean"] for r in summary_rows if r["oracle"] == f"K={k_post}")
    headroom_full = (full_acc_mean - initial_acc_mean) / max(1e-12, oracle_full_acc - initial_acc_mean)
    headroom_post = (posterior_acc_mean - initial_acc_mean) / max(1e-12, oracle_post_acc - initial_acc_mean)

    write_csv(OUT_DIR / "budget_matched_oracle_skyline_v1_split_rows.csv", split_rows)
    write_csv(OUT_DIR / "budget_matched_oracle_skyline_v1_summary.csv", summary_rows)
    write_csv(OUT_DIR / "budget_matched_oracle_skyline_v1_bootstrap.csv", boot_rows)

    boot_lookup = {(row["oracle"], row["metric"]): row for row in boot_rows}
    full_boot = boot_lookup[(f"K={k_full}", "item_accuracy")]
    post_boot = boot_lookup[(f"K={k_post}", "item_accuracy")]
    oracle_full_summary = next(r for r in summary_rows if r["oracle"] == f"K={k_full}")
    oracle_post_summary = next(r for r in summary_rows if r["oracle"] == f"K={k_post}")

    lines = [
        "# Budget-Matched Oracle Skyline",
        "",
        f"- Source splits: `{SOURCE_JSON}`",
        f"- Full-gate mean changed labels: `{np.mean(full_changed):.3f}`, budget-matched oracle K: `{k_full}`",
        f"- Posterior mean changed labels: `{np.mean(posterior_changed):.3f}`, budget-matched oracle K: `{k_post}`",
        "- Oracle rule: flips only hidden-label wrong bits from the fixed initial predictor; the number of flips is strictly budget-matched; among item-equivalent repairs, dynamic programming chooses the allocation with minimum total-severity SSE.",
        "",
        "| oracle | score | item acc | item acc 95% CI | total RMSE | changed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        boot = boot_lookup[(row["oracle"], "item_accuracy")]
        lines.append(
            f"| {row['oracle']} | {row['score_mean']:.5f} | {row['item_accuracy_mean']:.5f} | "
            f"[{boot['ci_low']:.5f}, {boot['ci_high']:.5f}] | {row['total_rmse_mean']:.3f} | {row['changed_labels_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Headroom Recovery",
            "",
            f"- Initial item accuracy: `{initial_acc_mean:.5f}`",
            f"- Full gate item accuracy: `{full_acc_mean:.5f}`",
            f"- Sparse posterior item accuracy: `{posterior_acc_mean:.5f}`",
            f"- Full gate recovery vs K={k_full} oracle: `{100.0 * headroom_full:.2f}%`",
            f"- Posterior recovery vs K={k_post} oracle: `{100.0 * headroom_post:.2f}%`",
            "",
            "## Values for Manuscript",
            "",
            f"- ORACLE_SCORE_K{k_full}: `{oracle_full_summary['score_mean']:.5f}`",
            f"- ORACLE_ACC_K{k_full}: `{oracle_full_summary['item_accuracy_mean']:.5f}`",
            f"- ORACLE_ACC_K{k_full}_CI: `[{full_boot['ci_low']:.5f}, {full_boot['ci_high']:.5f}]`",
            f"- ORACLE_RMSE_K{k_full}: `{oracle_full_summary['total_rmse_mean']:.3f}`",
            f"- ORACLE_SCORE_K{k_post}: `{oracle_post_summary['score_mean']:.5f}`",
            f"- ORACLE_ACC_K{k_post}: `{oracle_post_summary['item_accuracy_mean']:.5f}`",
            f"- ORACLE_ACC_K{k_post}_CI: `[{post_boot['ci_low']:.5f}, {post_boot['ci_high']:.5f}]`",
            f"- ORACLE_RMSE_K{k_post}: `{oracle_post_summary['total_rmse_mean']:.3f}`",
            f"- PCT_FULL: `{100.0 * headroom_full:.1f}%`",
            f"- PCT_POST: `{100.0 * headroom_post:.1f}%`",
        ]
    )
    (OUT_DIR / "budget_matched_oracle_skyline_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
