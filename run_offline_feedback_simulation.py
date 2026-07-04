import argparse
import csv
import json
from pathlib import Path

import numpy as np

import pose_feature_baseline as base


OUT_DIR = Path("outputs")
FEATURE_FILES = [
    OUT_DIR / "pose_features.npz",
    OUT_DIR / "physics_pose_features.npz",
    OUT_DIR / "temporal_pose_features.npz",
    OUT_DIR / "gait_event_phase_features.npz",
]
TRACK1_COLS = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
BUDGETS = [0, 1, 3, 5, 10, 20]
METHODS = [
    "initial",
    "physics_prior_only",
    "feedback_only",
    "feedback_total",
    "feedback_physics",
    "full_risk_gate",
]


def load_track1():
    rows = base.load_json("track1_train.json")
    ids = []
    y = []
    for row in rows:
        ids.append(int(row["patient_id"]))
        y.append(
            [int(row["left"][str(i)]) for i in range(1, 18)]
            + [int(row["right"][str(i)]) for i in range(1, 18)]
        )
    return np.asarray(ids, dtype=int), np.asarray(y, dtype=int)


def load_feature_block(path, wanted_ids, train_mask):
    z = np.load(path, allow_pickle=True)
    ids = [int(v) for v in z["patient_ids"]]
    index = {pid: i for i, pid in enumerate(ids)}
    x = np.asarray(z["X"][[index[int(pid)] for pid in wanted_ids]], dtype=np.float32)
    train = x[train_mask]
    mean = np.nanmean(train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    x = np.where(np.isfinite(x), x, mean)
    std = np.nanstd(x[train_mask], axis=0)
    keep = std > 1e-6
    if int(keep.sum()) > 6000:
        variances = np.nanvar(x[train_mask][:, keep], axis=0)
        selected = np.argsort(variances)[-6000:]
        keep_idx = np.flatnonzero(keep)[selected]
    else:
        keep_idx = np.flatnonzero(keep)
    x = (x[:, keep_idx] - mean[keep_idx]) / np.maximum(std[keep_idx], 1e-6)
    return np.clip(np.nan_to_num(x), -6.0, 6.0) / np.sqrt(max(1, len(keep_idx)))


def load_features(ids, train_mask):
    blocks = [load_feature_block(path, ids, train_mask) for path in FEATURE_FILES if path.exists()]
    return np.hstack(blocks)


def cosine_distance(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return 1.0 - b @ a.T


def weighted_knn(dist, y, k):
    order = np.argsort(dist)[: min(k, len(dist))]
    d = dist[order]
    scale = np.median(d) + 1e-6
    w = np.exp(-d / scale)
    w /= max(float(w.sum()), 1e-8)
    return (w[:, None] * y[order]).sum(axis=0)


def predict_probabilities(x_train, y_train, x_test):
    dist = cosine_distance(x_train, x_test)
    probs = []
    for i in range(len(x_test)):
        p7 = weighted_knn(dist[i], y_train, 7)
        p15 = weighted_knn(dist[i], y_train, 15)
        prior = y_train.mean(axis=0)
        probs.append(0.45 * p7 + 0.40 * p15 + 0.15 * prior)
    return np.clip(np.vstack(probs), 0.02, 0.98)


def track1_score(y_true, y_pred):
    item_acc = float((y_true == y_pred).mean())
    true_total = y_true.sum(axis=1)
    pred_total = y_pred.sum(axis=1)
    rmse = float(np.sqrt(np.mean((true_total - pred_total) ** 2)))
    return 0.5 * (item_acc + 1.0 - rmse / 34.0)


def metric_row(y_true, y_pred):
    true_total = y_true.sum(axis=1)
    pred_total = y_pred.sum(axis=1)
    return {
        "score": track1_score(y_true, y_pred),
        "item_accuracy": float((y_true == y_pred).mean()),
        "total_rmse": float(np.sqrt(np.mean((true_total - pred_total) ** 2))),
        "mean_abs_total_error": float(np.mean(np.abs(true_total - pred_total))),
        "changed_labels": None,
    }


def candidate_pool(prob, pred):
    pool = []
    pred_total = pred.sum(axis=1)
    prob_total = prob.sum(axis=1)
    for row_idx in range(pred.shape[0]):
        total_gap = float(prob_total[row_idx] - pred_total[row_idx])
        for col_idx, col in enumerate(TRACK1_COLS):
            before = int(pred[row_idx, col_idx])
            after = 1 - before
            p_after = float(prob[row_idx, col_idx] if after == 1 else 1.0 - prob[row_idx, col_idx])
            direction = 1 if after == 1 else -1
            total_agrees = direction * total_gap > 0.25
            pool.append(
                {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "column": col,
                    "before": before,
                    "after": after,
                    "p_after": p_after,
                    "total_gap": total_gap,
                    "total_agrees": bool(total_agrees),
                    "rank_feedback": abs(prob[row_idx, col_idx] - 0.5),
                    "rank_physics": p_after + (0.10 if total_agrees else 0.0),
                    "rank_full": 0.80 * p_after + (0.22 if total_agrees else -0.02) - 0.015 * abs(total_gap),
                }
            )
    return pool


def ordered_pool(pool, method):
    if method == "feedback_only":
        key = lambda r: (r["rank_feedback"], r["p_after"])
    elif method == "feedback_total":
        key = lambda r: (int(r["total_agrees"]), r["rank_feedback"], r["p_after"])
    elif method == "feedback_physics":
        key = lambda r: (r["p_after"], int(r["total_agrees"]), r["rank_feedback"])
    else:
        key = lambda r: (r["rank_full"], r["p_after"], int(r["total_agrees"]))
    return sorted(pool, key=key, reverse=True)


def gate_allows(method, flip):
    if method == "feedback_only":
        return True
    if method == "feedback_total":
        return flip["total_agrees"] or flip["p_after"] >= 0.47
    if method == "feedback_physics":
        return flip["p_after"] >= 0.43
    if method == "full_risk_gate":
        return flip["p_after"] >= 0.42 and (flip["total_agrees"] or flip["p_after"] >= 0.48)
    return False


def apply_flip(pred, flip):
    out = pred.copy()
    out[flip["row_idx"], flip["col_idx"]] = flip["after"]
    return out


def run_feedback_method(
    y_true,
    initial_pred,
    prob,
    method,
    max_budget,
    rng,
    feedback_noise=0.0,
    flip_feedback_prob=0.0,
    feedback_mask=None,
    eval_mask=None,
):
    pred = initial_pred.copy()
    trace = []
    used = set()
    if feedback_mask is None:
        feedback_mask = np.ones(pred.shape[0], dtype=bool)
    if eval_mask is None:
        eval_mask = np.ones(pred.shape[0], dtype=bool)
    for step in range(1, max_budget + 1):
        pool = ordered_pool(candidate_pool(prob, pred), method)
        flip = next(
            (
                row
                for row in pool
                if (row["row_idx"], row["col_idx"], row["after"]) not in used and gate_allows(method, row)
            ),
            None,
        )
        if flip is None:
            break
        used.add((flip["row_idx"], flip["col_idx"], flip["after"]))
        before_score = track1_score(y_true[feedback_mask], pred[feedback_mask])
        probe = apply_flip(pred, flip)
        after_score = track1_score(y_true[feedback_mask], probe[feedback_mask])
        delta = after_score - before_score
        before_eval_score = track1_score(y_true[eval_mask], pred[eval_mask])
        after_eval_score = track1_score(y_true[eval_mask], probe[eval_mask])
        eval_delta = after_eval_score - before_eval_score
        observed_delta = delta + float(rng.normal(0.0, feedback_noise))
        if float(rng.random()) < flip_feedback_prob:
            observed_delta = -observed_delta
        accepted = observed_delta > 0
        if accepted:
            pred = probe
        trace.append(
            {
                "step": step,
                "row_idx": int(flip["row_idx"]),
                "column": flip["column"],
                "before": int(flip["before"]),
                "after": int(flip["after"]),
                "p_after": round(float(flip["p_after"]), 4),
                "total_gap": round(float(flip["total_gap"]), 4),
                "total_agrees": bool(flip["total_agrees"]),
                "aggregate_delta": round(float(delta), 6),
                "evaluation_delta": round(float(eval_delta), 6),
                "observed_delta": round(float(observed_delta), 6),
                "accepted": bool(accepted),
                "harmful_accept": bool(accepted and eval_delta <= 0),
            }
        )
    return pred, trace


def run_physics_only(initial_pred, prob, budget):
    pred = initial_pred.copy()
    for flip in ordered_pool(candidate_pool(prob, pred), "full_risk_gate"):
        if budget <= 0:
            break
        if flip["p_after"] < 0.43 or not flip["total_agrees"]:
            continue
        pred = apply_flip(pred, flip)
        budget -= 1
    return pred


def make_feedback_eval_masks(n_test, rng, feedback_fraction, eval_scope):
    feedback_fraction = max(0.0, min(1.0, float(feedback_fraction)))
    if feedback_fraction >= 0.999:
        feedback_mask = np.ones(n_test, dtype=bool)
    else:
        n_feedback = int(round(n_test * feedback_fraction))
        min_feedback = 2 if eval_scope == "private" and n_test > 2 else 1
        n_feedback = max(min_feedback, min(n_feedback, n_test))
        if eval_scope == "private" and n_feedback >= n_test:
            n_feedback = n_test - 1
        order = np.arange(n_test)
        rng.shuffle(order)
        feedback_mask = np.zeros(n_test, dtype=bool)
        feedback_mask[order[:n_feedback]] = True
    if eval_scope == "private":
        eval_mask = ~feedback_mask
        if not bool(eval_mask.any()):
            eval_mask = np.ones(n_test, dtype=bool)
    else:
        eval_mask = np.ones(n_test, dtype=bool)
    return feedback_mask, eval_mask


def run_split(
    ids,
    y,
    seed,
    test_fraction,
    budgets,
    feedback_noise=0.0,
    flip_feedback_prob=0.0,
    feedback_fraction=1.0,
    eval_scope="full",
):
    rng = np.random.default_rng(seed)
    order = np.asarray(ids).copy()
    rng.shuffle(order)
    n_test = max(8, int(round(len(ids) * test_fraction)))
    test_ids = set(map(int, order[:n_test]))
    train_mask = np.asarray([int(pid) not in test_ids for pid in ids], dtype=bool)
    test_mask = ~train_mask
    x = load_features(ids, train_mask)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    prob = predict_probabilities(x_train, y_train, x_test)
    initial_pred = (prob >= 0.5).astype(int)
    feedback_mask, eval_mask = make_feedback_eval_masks(len(y_test), rng, feedback_fraction, eval_scope)
    initial_score = track1_score(y_test[eval_mask], initial_pred[eval_mask])

    rows = []
    traces = {}
    for method in METHODS:
        for budget in budgets:
            if method == "initial":
                pred = initial_pred.copy()
                trace = []
            elif method == "physics_prior_only":
                pred = run_physics_only(initial_pred, prob, budget)
                trace = []
            else:
                method_rng = np.random.default_rng(seed + budget * 101 + METHODS.index(method) * 1009)
                pred, trace = run_feedback_method(
                    y_test,
                    initial_pred,
                    prob,
                    method,
                    budget,
                    method_rng,
                    feedback_noise=feedback_noise,
                    flip_feedback_prob=flip_feedback_prob,
                    feedback_mask=feedback_mask,
                    eval_mask=eval_mask,
                )
            metrics = metric_row(y_test[eval_mask], pred[eval_mask])
            metrics["changed_labels"] = int((pred != initial_pred).sum())
            accepted = [row for row in trace if row.get("accepted")]
            metrics["accepted_updates"] = len(accepted)
            metrics["harmful_accepted_updates"] = sum(int(row.get("harmful_accept", False)) for row in accepted)
            metrics.update(
                {
                    "seed": seed,
                    "method": method,
                    "budget": budget,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "n_feedback": int(feedback_mask.sum()),
                    "n_eval": int(eval_mask.sum()),
                    "feedback_fraction": float(feedback_mask.mean()),
                    "eval_scope": eval_scope,
                    "initial_score": initial_score,
                    "score_gain": metrics["score"] - initial_score,
                }
            )
            rows.append(metrics)
            traces[f"{method}_budget{budget}"] = trace
    return {
        "seed": seed,
        "test_ids": sorted(test_ids),
        "feedback_test_indices": np.flatnonzero(feedback_mask).astype(int).tolist(),
        "eval_test_indices": np.flatnonzero(eval_mask).astype(int).tolist(),
        "rows": rows,
        "traces": traces,
    }


def summarize(rows):
    out = []
    keys = sorted({(row["method"], row["budget"]) for row in rows})
    for method, budget in keys:
        subset = [row for row in rows if row["method"] == method and row["budget"] == budget]
        out.append(
            {
                "method": method,
                "budget": budget,
                "splits": len(subset),
                "score_mean": float(np.mean([r["score"] for r in subset])),
                "score_std": float(np.std([r["score"] for r in subset])),
                "gain_mean": float(np.mean([r["score_gain"] for r in subset])),
                "gain_std": float(np.std([r["score_gain"] for r in subset])),
                "item_accuracy_mean": float(np.mean([r["item_accuracy"] for r in subset])),
                "total_rmse_mean": float(np.mean([r["total_rmse"] for r in subset])),
                "changed_labels_mean": float(np.mean([r["changed_labels"] for r in subset])),
                "accepted_updates_mean": float(np.mean([r["accepted_updates"] for r in subset])),
                "harmful_accepted_mean": float(np.mean([r["harmful_accepted_updates"] for r in subset])),
            }
        )
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--feedback-fraction", type=float, default=1.0)
    parser.add_argument("--eval-scope", choices=["full", "private"], default="full")
    parser.add_argument("--feedback-noise", type=float, default=0.0)
    parser.add_argument("--flip-feedback-prob", type=float, default=0.0)
    parser.add_argument("--out-prefix", default="offline_feedback_simulation_v1")
    args = parser.parse_args()

    ids, y = load_track1()
    all_rows = []
    split_reports = []
    for i in range(args.splits):
        report = run_split(
            ids,
            y,
            args.seed + i * 17,
            args.test_fraction,
            BUDGETS,
            feedback_noise=args.feedback_noise,
            flip_feedback_prob=args.flip_feedback_prob,
            feedback_fraction=args.feedback_fraction,
            eval_scope=args.eval_scope,
        )
        all_rows.extend(report["rows"])
        split_reports.append(report)
        print(f"completed split {i + 1}/{args.splits}", flush=True)

    summary = summarize(all_rows)
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    row_csv = OUT_DIR / f"{args.out_prefix}_rows.csv"
    summary_csv = OUT_DIR / f"{args.out_prefix}_summary.csv"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    json_path.write_text(
        json.dumps({"config": vars(args), "budgets": BUDGETS, "methods": METHODS, "summary": summary, "splits": split_reports}, indent=2),
        encoding="utf-8",
    )
    write_csv(row_csv, all_rows)
    write_csv(summary_csv, summary)
    lines = [
        "# Offline Feedback Simulation v1",
        "",
        f"- Splits: `{args.splits}`",
        f"- Test fraction: `{args.test_fraction}`",
        f"- Feedback fraction: `{args.feedback_fraction}`",
        f"- Eval scope: `{args.eval_scope}`",
        f"- Feedback noise: `{args.feedback_noise}`",
        f"- Flip feedback probability: `{args.flip_feedback_prob}`",
        f"- Budgets: `{BUDGETS}`",
        "",
        "## Summary",
        "",
        "| method | budget | score | gain | item acc | total rmse | changed | harmful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['budget']} | {row['score_mean']:.5f} +/- {row['score_std']:.5f} | "
            f"{row['gain_mean']:+.5f} | {row['item_accuracy_mean']:.5f} | "
            f"{row['total_rmse_mean']:.3f} | {row['changed_labels_mean']:.2f} | "
            f"{row['harmful_accepted_mean']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "summary_csv": str(summary_csv), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
