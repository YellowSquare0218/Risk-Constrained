import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
THRESHOLD_GRID = np.round(np.linspace(0.34, 0.66, 17), 3)
METHODS = [
    "initial",
    "llp_prevalence_projection",
    "llp_feedback_projection",
    "llp_constrained_sparse_gate",
    "aggregate_threshold_feedback",
    "full_risk_gate",
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prevalence_calibration(prob, train_prevalence):
    pred = np.zeros_like(prob, dtype=int)
    n = prob.shape[0]
    for j in range(prob.shape[1]):
        k = int(round(float(train_prevalence[j]) * n))
        if k <= 0:
            continue
        order = np.argsort(prob[:, j])[::-1]
        pred[order[:k], j] = 1
    return pred


def llp_projection_targets(prob, train_prevalence):
    """Deterministic marginal targets for LLP-style projection baselines."""
    prob_prevalence = np.clip(prob.mean(axis=0), 0.02, 0.98)
    train_prevalence = np.clip(np.asarray(train_prevalence, dtype=float), 0.02, 0.98)
    targets = []

    def add(name, target):
        target = np.clip(np.asarray(target, dtype=float), 0.02, 0.98)
        key = tuple(np.round(target, 4))
        if key not in {row[2] for row in targets}:
            targets.append((name, target, key))

    add("train_prevalence", train_prevalence)
    add("probability_prevalence", prob_prevalence)
    for alpha in [0.25, 0.5, 0.75, -0.25, 1.25]:
        add(f"blend_{alpha:+.2f}", (1.0 - alpha) * train_prevalence + alpha * prob_prevalence)
    for shift in [-0.12, -0.08, -0.04, 0.04, 0.08, 0.12]:
        add(f"global_shift_{shift:+.2f}", train_prevalence + shift)
    for scale in [0.70, 0.85, 1.15, 1.30]:
        add(f"global_scale_{scale:.2f}", train_prevalence * scale)
    for alpha in [0.35, 0.65]:
        # A conservative target between train prevalence and the model's own marginal output.
        add(f"conservative_blend_{alpha:.2f}", alpha * train_prevalence + (1.0 - alpha) * prob_prevalence)
    return [(name, target) for name, target, _ in targets]


def run_llp_feedback_projection(
    y_true,
    initial_pred,
    prob,
    train_prevalence,
    budget,
    rng,
    feedback_mask,
    eval_mask,
    feedback_noise=0.0,
    flip_feedback_prob=0.0,
):
    pred = initial_pred.copy()
    trace = []
    seen_predictions = {pred.tobytes()}
    candidates = llp_projection_targets(prob, train_prevalence)
    step = 0
    for name, target in candidates:
        if step >= budget:
            break
        probe = prevalence_calibration(prob, target)
        key = probe.tobytes()
        if key in seen_predictions:
            continue
        seen_predictions.add(key)
        step += 1
        before_score = sim.track1_score(y_true[feedback_mask], pred[feedback_mask])
        after_score = sim.track1_score(y_true[feedback_mask], probe[feedback_mask])
        delta = after_score - before_score
        before_eval_score = sim.track1_score(y_true[eval_mask], pred[eval_mask])
        after_eval_score = sim.track1_score(y_true[eval_mask], probe[eval_mask])
        eval_delta = after_eval_score - before_eval_score
        observed_delta = delta + float(rng.normal(0.0, feedback_noise))
        if float(rng.random()) < flip_feedback_prob:
            observed_delta = -observed_delta
        accepted = observed_delta > 0.0
        if accepted:
            pred = probe
        trace.append(
            {
                "step": step,
                "target": name,
                "target_mean_prevalence": float(np.mean(target)),
                "aggregate_delta": float(delta),
                "evaluation_delta": float(eval_delta),
                "observed_delta": float(observed_delta),
                "accepted": bool(accepted),
                "harmful_accept": bool(accepted and eval_delta <= 0.0),
                "changed_labels": int((probe != initial_pred).sum()),
            }
        )
    return pred, trace


def flip_marginal_support(pred, flip, targets):
    current_prevalence = pred.mean(axis=0)
    direction = 1.0 if int(flip["after"]) == 1 else -1.0
    col_idx = int(flip["col_idx"])
    return max(float(direction * (target[col_idx] - current_prevalence[col_idx])) for _, target in targets)


def run_llp_constrained_sparse_gate(
    y_true,
    initial_pred,
    prob,
    train_prevalence,
    budget,
    rng,
    feedback_mask,
    eval_mask,
    feedback_noise=0.0,
    flip_feedback_prob=0.0,
):
    pred = initial_pred.copy()
    trace = []
    used = set()
    targets = llp_projection_targets(prob, train_prevalence)
    for step in range(1, budget + 1):
        ranked = []
        for flip in sim.candidate_pool(prob, pred):
            key = (int(flip["row_idx"]), int(flip["col_idx"]), int(flip["after"]))
            if key in used:
                continue
            if not sim.gate_allows("full_risk_gate", flip):
                continue
            marginal_support = flip_marginal_support(pred, flip, targets)
            if marginal_support <= 0.0:
                continue
            ranked.append((marginal_support, float(flip["rank_full"]), float(flip["p_after"]), flip))
        if not ranked:
            break
        _, marginal_rank, _, flip = max(ranked, key=lambda row: row[:3])
        used.add((int(flip["row_idx"]), int(flip["col_idx"]), int(flip["after"])))
        before_score = sim.track1_score(y_true[feedback_mask], pred[feedback_mask])
        probe = sim.apply_flip(pred, flip)
        after_score = sim.track1_score(y_true[feedback_mask], probe[feedback_mask])
        delta = after_score - before_score
        before_eval_score = sim.track1_score(y_true[eval_mask], pred[eval_mask])
        after_eval_score = sim.track1_score(y_true[eval_mask], probe[eval_mask])
        eval_delta = after_eval_score - before_eval_score
        observed_delta = delta + float(rng.normal(0.0, feedback_noise))
        if float(rng.random()) < flip_feedback_prob:
            observed_delta = -observed_delta
        accepted = observed_delta > 0.0
        if accepted:
            pred = probe
        trace.append(
            {
                "step": step,
                "row_idx": int(flip["row_idx"]),
                "column": str(flip["column"]),
                "before": int(flip["before"]),
                "after": int(flip["after"]),
                "p_after": round(float(flip["p_after"]), 4),
                "total_gap": round(float(flip["total_gap"]), 4),
                "total_agrees": bool(flip["total_agrees"]),
                "marginal_support": round(float(flip_marginal_support(pred, flip, targets)), 6),
                "marginal_rank": round(float(marginal_rank), 6),
                "aggregate_delta": float(delta),
                "evaluation_delta": float(eval_delta),
                "observed_delta": float(observed_delta),
                "accepted": bool(accepted),
                "harmful_accept": bool(accepted and eval_delta <= 0.0),
                "changed_labels": int((probe != initial_pred).sum()),
            }
        )
    return pred, trace


def run_aggregate_threshold_feedback(y_true, initial_pred, prob, budget, rng, feedback_mask, eval_mask, feedback_noise=0.0, flip_feedback_prob=0.0):
    pred = initial_pred.copy()
    trace = []
    used = set()
    ranked = sorted(THRESHOLD_GRID, key=lambda x: abs(float(x) - 0.5))
    for step, threshold in enumerate(ranked[:budget], 1):
        if float(threshold) in used:
            continue
        used.add(float(threshold))
        probe = (prob >= float(threshold)).astype(int)
        before_score = sim.track1_score(y_true[feedback_mask], pred[feedback_mask])
        after_score = sim.track1_score(y_true[feedback_mask], probe[feedback_mask])
        delta = after_score - before_score
        before_eval_score = sim.track1_score(y_true[eval_mask], pred[eval_mask])
        after_eval_score = sim.track1_score(y_true[eval_mask], probe[eval_mask])
        eval_delta = after_eval_score - before_eval_score
        observed_delta = delta + float(rng.normal(0.0, feedback_noise))
        if float(rng.random()) < flip_feedback_prob:
            observed_delta = -observed_delta
        accepted = observed_delta > 0.0
        if accepted:
            pred = probe
        trace.append(
            {
                "step": step,
                "threshold": float(threshold),
                "aggregate_delta": float(delta),
                "evaluation_delta": float(eval_delta),
                "observed_delta": float(observed_delta),
                "accepted": bool(accepted),
                "harmful_accept": bool(accepted and eval_delta <= 0.0),
                "changed_labels": int((probe != initial_pred).sum()),
            }
        )
    return pred, trace


def run_split(ids, y, seed, test_fraction, budget, args):
    rng = np.random.default_rng(seed)
    order = np.asarray(ids).copy()
    rng.shuffle(order)
    n_test = max(8, int(round(len(ids) * test_fraction)))
    test_ids = set(map(int, order[:n_test]))
    train_mask = np.asarray([int(pid) not in test_ids for pid in ids], dtype=bool)
    test_mask = ~train_mask

    x = sim.load_features(ids, train_mask)
    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    prob = sim.predict_probabilities(x_train, y_train, x_test)
    initial_pred = (prob >= 0.5).astype(int)
    feedback_mask, eval_mask = sim.make_feedback_eval_masks(len(y_test), rng, args.feedback_fraction, args.eval_scope)
    initial_score = sim.track1_score(y_test[eval_mask], initial_pred[eval_mask])

    rows = []
    traces = {}
    for method in METHODS:
        if method == "initial":
            pred = initial_pred.copy()
            trace = []
        elif method == "llp_prevalence_projection":
            pred = prevalence_calibration(prob, y_train.mean(axis=0))
            trace = []
        elif method == "llp_feedback_projection":
            method_rng = np.random.default_rng(seed + 1777)
            pred, trace = run_llp_feedback_projection(
                y_test,
                initial_pred,
                prob,
                y_train.mean(axis=0),
                budget,
                method_rng,
                feedback_mask,
                eval_mask,
                feedback_noise=args.feedback_noise,
                flip_feedback_prob=args.flip_feedback_prob,
            )
        elif method == "llp_constrained_sparse_gate":
            method_rng = np.random.default_rng(seed + 2111)
            pred, trace = run_llp_constrained_sparse_gate(
                y_test,
                initial_pred,
                prob,
                y_train.mean(axis=0),
                budget,
                method_rng,
                feedback_mask,
                eval_mask,
                feedback_noise=args.feedback_noise,
                flip_feedback_prob=args.flip_feedback_prob,
            )
        elif method == "aggregate_threshold_feedback":
            method_rng = np.random.default_rng(seed + 2029)
            pred, trace = run_aggregate_threshold_feedback(
                y_test,
                initial_pred,
                prob,
                budget,
                method_rng,
                feedback_mask,
                eval_mask,
                feedback_noise=args.feedback_noise,
                flip_feedback_prob=args.flip_feedback_prob,
            )
        elif method == "full_risk_gate":
            method_rng = np.random.default_rng(seed + budget * 101 + sim.METHODS.index("full_risk_gate") * 1009)
            pred, trace = sim.run_feedback_method(
                y_test,
                initial_pred,
                prob,
                "full_risk_gate",
                budget,
                method_rng,
                feedback_noise=args.feedback_noise,
                flip_feedback_prob=args.flip_feedback_prob,
                feedback_mask=feedback_mask,
                eval_mask=eval_mask,
            )
        else:
            raise ValueError(method)

        metrics = sim.metric_row(y_test[eval_mask], pred[eval_mask])
        accepted = [row for row in trace if row.get("accepted")]
        metrics.update(
            {
                "seed": seed,
                "method": method,
                    "budget": budget if method in {"llp_feedback_projection", "llp_constrained_sparse_gate", "aggregate_threshold_feedback", "full_risk_gate"} else 0,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_feedback": int(feedback_mask.sum()),
                "n_eval": int(eval_mask.sum()),
                "feedback_fraction": float(feedback_mask.mean()),
                "eval_scope": args.eval_scope,
                "initial_score": initial_score,
                "score_gain": metrics["score"] - initial_score,
                "changed_labels": int((pred != initial_pred).sum()),
                "accepted_updates": len(accepted),
                "harmful_accepted_updates": sum(int(row.get("harmful_accept", False)) for row in accepted),
            }
        )
        rows.append(metrics)
        traces[method] = trace
    return {"seed": seed, "rows": rows, "traces": traces}


def summarize(rows):
    out = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        out.append(
            {
                "method": method,
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
    parser.add_argument("--out-prefix", default="external_adaptation_baselines_v1")
    args = parser.parse_args()

    ids, y = sim.load_track1()
    rows = []
    split_reports = []
    for i in range(args.splits):
        report = run_split(ids, y, args.seed + i * 17, args.test_fraction, args.budget, args)
        rows.extend(report["rows"])
        split_reports.append(report)
        print(f"completed split {i + 1}/{args.splits}", flush=True)

    summary = summarize(rows)
    row_csv = OUT_DIR / f"{args.out_prefix}_rows.csv"
    summary_csv = OUT_DIR / f"{args.out_prefix}_summary.csv"
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    write_csv(row_csv, rows)
    write_csv(summary_csv, summary)
    json_path.write_text(json.dumps({"config": vars(args), "summary": summary, "splits": split_reports}, indent=2), encoding="utf-8")

    lines = [
        "# External Adaptation Baselines",
        "",
        f"- Splits: `{args.splits}`",
        f"- Budget: `{args.budget}`",
        f"- Feedback noise: `{args.feedback_noise}`",
        f"- Flip feedback probability: `{args.flip_feedback_prob}`",
        "",
        "| method | score | gain | item acc | total rmse | changed | accepted | harmful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['score_mean']:.5f} +/- {row['score_std']:.5f} | "
            f"{row['gain_mean']:+.5f} | {row['item_accuracy_mean']:.5f} | "
            f"{row['total_rmse_mean']:.3f} | {row['changed_labels_mean']:.2f} | "
            f"{row['accepted_updates_mean']:.2f} | {row['harmful_accepted_mean']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
