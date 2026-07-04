import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
POSTERIOR_METHOD = "sparse_state_posterior_gate"
METHODS = [
    "initial",
    "physics_prior_only",
    "feedback_only",
    "full_risk_gate",
    POSTERIOR_METHOD,
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def feedback_delta_for_flip(states, true_totals, ref_totals, flip, mutable_index, feedback_mask, base_pred):
    mask_idx = np.flatnonzero(feedback_mask)
    n_mask = max(1, len(mask_idx))
    row_idx = int(flip["row_idx"])
    if not bool(feedback_mask[row_idx]):
        return np.zeros(states.shape[0], dtype=np.float64)

    key = (row_idx, int(flip["col_idx"]))
    state_idx = mutable_index.get(key)
    if state_idx is None:
        true_bit = np.full(states.shape[0], int(base_pred[key]), dtype=np.int8)
    else:
        true_bit = states[:, state_idx]

    after = int(flip["after"])
    item_delta = np.where(after == true_bit, 1.0, -1.0) / float(n_mask * len(sim.TRACK1_COLS))

    row_pos = int(np.where(mask_idx == row_idx)[0][0])
    before_total = ref_totals[row_idx]
    direction = 1 if after == 1 else -1
    diff = ref_totals[mask_idx][None, :] - true_totals[:, mask_idx]
    sse_before = np.sum(diff * diff, axis=1)
    old_error = diff[:, row_pos] ** 2
    new_error = (before_total + direction - true_totals[:, row_idx]) ** 2
    rmse_before = np.sqrt(sse_before / float(n_mask))
    rmse_after = np.sqrt(np.maximum(0.0, sse_before - old_error + new_error) / float(n_mask))
    return 0.5 * (item_delta - (rmse_after - rmse_before) / 34.0)


def build_mutable_positions(prob, pred, history, max_mutable):
    pool = sim.ordered_pool(sim.candidate_pool(prob, pred), "full_risk_gate")
    selected = []
    seen = set()
    for item in history:
        flip = item["flip"]
        key = (int(flip["row_idx"]), int(flip["col_idx"]))
        if key not in seen:
            seen.add(key)
            selected.append(key)
    for flip in pool:
        key = (int(flip["row_idx"]), int(flip["col_idx"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(key)
        if len(selected) >= max_mutable:
            break
    return selected


def sample_states(prob, base_pred, mutable_positions, rng, num_states):
    probs = np.asarray([prob[row, col] for row, col in mutable_positions], dtype=np.float64)
    probs = np.clip(probs, 0.04, 0.96)
    states = (rng.random((num_states, len(mutable_positions))) < probs[None, :]).astype(np.int8)
    if num_states > 0:
        states[0] = np.asarray([base_pred[row, col] for row, col in mutable_positions], dtype=np.int8)
    if num_states > 1:
        states[1] = (probs >= 0.5).astype(np.int8)
    if num_states > 2:
        uncertain = np.abs(probs - 0.5) < 0.18
        states[2] = states[0]
        flips = (rng.random(len(mutable_positions)) < 0.35) & uncertain
        states[2, flips] = 1 - states[2, flips]
    return states, probs


def true_totals_from_states(states, base_pred, mutable_positions):
    base_totals = base_pred.sum(axis=1).astype(np.float64)
    if not mutable_positions:
        return np.tile(base_totals[None, :], (states.shape[0], 1))
    rows = np.asarray([row for row, _ in mutable_positions], dtype=np.int32)
    base_bits = np.asarray([base_pred[row, col] for row, col in mutable_positions], dtype=np.float64)
    out = np.tile(base_totals[None, :], (states.shape[0], 1))
    for s_idx in range(states.shape[0]):
        out[s_idx] += np.bincount(
            rows,
            weights=states[s_idx].astype(np.float64) - base_bits,
            minlength=base_pred.shape[0],
        )
    return out


def posterior_weights(states, probs, true_totals, history, mutable_index, feedback_mask, base_pred, sigma, prior_weight, sparse_weight):
    if not history:
        return np.full(states.shape[0], 1.0 / max(1, states.shape[0]), dtype=np.float64)
    loss = np.zeros(states.shape[0], dtype=np.float64)
    for item in history:
        pred_delta = feedback_delta_for_flip(
            states,
            true_totals,
            item["ref_totals"],
            item["flip"],
            mutable_index,
            feedback_mask,
            base_pred,
        )
        loss += ((pred_delta - float(item["observed_delta"])) / sigma) ** 2
    prior_loss = np.sum((states.astype(np.float64) - probs[None, :]) ** 2, axis=1)
    base_bits = states[0].astype(np.int8)
    sparse_loss = np.sum(states != base_bits[None, :], axis=1)
    log_w = -0.5 * loss - prior_weight * prior_loss - sparse_weight * sparse_loss
    log_w -= float(np.max(log_w))
    w = np.exp(log_w)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(states.shape[0], 1.0 / max(1, states.shape[0]), dtype=np.float64)
    return w / total


def select_posterior_flip(
    prob,
    pred,
    base_pred,
    history,
    used,
    feedback_mask,
    rng,
    max_mutable,
    num_states,
    candidate_limit,
    sigma,
    prior_weight,
    sparse_weight,
    min_expected_delta,
):
    pool = sim.ordered_pool(sim.candidate_pool(prob, pred), "full_risk_gate")
    if not history:
        return next(
            (
                row
                for row in pool
                if (row["row_idx"], row["col_idx"], row["after"]) not in used and sim.gate_allows("full_risk_gate", row)
            ),
            None,
        )

    mutable_positions = build_mutable_positions(prob, pred, history, max_mutable)
    mutable_index = {pos: idx for idx, pos in enumerate(mutable_positions)}
    states, probs = sample_states(prob, base_pred, mutable_positions, rng, num_states)
    true_totals = true_totals_from_states(states, base_pred, mutable_positions)
    weights = posterior_weights(
        states,
        probs,
        true_totals,
        history,
        mutable_index,
        feedback_mask,
        base_pred,
        sigma,
        prior_weight,
        sparse_weight,
    )

    scored = []
    ref_totals = pred.sum(axis=1).astype(np.float64)
    for flip in pool[:candidate_limit]:
        key = (int(flip["row_idx"]), int(flip["col_idx"]), int(flip["after"]))
        if key in used or not sim.gate_allows("full_risk_gate", flip):
            continue
        if (int(flip["row_idx"]), int(flip["col_idx"])) not in mutable_index:
            continue
        deltas = feedback_delta_for_flip(states, true_totals, ref_totals, flip, mutable_index, feedback_mask, base_pred)
        expected = float(np.dot(weights, deltas))
        positive = float(np.dot(weights, (deltas > 0.0).astype(np.float64)))
        scored.append((expected, positive, float(flip["rank_full"]), flip))
    if not scored:
        return None
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    if scored[0][0] < min_expected_delta:
        fallback = next(
            (
                row
                for row in pool
                if (row["row_idx"], row["col_idx"], row["after"]) not in used and sim.gate_allows("full_risk_gate", row)
            ),
            None,
        )
        return fallback
    flip = dict(scored[0][3])
    flip["posterior_expected_delta"] = round(scored[0][0], 6)
    flip["posterior_positive_fraction"] = round(scored[0][1], 4)
    return flip


def run_sparse_state_posterior(
    y_true,
    initial_pred,
    prob,
    max_budget,
    rng,
    feedback_noise,
    flip_feedback_prob,
    feedback_mask,
    eval_mask,
    args,
):
    pred = initial_pred.copy()
    history = []
    trace = []
    used = set()
    for step in range(1, max_budget + 1):
        flip = select_posterior_flip(
            prob,
            pred,
            initial_pred,
            history,
            used,
            feedback_mask,
            rng,
            args.max_mutable,
            args.num_states,
            args.candidate_limit,
            args.posterior_sigma,
            args.prior_weight,
            args.sparse_weight,
            args.min_expected_delta,
        )
        if flip is None:
            break
        used.add((int(flip["row_idx"]), int(flip["col_idx"]), int(flip["after"])))
        ref_totals = pred.sum(axis=1).astype(np.float64)
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
        history.append({"ref_totals": ref_totals, "flip": dict(flip), "observed_delta": float(observed_delta)})
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
                "posterior_expected_delta": flip.get("posterior_expected_delta"),
                "posterior_positive_fraction": flip.get("posterior_positive_fraction"),
                "aggregate_delta": round(float(delta), 6),
                "evaluation_delta": round(float(eval_delta), 6),
                "observed_delta": round(float(observed_delta), 6),
                "accepted": bool(accepted),
                "harmful_accept": bool(accepted and eval_delta <= 0),
            }
        )
    return pred, trace


def run_split(ids, y, seed, test_fraction, budgets, args):
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
        for budget in budgets:
            if method == "initial":
                pred = initial_pred.copy()
                trace = []
            elif method == "physics_prior_only":
                pred = sim.run_physics_only(initial_pred, prob, budget)
                trace = []
            elif method in {"feedback_only", "full_risk_gate"}:
                method_rng = np.random.default_rng(seed + budget * 101 + sim.METHODS.index(method) * 1009)
                pred, trace = sim.run_feedback_method(
                    y_test,
                    initial_pred,
                    prob,
                    method,
                    budget,
                    method_rng,
                    feedback_noise=args.feedback_noise,
                    flip_feedback_prob=args.flip_feedback_prob,
                    feedback_mask=feedback_mask,
                    eval_mask=eval_mask,
                )
            else:
                method_rng = np.random.default_rng(seed + budget * 101 + 7919)
                pred, trace = run_sparse_state_posterior(
                    y_test,
                    initial_pred,
                    prob,
                    budget,
                    method_rng,
                    args.feedback_noise,
                    args.flip_feedback_prob,
                    feedback_mask,
                    eval_mask,
                    args,
                )
            metrics = sim.metric_row(y_test[eval_mask], pred[eval_mask])
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
                    "eval_scope": args.eval_scope,
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
    for method, budget in sorted({(row["method"], row["budget"]) for row in rows}):
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


def parse_budgets(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026053101)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--budgets", default="0,20")
    parser.add_argument("--feedback-fraction", type=float, default=1.0)
    parser.add_argument("--eval-scope", choices=["full", "private"], default="full")
    parser.add_argument("--feedback-noise", type=float, default=0.0)
    parser.add_argument("--flip-feedback-prob", type=float, default=0.0)
    parser.add_argument("--max-mutible", "--max-mutable", dest="max_mutable", type=int, default=220)
    parser.add_argument("--num-states", type=int, default=5000)
    parser.add_argument("--candidate-limit", type=int, default=260)
    parser.add_argument("--posterior-sigma", type=float, default=0.0012)
    parser.add_argument("--prior-weight", type=float, default=0.03)
    parser.add_argument("--sparse-weight", type=float, default=0.002)
    parser.add_argument("--min-expected-delta", type=float, default=-0.00005)
    parser.add_argument("--out-prefix", default="offline_sparse_state_posterior_v1")
    args = parser.parse_args()
    budgets = parse_budgets(args.budgets)

    ids, y = sim.load_track1()
    all_rows = []
    split_reports = []
    for i in range(args.splits):
        report = run_split(ids, y, args.seed + i * 17, args.test_fraction, budgets, args)
        all_rows.extend(report["rows"])
        split_reports.append(report)
        print(f"completed split {i + 1}/{args.splits}", flush=True)
    summary = summarize(all_rows)

    json_path = OUT_DIR / f"{args.out_prefix}.json"
    row_csv = OUT_DIR / f"{args.out_prefix}_rows.csv"
    summary_csv = OUT_DIR / f"{args.out_prefix}_summary.csv"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    json_path.write_text(
        json.dumps(
            {"config": vars(args), "budgets": budgets, "methods": METHODS, "summary": summary, "splits": split_reports},
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(row_csv, all_rows)
    write_csv(summary_csv, summary)
    lines = [
        "# Offline Sparse-State Posterior Variant",
        "",
        f"- Splits: `{args.splits}`",
        f"- Seed: `{args.seed}`",
        f"- Budgets: `{budgets}`",
        f"- Feedback noise: `{args.feedback_noise}`",
        f"- Flip feedback probability: `{args.flip_feedback_prob}`",
        f"- Mutable positions per step: `{args.max_mutable}`",
        f"- Posterior states: `{args.num_states}`",
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
