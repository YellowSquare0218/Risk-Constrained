import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge

import run_offline_feedback_simulation as sim


OUT_DIR = Path("outputs")
PREDICTORS = ["knn_prior", "ridge_regression", "extra_trees", "logistic_l2"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ridge_probabilities(x_train, y_train, x_test):
    model = Ridge(alpha=2.0)
    model.fit(x_train, y_train)
    raw = model.predict(x_test)
    return np.clip(raw, 0.02, 0.98)


def extra_trees_probabilities(x_train, y_train, x_test, seed):
    model = ExtraTreesClassifier(
        n_estimators=180,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prob_list = model.predict_proba(x_test)
    cols = []
    for j, proba in enumerate(prob_list):
        classes = list(model.classes_[j])
        if 1 in classes:
            cols.append(proba[:, classes.index(1)])
        else:
            cols.append(np.full(x_test.shape[0], float(y_train[:, j].mean())))
    return np.clip(np.vstack(cols).T, 0.02, 0.98)


def logistic_probabilities(x_train, y_train, x_test):
    cols = []
    for j in range(y_train.shape[1]):
        yj = y_train[:, j]
        if len(np.unique(yj)) < 2:
            cols.append(np.full(x_test.shape[0], float(yj.mean())))
            continue
        model = LogisticRegression(
            C=0.5,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=1000 + j,
        )
        model.fit(x_train, yj)
        cols.append(model.predict_proba(x_test)[:, 1])
    return np.clip(np.vstack(cols).T, 0.02, 0.98)


def predict_probabilities(name, x_train, y_train, x_test, seed):
    if name == "knn_prior":
        return sim.predict_probabilities(x_train, y_train, x_test)
    if name == "ridge_regression":
        return ridge_probabilities(x_train, y_train, x_test)
    if name == "extra_trees":
        return extra_trees_probabilities(x_train, y_train, x_test, seed)
    if name == "logistic_l2":
        return logistic_probabilities(x_train, y_train, x_test)
    raise ValueError(f"Unknown predictor: {name}")


def run_split(ids, y, seed, test_fraction, budget, predictor, args):
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
    prob = predict_probabilities(predictor, x_train, y_train, x_test, seed)
    initial_pred = (prob >= 0.5).astype(int)
    feedback_mask, eval_mask = sim.make_feedback_eval_masks(len(y_test), rng, args.feedback_fraction, args.eval_scope)
    initial_metrics = sim.metric_row(y_test[eval_mask], initial_pred[eval_mask])
    method_rng = np.random.default_rng(seed + budget * 101 + sim.METHODS.index("full_risk_gate") * 1009)
    refined_pred, trace = sim.run_feedback_method(
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
    refined_metrics = sim.metric_row(y_test[eval_mask], refined_pred[eval_mask])
    accepted = [row for row in trace if row.get("accepted")]
    return {
        "seed": seed,
        "predictor": predictor,
        "budget": budget,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "initial_score": float(initial_metrics["score"]),
        "initial_item_accuracy": float(initial_metrics["item_accuracy"]),
        "initial_total_rmse": float(initial_metrics["total_rmse"]),
        "refined_score": float(refined_metrics["score"]),
        "refined_item_accuracy": float(refined_metrics["item_accuracy"]),
        "refined_total_rmse": float(refined_metrics["total_rmse"]),
        "score_gain": float(refined_metrics["score"] - initial_metrics["score"]),
        "changed_labels": int((refined_pred != initial_pred).sum()),
        "accepted_updates": len(accepted),
        "harmful_accepted_updates": sum(int(row.get("harmful_accept", False)) for row in accepted),
    }


def summarize(rows):
    out = []
    for predictor in PREDICTORS:
        subset = [row for row in rows if row["predictor"] == predictor]
        out.append(
            {
                "predictor": predictor,
                "splits": len(subset),
                "initial_score_mean": float(np.mean([r["initial_score"] for r in subset])),
                "initial_score_std": float(np.std([r["initial_score"] for r in subset])),
                "refined_score_mean": float(np.mean([r["refined_score"] for r in subset])),
                "refined_score_std": float(np.std([r["refined_score"] for r in subset])),
                "gain_mean": float(np.mean([r["score_gain"] for r in subset])),
                "gain_std": float(np.std([r["score_gain"] for r in subset])),
                "refined_item_accuracy_mean": float(np.mean([r["refined_item_accuracy"] for r in subset])),
                "refined_total_rmse_mean": float(np.mean([r["refined_total_rmse"] for r in subset])),
                "changed_labels_mean": float(np.mean([r["changed_labels"] for r in subset])),
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
    parser.add_argument("--out-prefix", default="initial_predictor_robustness_v1")
    args = parser.parse_args()

    base_ids, y = sim.load_track1()
    rows = []
    for predictor in PREDICTORS:
        ids = base_ids.copy()
        for i in range(args.splits):
            row = run_split(ids, y, args.seed + i * 17, args.test_fraction, args.budget, predictor, args)
            rows.append(row)
            print(f"completed {predictor} split {i + 1}/{args.splits}", flush=True)

    summary = summarize(rows)
    row_csv = OUT_DIR / f"{args.out_prefix}_rows.csv"
    summary_csv = OUT_DIR / f"{args.out_prefix}_summary.csv"
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    write_csv(row_csv, rows)
    write_csv(summary_csv, summary)
    json_path.write_text(json.dumps({"config": vars(args), "predictors": PREDICTORS, "summary": summary}, indent=2), encoding="utf-8")
    lines = [
        "# Initial Predictor Robustness",
        "",
        f"- Splits: `{args.splits}`",
        f"- Budget: `{args.budget}`",
        f"- Feedback noise: `{args.feedback_noise}`",
        "",
        "| initial predictor | initial score | refined score | gain | item acc | total rmse | changed | harmful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['predictor']} | {row['initial_score_mean']:.5f} +/- {row['initial_score_std']:.5f} | "
            f"{row['refined_score_mean']:.5f} +/- {row['refined_score_std']:.5f} | {row['gain_mean']:+.5f} | "
            f"{row['refined_item_accuracy_mean']:.5f} | {row['refined_total_rmse_mean']:.3f} | "
            f"{row['changed_labels_mean']:.2f} | {row['harmful_accepted_mean']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
