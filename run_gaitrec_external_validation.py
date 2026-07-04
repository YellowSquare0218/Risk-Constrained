import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge


REQUIRED_FILES = [
    "GRF_metadata.csv",
    "GRF_F_AP_PRO_left.csv",
    "GRF_F_AP_PRO_right.csv",
    "GRF_F_ML_PRO_left.csv",
    "GRF_F_ML_PRO_right.csv",
    "GRF_F_V_PRO_left.csv",
    "GRF_F_V_PRO_right.csv",
]
METHODS = ["initial", "feature_prior_only", "feedback_only", "full_risk_gate"]


def canonical_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_id_columns(df):
    lower = {c.lower(): c for c in df.columns}
    candidates = [
        ["subject_id", "session_id", "trial_id"],
        ["subject_id", "session_id"],
        ["subject", "session", "trial"],
        ["subject", "session"],
        ["id", "session", "trial"],
        ["id", "session"],
    ]
    for names in candidates:
        if all(name in lower for name in names):
            return [lower[name] for name in names]
    first_three = list(df.columns[:3])
    if len(first_three) < 3:
        raise ValueError("Could not infer GaitRec identifier columns")
    return first_three


def load_signal_features(path):
    df = canonical_columns(pd.read_csv(path))
    id_cols = find_id_columns(df)
    value_cols = [c for c in df.columns if c not in id_cols]
    values = df[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    first = values[:, : values.shape[1] // 2]
    second = values[:, values.shape[1] // 2 :]
    feats = np.column_stack(
        [
            values.mean(axis=1),
            values.std(axis=1),
            values.min(axis=1),
            values.max(axis=1),
            np.abs(values).max(axis=1),
            first.mean(axis=1),
            second.mean(axis=1),
            first.std(axis=1),
            second.std(axis=1),
        ]
    )
    prefix = path.stem
    feat_cols = [f"{prefix}_{name}" for name in ["mean", "std", "min", "max", "absmax", "first_mean", "second_mean", "first_std", "second_std"]]
    out = df[id_cols].copy()
    for idx, col in enumerate(feat_cols):
        out[col] = feats[:, idx]
    return out, id_cols


def identify_label_columns(meta):
    lower = {c.lower(): c for c in meta.columns}
    class_col = lower.get("class_label") or lower.get("label") or lower.get("class")
    detailed_col = lower.get("class_label_detailed") or lower.get("diagnosis") or lower.get("diagnosis_label")
    side_col = lower.get("affected_side") or lower.get("side")
    if class_col is None and detailed_col is None:
        raise ValueError("Could not find class/diagnosis label in GaitRec metadata")
    return class_col, detailed_col, side_col


def build_labels(meta):
    class_col, detailed_col, side_col = identify_label_columns(meta)
    y_cols = []
    labels = []
    if class_col is not None:
        class_text = meta[class_col].astype(str).str.lower()
        impaired = (~class_text.str.contains("healthy|control|hc|normal", regex=True)).astype(int)
        labels.append(impaired.to_numpy())
        y_cols.append("impaired")
    if side_col is not None:
        side = meta[side_col].astype(str).str.lower()
        numeric_side = pd.to_numeric(meta[side_col], errors="coerce")
        if numeric_side.notna().any():
            side_specs = [
                ("affected_side_0", numeric_side.eq(0)),
                ("affected_side_1", numeric_side.eq(1)),
                ("bilateral", numeric_side.eq(2)),
            ]
        else:
            side_specs = [
                ("left_affected", side.str.contains("left|l", regex=True)),
                ("right_affected", side.str.contains("right|r", regex=True)),
                ("bilateral", side.str.contains("bilateral|both|b", regex=True)),
            ]
        for name, mask in side_specs:
            labels.append(mask.astype(int).to_numpy())
            y_cols.append(name)
    if detailed_col is not None:
        detailed = meta[detailed_col].astype(str).fillna("unknown")
        counts = detailed.value_counts()
        top = [v for v in counts.index.tolist() if str(v).lower() not in {"nan", "none", "unknown"}][:6]
        for value in top:
            labels.append((detailed == value).astype(int).to_numpy())
            y_cols.append(f"dx_{value}".replace(" ", "_").replace("/", "_"))
    y = np.vstack(labels).T.astype(int)
    keep = (y.mean(axis=0) > 0.02) & (y.mean(axis=0) < 0.98)
    return y[:, keep], [c for c, k in zip(y_cols, keep) if k]


def load_gaitrec(data_dir):
    data_dir = Path(data_dir)
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing GaitRec files in {data_dir}: {missing}")

    meta = canonical_columns(pd.read_csv(data_dir / "GRF_metadata.csv"))
    meta_id_cols = find_id_columns(meta)
    signal_merged = None
    signal_id_cols = None
    for name in REQUIRED_FILES[1:]:
        feats, id_cols = load_signal_features(data_dir / name)
        if signal_merged is None:
            signal_merged = feats
            signal_id_cols = id_cols
        else:
            common_signal_ids = [col for col in signal_id_cols if col in id_cols]
            if not common_signal_ids:
                raise ValueError(f"No common signal identifiers for {name}")
            signal_merged = signal_merged.merge(feats, on=common_signal_ids, how="inner")
    common_meta_ids = [col for col in meta_id_cols if col in signal_merged.columns]
    if not common_meta_ids:
        raise ValueError("No common metadata/signal identifiers for GaitRec merge")
    signal_feature_cols = [c for c in signal_merged.columns if c.startswith("GRF_")]
    session_features = signal_merged.groupby(common_meta_ids, as_index=False)[signal_feature_cols].agg(["mean", "std"])
    session_features.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in session_features.columns
    ]
    session_features = session_features.fillna(0.0)
    merged = session_features.merge(meta, on=common_meta_ids, how="inner")
    y, y_cols = build_labels(merged)
    feature_cols = [c for c in merged.columns if c.startswith("GRF_")]
    x = merged[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    subject = merged[meta_id_cols[0]].astype(int).to_numpy()
    return subject, x, y, y_cols


def standardize(x_train, x_test):
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def predict_probabilities(x_train, y_train, x_test):
    model = Ridge(alpha=2.0)
    model.fit(x_train, y_train)
    raw = model.predict(x_test)
    prevalence = y_train.mean(axis=0)
    return np.clip(0.85 * raw + 0.15 * prevalence[None, :], 0.02, 0.98)


def score(y_true, y_pred):
    item_acc = float((y_true == y_pred).mean())
    true_total = y_true.sum(axis=1)
    pred_total = y_pred.sum(axis=1)
    rmse = float(np.sqrt(np.mean((true_total - pred_total) ** 2)))
    return 0.5 * (item_acc + 1.0 - rmse / max(1.0, y_true.shape[1]))


def metric_row(y_true, y_pred):
    true_total = y_true.sum(axis=1)
    pred_total = y_pred.sum(axis=1)
    return {
        "score": score(y_true, y_pred),
        "item_accuracy": float((y_true == y_pred).mean()),
        "total_rmse": float(np.sqrt(np.mean((true_total - pred_total) ** 2))),
    }


def candidate_pool(prob, pred):
    pool = []
    pred_total = pred.sum(axis=1)
    prob_total = prob.sum(axis=1)
    for row_idx in range(pred.shape[0]):
        total_gap = float(prob_total[row_idx] - pred_total[row_idx])
        for col_idx in range(pred.shape[1]):
            before = int(pred[row_idx, col_idx])
            after = 1 - before
            p_after = float(prob[row_idx, col_idx] if after == 1 else 1.0 - prob[row_idx, col_idx])
            direction = 1 if after == 1 else -1
            total_agrees = direction * total_gap > 0.20
            pool.append(
                {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "before": before,
                    "after": after,
                    "p_after": p_after,
                    "total_agrees": bool(total_agrees),
                    "rank_feedback": abs(prob[row_idx, col_idx] - 0.5),
                    "rank_full": 0.80 * p_after + (0.22 if total_agrees else -0.02) - 0.015 * abs(total_gap),
                }
            )
    return pool


def ordered_pool(pool, method):
    if method == "feedback_only":
        return sorted(pool, key=lambda r: (r["rank_feedback"], r["p_after"]), reverse=True)
    return sorted(pool, key=lambda r: (r["rank_full"], r["p_after"], int(r["total_agrees"])), reverse=True)


def gate_allows(method, flip):
    if method == "feedback_only":
        return True
    if method == "full_risk_gate":
        return flip["p_after"] >= 0.42 and (flip["total_agrees"] or flip["p_after"] >= 0.48)
    return False


def apply_flip(pred, flip):
    out = pred.copy()
    out[flip["row_idx"], flip["col_idx"]] = int(flip["after"])
    return out


def run_feature_prior_only(initial_pred, prob, budget):
    pred = initial_pred.copy()
    for flip in ordered_pool(candidate_pool(prob, pred), "full_risk_gate"):
        if budget <= 0:
            break
        if flip["p_after"] >= 0.43 and flip["total_agrees"]:
            pred = apply_flip(pred, flip)
            budget -= 1
    return pred, []


def run_feedback_method(y_true, initial_pred, prob, method, budget, rng, feedback_noise, flip_feedback_prob):
    pred = initial_pred.copy()
    trace = []
    used = set()
    for step in range(1, budget + 1):
        flip = next(
            (
                row
                for row in ordered_pool(candidate_pool(prob, pred), method)
                if (row["row_idx"], row["col_idx"], row["after"]) not in used and gate_allows(method, row)
            ),
            None,
        )
        if flip is None:
            break
        used.add((flip["row_idx"], flip["col_idx"], flip["after"]))
        probe = apply_flip(pred, flip)
        delta = score(y_true, probe) - score(y_true, pred)
        observed = delta + float(rng.normal(0.0, feedback_noise))
        if float(rng.random()) < flip_feedback_prob:
            observed = -observed
        accepted = observed > 0.0
        if accepted:
            pred = probe
        trace.append({"accepted": accepted, "harmful_accept": bool(accepted and delta <= 0.0)})
    return pred, trace


def run_split(subject, x, y, seed, test_fraction, budget, args):
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject)
    rng.shuffle(unique_subjects)
    n_test = max(10, int(round(len(unique_subjects) * test_fraction)))
    test_subjects = set(map(int, unique_subjects[:n_test]))
    train_mask = np.asarray([int(s) not in test_subjects for s in subject], dtype=bool)
    test_mask = ~train_mask
    x_train, x_test = standardize(x[train_mask], x[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]
    prob = predict_probabilities(x_train, y_train, x_test)
    initial_pred = (prob >= 0.5).astype(int)
    initial_score = score(y_test, initial_pred)
    rows = []
    for method in METHODS:
        if method == "initial":
            pred, trace = initial_pred.copy(), []
        elif method == "feature_prior_only":
            pred, trace = run_feature_prior_only(initial_pred, prob, budget)
        else:
            method_rng = np.random.default_rng(seed + budget * 101 + METHODS.index(method) * 1009)
            pred, trace = run_feedback_method(
                y_test,
                initial_pred,
                prob,
                method,
                budget,
                method_rng,
                args.feedback_noise,
                args.flip_feedback_prob,
            )
        metrics = metric_row(y_test, pred)
        accepted = [row for row in trace if row.get("accepted")]
        metrics.update(
            {
                "seed": seed,
                "method": method,
                "initial_score": initial_score,
                "score_gain": metrics["score"] - initial_score,
                "changed_labels": int((pred != initial_pred).sum()),
                "accepted_updates": len(accepted),
                "harmful_accepted_updates": sum(int(row.get("harmful_accept", False)) for row in accepted),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )
        rows.append(metrics)
    return rows


def summarize(rows):
    out = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
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
                "harmful_accepted_mean": float(np.mean([r["harmful_accepted_updates"] for r in subset])),
            }
        )
    return out


def paired_tests(rows):
    df = pd.DataFrame(rows)
    wide = df.pivot(index="seed", columns="method", values="score")
    comparisons = [
        ("full_risk_gate", "initial"),
        ("full_risk_gate", "feedback_only"),
        ("full_risk_gate", "feature_prior_only"),
        ("feedback_only", "initial"),
        ("feature_prior_only", "initial"),
    ]
    out = []
    for left, right in comparisons:
        diff = (wide[left] - wide[right]).to_numpy(dtype=float)
        n = int(diff.size)
        mean = float(diff.mean())
        sem = float(stats.sem(diff)) if n > 1 else 0.0
        if n > 1 and sem > 0:
            margin = float(stats.t.ppf(0.975, n - 1) * sem)
        else:
            margin = 0.0
        p_value = float(stats.ttest_1samp(diff, 0.0).pvalue) if n > 1 else float("nan")
        out.append(
            {
                "comparison": f"{left} - {right}",
                "mean_delta": mean,
                "ci_low": mean - margin,
                "ci_high": mean + margin,
                "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()),
                "p_value": p_value,
            }
        )
    return out


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/gaitrec")
    parser.add_argument("--splits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--feedback-noise", type=float, default=0.0)
    parser.add_argument("--flip-feedback-prob", type=float, default=0.0)
    parser.add_argument("--out-prefix", default="gaitrec_external_validation_v1")
    args = parser.parse_args()

    subject, x, y, y_cols = load_gaitrec(args.data_dir)
    rows = []
    for i in range(args.splits):
        rows.extend(run_split(subject, x, y, args.seed + i * 17, args.test_fraction, args.budget, args))
        print(f"completed split {i + 1}/{args.splits}", flush=True)
    summary = summarize(rows)
    paired = paired_tests(rows)
    out_dir = Path("outputs")
    row_csv = out_dir / f"{args.out_prefix}_rows.csv"
    summary_csv = out_dir / f"{args.out_prefix}_summary.csv"
    paired_csv = out_dir / f"{args.out_prefix}_paired_tests.csv"
    json_path = out_dir / f"{args.out_prefix}.json"
    md_path = out_dir / f"{args.out_prefix}.md"
    write_csv(row_csv, rows)
    write_csv(summary_csv, summary)
    write_csv(paired_csv, paired)
    json_path.write_text(json.dumps({"config": vars(args), "labels": y_cols, "summary": summary, "paired_tests": paired}, indent=2), encoding="utf-8")
    lines = [
        "# GaitRec External Validation",
        "",
        f"- Labels: `{y_cols}`",
        f"- Splits: `{args.splits}`",
        f"- Budget: `{args.budget}`",
        f"- Feedback noise: `{args.feedback_noise}`",
        f"- Flip feedback probability: `{args.flip_feedback_prob}`",
        "",
        "| method | score | gain | item acc | total rmse | changed | harmful |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['score_mean']:.5f} +/- {row['score_std']:.5f} | "
            f"{row['gain_mean']:+.5f} | {row['item_accuracy_mean']:.5f} | "
            f"{row['total_rmse_mean']:.3f} | {row['changed_labels_mean']:.2f} | "
            f"{row['harmful_accepted_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired score tests",
            "",
            "| comparison | mean delta | 95% CI | wins/losses | p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        p_text = "<1e-4" if row["p_value"] < 1e-4 else f"{row['p_value']:.4f}"
        lines.append(
            f"| {row['comparison']} | {row['mean_delta']:+.6f} | "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"{row['wins']}/{row['losses']} | {p_text} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
