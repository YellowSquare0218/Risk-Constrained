import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import gait_event_phase_features as event_phase
import physics_gait_features as physics
import pose_feature_baseline as base


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_total_pipeline(seed, k=5200):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_regression, k=k)),
            (
                "reg",
                ExtraTreesRegressor(
                    n_estimators=700,
                    random_state=seed,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def topk_from_scores(scores, totals):
    pred = np.zeros_like(scores, dtype=np.float32)
    for row_idx, total in enumerate(totals):
        k = int(np.clip(round(float(total)), 0, scores.shape[1]))
        if k <= 0:
            continue
        order = np.argsort(-scores[row_idx])[:k]
        pred[row_idx, order] = 1.0
    return pred


def calibrate_total_constrained(y_true, prob, total_pred, item_thresholds):
    threshold_score = prob - item_thresholds[None, :]
    base_counts = (prob >= item_thresholds[None, :]).sum(axis=1).astype(np.float32)
    expected_counts = prob.sum(axis=1).astype(np.float32)
    best = {
        "score": -1.0,
        "alpha": None,
        "beta": None,
        "offset": None,
        "mode": None,
        "pred": None,
    }
    modes = {
        "reg_base": (total_pred, base_counts),
        "reg_expect": (total_pred, expected_counts),
    }
    for mode, (a_source, b_source) in modes.items():
        for alpha in np.arange(0.0, 1.01, 0.10):
            blended = alpha * a_source + (1.0 - alpha) * b_source
            for offset in np.arange(-3.0, 3.01, 0.5):
                totals = np.clip(blended + offset, 0.0, y_true.shape[1])
                pred = topk_from_scores(threshold_score, totals)
                score = base.track1_score(y_true, pred)
                if score > best["score"] + 1e-12:
                    best = {
                        "score": float(score),
                        "alpha": float(alpha),
                        "beta": float(1.0 - alpha),
                        "offset": float(offset),
                        "mode": mode,
                        "pred": pred,
                    }
    return best


def apply_total_constrained(prob, total_pred, item_thresholds, params):
    threshold_score = prob - item_thresholds[None, :]
    base_counts = (prob >= item_thresholds[None, :]).sum(axis=1).astype(np.float32)
    expected_counts = prob.sum(axis=1).astype(np.float32)
    if params["mode"] == "reg_base":
        other = base_counts
    elif params["mode"] == "reg_expect":
        other = expected_counts
    else:
        raise ValueError(params["mode"])
    totals = params["alpha"] * total_pred + params["beta"] * other + params["offset"]
    return topk_from_scores(threshold_score, totals).astype(int)


def write_submission_from_binary(pred, path):
    track1_pred = {int(pid): pred[i].tolist() for i, pid in enumerate(base.TRACK1_TEST_IDS)}
    base.write_submission_csv(track1_pred, physics.old_track2_perfect_predictions(), path)
    base.validate_submission_csv(path)


def evaluate_and_write(args):
    from sklearn.model_selection import KFold

    out_dir = Path(args.out_dir)
    patient_ids, feature_names, X_all = event_phase.load_all_features(
        args.dataset_dir, out_dir, args.workers, force_event=False
    )
    print(f"total-calibration features: {X_all.shape[0]} patients x {X_all.shape[1]} features", flush=True)

    track1_rows = event_phase.load_json(args.track1_train)
    train_ids, y_train = base.track1_targets(track1_rows)
    X_train = base.make_feature_matrix(train_ids, patient_ids, X_all)
    X_test = base.make_feature_matrix(base.TRACK1_TEST_IDS, patient_ids, X_all)
    X_train = np.hstack([X_train, (train_ids[:, None] / 110.0).astype(np.float32)])
    X_test = np.hstack([X_test, (np.asarray(base.TRACK1_TEST_IDS)[:, None] / 110.0).astype(np.float32)])
    y_total = y_train.sum(axis=1).astype(np.float32)

    oof = {}
    test_prob = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)
    for model_name, seed in [("rf", 11300), ("extra", 12300)]:
        oof_prob = np.zeros_like(y_train, dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
            pipe = event_phase.make_pipeline(model_name, seed + fold)
            pipe.fit(X_train[train_idx], y_train[train_idx].astype(int))
            oof_prob[valid_idx] = event_phase.sklearn_probs(pipe, X_train[valid_idx])
            print(f"{model_name} class fold {fold}/5", flush=True)
        pipe = event_phase.make_pipeline(model_name, seed + 99)
        pipe.fit(X_train, y_train.astype(int))
        oof[model_name] = oof_prob
        test_prob[model_name] = event_phase.sklearn_probs(pipe, X_test)

    oof_total = np.zeros(y_train.shape[0], dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        reg = make_total_pipeline(13300 + fold)
        reg.fit(X_train[train_idx], y_total[train_idx])
        oof_total[valid_idx] = np.clip(reg.predict(X_train[valid_idx]), 0.0, y_train.shape[1])
        print(f"total reg fold {fold}/5", flush=True)
    reg = make_total_pipeline(13399)
    reg.fit(X_train, y_total)
    test_total = np.clip(reg.predict(X_test), 0.0, y_train.shape[1]).astype(np.float32)

    candidates = {}
    for rf_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        name = f"blend_rf{int(rf_weight * 100):03d}"
        candidates[name] = {
            "oof": rf_weight * oof["rf"] + (1.0 - rf_weight) * oof["extra"],
            "test": rf_weight * test_prob["rf"] + (1.0 - rf_weight) * test_prob["extra"],
        }

    results = {}
    for name, data in candidates.items():
        global_scores = {}
        for threshold in np.arange(0.30, 0.56, 0.01):
            global_scores[f"{threshold:.2f}"] = float(base.track1_score(y_train, (data["oof"] >= threshold).astype(np.float32)))
        best_threshold = float(max(global_scores, key=global_scores.get))
        item_thresholds, item_score = physics.optimize_item_thresholds(y_train, data["oof"], start=best_threshold)
        total_best = calibrate_total_constrained(y_train, data["oof"], oof_total, item_thresholds)
        test_pred = apply_total_constrained(data["test"], test_total, item_thresholds, total_best)
        path = out_dir / f"candidate_eventphase_total_{name}_v18.csv"
        write_submission_from_binary(test_pred, path)
        results[name] = {
            "best_global_threshold": best_threshold,
            "best_global_score": float(global_scores[f"{best_threshold:.2f}"]),
            "item_score": float(item_score),
            "total_constrained_score": float(total_best["score"]),
            "total_params": {
                "mode": total_best["mode"],
                "alpha": total_best["alpha"],
                "beta": total_best["beta"],
                "offset": total_best["offset"],
            },
            "candidate_csv": str(path),
            "test_total_pred": [float(x) for x in test_total],
            "test_total_selected": [int(x) for x in test_pred.sum(axis=1)],
        }
        print(name, results[name], flush=True)

    report = {
        "feature_shape": [int(X_all.shape[0]), int(X_all.shape[1])],
        "feature_count": len(feature_names),
        "total_reg_oof_rmse": float(np.sqrt(np.mean((oof_total - y_total) ** 2))),
        "results": results,
        "note": "Total-constrained decoding chooses K items per patient from event-phase probabilities; Track2 uses known perfect public labels.",
    }
    dump_json(report, out_dir / "event_phase_total_calibration_report.json")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--track1-train", default="track1_train.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()
    evaluate_and_write(args)


if __name__ == "__main__":
    main()
