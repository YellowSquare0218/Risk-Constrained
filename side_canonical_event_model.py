import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import gait_event_phase_features as event_phase
import physics_gait_features as physics
import pose_feature_baseline as base


TRACK1_COLUMNS = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def canonical_name(name, side):
    if side == "left":
        name = re.sub(r"(^|_)left(_|$)", r"\1self\2", name)
        name = re.sub(r"(^|_)right(_|$)", r"\1opp\2", name)
    else:
        name = re.sub(r"(^|_)right(_|$)", r"\1self\2", name)
        name = re.sub(r"(^|_)left(_|$)", r"\1opp\2", name)
    return name


def should_use_feature(name):
    # Keep side-local and global quality/count features. Raw patient-level left-right
    # signed gaps are noisy after canonicalization, but absolute/correlation features
    # still carry useful symmetry information.
    if "_gap_x" in name or "_gap_y" in name or "signed_mean" in name:
        return False
    return True


def build_side_matrix(patient_ids, feature_names, X):
    feature_names = list(feature_names)
    used = [idx for idx, name in enumerate(feature_names) if should_use_feature(name)]
    canon_names = sorted({canonical_name(feature_names[idx], side) for idx in used for side in ["left", "right"]})
    canon_index = {name: idx for idx, name in enumerate(canon_names)}

    rows = []
    row_patient_ids = []
    row_sides = []
    for row_idx, pid in enumerate(patient_ids):
        for side in ["left", "right"]:
            values = np.full(len(canon_names) + 2, np.nan, dtype=np.float32)
            for feat_idx in used:
                cname = canonical_name(feature_names[feat_idx], side)
                values[canon_index[cname]] = X[row_idx, feat_idx]
            values[-2] = 1.0 if side == "right" else 0.0
            values[-1] = float(int(pid) / 110.0)
            rows.append(values)
            row_patient_ids.append(int(pid))
            row_sides.append(side)
    full_names = canon_names + ["side_is_right", "patient_id_scaled"]
    return np.vstack(rows), np.asarray(row_patient_ids, dtype=np.int32), row_sides, full_names


def side_targets(track1_rows):
    patient_ids = []
    y = []
    sides = []
    for row in track1_rows:
        pid = int(row["patient_id"])
        for side_name, side_key in [("left", "left"), ("right", "right")]:
            patient_ids.append(pid)
            sides.append(side_name)
            y.append([int(row[side_key][str(i)]) for i in range(1, 18)])
    return np.asarray(patient_ids, dtype=np.int32), sides, np.asarray(y, dtype=np.float32)


def make_pipeline(model_name, seed, k=6000):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=700,
            random_state=seed,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    elif model_name == "extra":
        clf = ExtraTreesClassifier(
            n_estimators=700,
            random_state=seed,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        raise ValueError(model_name)
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold()),
            ("select", physics.TotalScoreSelectKBest(k=k)),
            ("clf", MultiOutputClassifier(clf, n_jobs=1)),
        ]
    )


def sklearn_probs(pipe, X):
    return physics.sklearn_probs(pipe, X)


def patient_prob_from_side(side_patient_ids, side_names, side_prob, patient_order):
    out = np.full((len(patient_order), 34), np.nan, dtype=np.float32)
    index = {int(pid): idx for idx, pid in enumerate(patient_order)}
    for row_idx, (pid, side) in enumerate(zip(side_patient_ids, side_names)):
        patient_idx = index[int(pid)]
        if side == "left":
            out[patient_idx, :17] = side_prob[row_idx]
        else:
            out[patient_idx, 17:] = side_prob[row_idx]
    return out


def grouped_patient_folds(patient_ids, n_splits=5, seed=2026):
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(map(int, patient_ids))), dtype=np.int32)
    rng.shuffle(unique)
    folds = [set(map(int, unique[i::n_splits])) for i in range(n_splits)]
    for fold in folds:
        valid = np.asarray([int(pid) in fold for pid in patient_ids], dtype=bool)
        train = ~valid
        yield train, valid


def write_track1_submission(prob, thresholds, path):
    pred = (prob >= thresholds).astype(int)
    track1_pred = {int(pid): pred[i].tolist() for i, pid in enumerate(base.TRACK1_TEST_IDS)}
    base.write_submission_csv(track1_pred, physics.old_track2_perfect_predictions(), path)
    base.validate_submission_csv(path)


def read_current_best_matrix(path):
    rows_by_id = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_id[row["ID"]] = row
    matrix = []
    for pid in base.TRACK1_TEST_IDS:
        row = rows_by_id[f"track1-{pid}"]
        matrix.append([int(row[col]) for col in TRACK1_COLUMNS])
    return np.asarray(matrix, dtype=np.int8)


def rank_flips_from_current(current_best_path, prob, thresholds):
    current = read_current_best_matrix(current_best_path)
    rows = []
    for patient_idx, pid in enumerate(base.TRACK1_TEST_IDS):
        for item_idx, col in enumerate(TRACK1_COLUMNS):
            current_value = int(current[patient_idx, item_idx])
            p = float(prob[patient_idx, item_idx])
            threshold = float(thresholds[item_idx])
            if current_value:
                margin = threshold - p
                flip_to = 0
            else:
                margin = p - threshold
                flip_to = 1
            rows.append(
                {
                    "patient_id": int(pid),
                    "column": col,
                    "current": current_value,
                    "flip_to": flip_to,
                    "prob": round(p, 6),
                    "threshold": round(threshold, 6),
                    "margin": round(float(margin), 6),
                }
            )
    rows.sort(key=lambda r: (-r["margin"], r["patient_id"], r["column"]))
    return rows


def evaluate_and_write(args):
    out_dir = Path(args.out_dir)
    patient_ids, feature_names, X_all = event_phase.load_all_features(
        args.dataset_dir, out_dir, args.workers, force_event=False
    )
    X_side_all, side_all_patient_ids, side_all_names, side_feature_names = build_side_matrix(patient_ids, feature_names, X_all)
    print(
        f"side-canonical features: {X_side_all.shape[0]} sides x {X_side_all.shape[1]} features",
        flush=True,
    )

    all_side_index = {(int(pid), side): idx for idx, (pid, side) in enumerate(zip(side_all_patient_ids, side_all_names))}
    track1_rows = event_phase.load_json(args.track1_train)
    train_patient_ids, y_train_patient = base.track1_targets(track1_rows)
    side_train_patient_ids, side_train_names, y_side = side_targets(track1_rows)
    train_side_rows = [all_side_index[(int(pid), side)] for pid, side in zip(side_train_patient_ids, side_train_names)]
    X_side_train = X_side_all[train_side_rows]

    test_side_patient_ids = []
    test_side_names = []
    test_side_rows = []
    for pid in base.TRACK1_TEST_IDS:
        for side in ["left", "right"]:
            test_side_patient_ids.append(int(pid))
            test_side_names.append(side)
            test_side_rows.append(all_side_index[(int(pid), side)])
    X_side_test = X_side_all[test_side_rows]

    oof = {}
    test_prob = {}
    for model_name, seed in [("rf", 14300), ("extra", 15300)]:
        side_oof = np.zeros_like(y_side, dtype=np.float32)
        for fold, (train_mask, valid_mask) in enumerate(grouped_patient_folds(side_train_patient_ids), 1):
            pipe = make_pipeline(model_name, seed + fold)
            pipe.fit(X_side_train[train_mask], y_side[train_mask].astype(int))
            side_oof[valid_mask] = sklearn_probs(pipe, X_side_train[valid_mask])
            print(f"{model_name} side fold {fold}/5", flush=True)
        pipe = make_pipeline(model_name, seed + 99)
        pipe.fit(X_side_train, y_side.astype(int))
        oof[model_name] = patient_prob_from_side(side_train_patient_ids, side_train_names, side_oof, train_patient_ids)
        side_test_prob = sklearn_probs(pipe, X_side_test)
        test_prob[model_name] = patient_prob_from_side(test_side_patient_ids, test_side_names, side_test_prob, base.TRACK1_TEST_IDS)

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
            global_scores[f"{threshold:.2f}"] = float(
                base.track1_score(y_train_patient, (data["oof"] >= threshold).astype(np.float32))
            )
        best_threshold = float(max(global_scores, key=global_scores.get))
        item_thresholds, item_score = physics.optimize_item_thresholds(y_train_patient, data["oof"], start=best_threshold)
        global_path = out_dir / f"candidate_sidecanonical_{name}_global_v19.csv"
        item_path = out_dir / f"candidate_sidecanonical_{name}_item_v19.csv"
        write_track1_submission(data["test"], best_threshold, global_path)
        write_track1_submission(data["test"], item_thresholds, item_path)
        results[name] = {
            "best_global_threshold": best_threshold,
            "best_global_score": float(global_scores[f"{best_threshold:.2f}"]),
            "item_score": float(item_score),
            "item_thresholds": item_thresholds.tolist(),
            "global_csv": str(global_path),
            "item_csv": str(item_path),
        }
        print(name, results[name], flush=True)

    best_name = max(results, key=lambda key: results[key]["item_score"])
    best_thresholds = np.asarray(results[best_name]["item_thresholds"], dtype=np.float32)
    flip_scan = rank_flips_from_current(args.current_best, candidates[best_name]["test"], best_thresholds)
    report = {
        "feature_shape": [int(X_side_all.shape[0]), int(X_side_all.shape[1])],
        "feature_count": len(side_feature_names),
        "patient_grouped_cv": True,
        "results": results,
        "best_item_model": best_name,
        "top_flips_from_current_best": flip_scan[:50],
        "note": (
            "Side-canonical model converts left/right features into self/opposite coordinates "
            "and evaluates OOF with patient-grouped folds. Track2 labels are fixed to the public-perfect surface."
        ),
    }
    dump_json(report, out_dir / "side_canonical_event_report.json")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--track1-train", default="track1_train.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--current-best", default="outputs/candidate_after_v12_track1_72_l16_down_v13.csv")
    args = parser.parse_args()
    evaluate_and_write(args)


if __name__ == "__main__":
    main()
