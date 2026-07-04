import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import pose_feature_baseline as base


TEMPORAL_VARS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "left_shank_angle",
    "right_shank_angle",
    "left_thigh_angle",
    "right_thigh_angle",
    "left_foot_angle",
    "right_foot_angle",
    "left_ankle_x",
    "right_ankle_x",
    "left_ankle_y",
    "right_ankle_y",
    "left_knee_y",
    "right_knee_y",
    "left_hip_y",
    "right_hip_y",
    "left_toe_y",
    "right_toe_y",
    "left_heel_y",
    "right_heel_y",
    "ankle_x_gap",
    "ankle_y_gap",
    "knee_y_gap",
    "hip_y_gap",
    "toe_y_gap",
    "heel_y_gap",
    "left_step_reach",
    "right_step_reach",
    "foot_clearance_gap",
    "left_leg_extension",
    "right_leg_extension",
    "leg_extension_gap",
]
PHASE_POINTS = 16
FFT_BINS = 6


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def point_angle(points, a, b, c):
    return base.safe_angle(points, a, b, c)


def segment_angle(points, a, b):
    delta = points[b] - points[a]
    return math.atan2(float(delta[1]), float(delta[0])) / math.pi


def dist(points, a, b):
    return float(np.linalg.norm(points[a] - points[b]))


def temporal_values(points):
    left_leg_extension = dist(points, 11, 15)
    right_leg_extension = dist(points, 12, 16)
    vals = {
        "left_knee_angle": point_angle(points, 11, 13, 15),
        "right_knee_angle": point_angle(points, 12, 14, 16),
        "left_hip_angle": point_angle(points, 5, 11, 13),
        "right_hip_angle": point_angle(points, 6, 12, 14),
        "left_ankle_angle": point_angle(points, 13, 15, 17),
        "right_ankle_angle": point_angle(points, 14, 16, 20),
        "left_shank_angle": segment_angle(points, 13, 15),
        "right_shank_angle": segment_angle(points, 14, 16),
        "left_thigh_angle": segment_angle(points, 11, 13),
        "right_thigh_angle": segment_angle(points, 12, 14),
        "left_foot_angle": segment_angle(points, 19, 17),
        "right_foot_angle": segment_angle(points, 22, 20),
        "left_ankle_x": float(points[15, 0]),
        "right_ankle_x": float(points[16, 0]),
        "left_ankle_y": float(points[15, 1]),
        "right_ankle_y": float(points[16, 1]),
        "left_knee_y": float(points[13, 1]),
        "right_knee_y": float(points[14, 1]),
        "left_hip_y": float(points[11, 1]),
        "right_hip_y": float(points[12, 1]),
        "left_toe_y": float(points[17, 1]),
        "right_toe_y": float(points[20, 1]),
        "left_heel_y": float(points[19, 1]),
        "right_heel_y": float(points[22, 1]),
        "ankle_x_gap": float(points[15, 0] - points[16, 0]),
        "ankle_y_gap": float(points[15, 1] - points[16, 1]),
        "knee_y_gap": float(points[13, 1] - points[14, 1]),
        "hip_y_gap": float(points[11, 1] - points[12, 1]),
        "toe_y_gap": float(points[17, 1] - points[20, 1]),
        "heel_y_gap": float(points[19, 1] - points[22, 1]),
        "left_step_reach": float(points[17, 0] - points[11, 0]),
        "right_step_reach": float(points[20, 0] - points[12, 0]),
        "foot_clearance_gap": float(min(points[17, 1], points[19, 1]) - min(points[20, 1], points[22, 1])),
        "left_leg_extension": left_leg_extension,
        "right_leg_extension": right_leg_extension,
        "leg_extension_gap": left_leg_extension - right_leg_extension,
    }
    return np.asarray([vals[name] for name in TEMPORAL_VARS], dtype=np.float32)


def autocorr_features(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size < 6 or float(np.std(x)) < 1e-7:
        return 0.0, 0.0
    z = x - float(np.mean(x))
    denom = float(np.dot(z, z)) + 1e-7
    max_lag = min(80, x.size - 2)
    values = []
    for lag in range(2, max_lag + 1):
        values.append(float(np.dot(z[:-lag], z[lag:]) / denom))
    if not values:
        return 0.0, 0.0
    best_idx = int(np.argmax(values))
    return values[best_idx], (best_idx + 2) / max_lag


def series_features(x):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    dx = np.diff(x) if x.size > 1 else np.asarray([0.0], dtype=np.float32)
    resampled = np.interp(np.linspace(0, max(x.size - 1, 0), PHASE_POINTS), np.arange(x.size), x)
    centered = resampled - float(np.mean(resampled))
    fft = np.abs(np.fft.rfft(centered))
    fft = fft[1 : FFT_BINS + 1]
    fft_sum = float(fft.sum()) + 1e-7
    ac_max, ac_lag = autocorr_features(x)

    out = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "range": float(np.percentile(x, 90) - np.percentile(x, 10)),
        "vel_mean": float(np.mean(dx)),
        "vel_std": float(np.std(dx)),
        "vel_abs_p90": float(np.percentile(np.abs(dx), 90)),
        "autocorr_max": ac_max,
        "autocorr_lag": ac_lag,
    }
    for i, value in enumerate(resampled):
        out[f"phase_{i:02d}"] = float(value)
    for i, value in enumerate(fft):
        out[f"fft_{i + 1}"] = float(value / fft_sum)
    return out


def video_temporal_features(matrix):
    features = {}
    if matrix.size == 0:
        return features
    for col, var_name in enumerate(TEMPORAL_VARS):
        for stat, value in series_features(matrix[:, col]).items():
            features[f"{var_name}_{stat}"] = value
    if matrix.shape[0] > 1:
        left = matrix[:, TEMPORAL_VARS.index("left_ankle_y")]
        right = matrix[:, TEMPORAL_VARS.index("right_ankle_y")]
        features["ankle_y_corr"] = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-7 and np.std(right) > 1e-7 else 0.0
        left = matrix[:, TEMPORAL_VARS.index("left_step_reach")]
        right = matrix[:, TEMPORAL_VARS.index("right_step_reach")]
        features["step_reach_corr"] = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-7 and np.std(right) > 1e-7 else 0.0
    return features


def combine_video_features(prefix, feature_dicts):
    out = {f"{prefix}_video_count": float(len(feature_dicts))}
    if not feature_dicts:
        return out
    names = sorted({name for d in feature_dicts for name in d})
    for name in names:
        values = np.asarray([d.get(name, np.nan) for d in feature_dicts], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            out[f"{prefix}_mean_{name}"] = float(np.mean(values))
            out[f"{prefix}_std_{name}"] = float(np.std(values))
    return out


def patient_temporal_features(patient_dir):
    patient_dir = Path(patient_dir)
    patient_id = int(patient_dir.name)
    by_view = {view: [] for view in ["forward", "backward", "left", "right", "all"]}
    frame_counts = Counter()

    for video_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
        view = base.infer_view(video_dir.name)
        if view == "unknown":
            continue
        rows = []
        for frame_path in sorted(video_dir.glob("frame_*.json")):
            result = base.read_frame_features(frame_path)
            if result is None:
                continue
            _, pose_row = result
            points = pose_row.reshape(-1, 2)
            rows.append(temporal_values(points))
        if not rows:
            continue
        matrix = np.vstack(rows)
        feats = video_temporal_features(matrix)
        feats["frames"] = float(matrix.shape[0])
        by_view[view].append(feats)
        by_view["all"].append(feats)
        frame_counts[view] += matrix.shape[0]
        frame_counts["all"] += matrix.shape[0]

    features = {"patient_id": patient_id}
    for view, dicts in by_view.items():
        features[f"{view}_frames"] = float(frame_counts[view])
        features.update(combine_video_features(f"{view}_temporal", dicts))
    return patient_id, features


def extract_temporal_features(dataset_dir, cache_path, workers=4, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["patient_ids"], data["feature_names"].tolist(), data["X"]

    patient_dirs = sorted([p for p in Path(dataset_dir).iterdir() if p.is_dir()])
    rows = {}
    if workers <= 1:
        for idx, p in enumerate(patient_dirs, 1):
            pid, feats = patient_temporal_features(p)
            rows[pid] = feats
            if idx % 10 == 0 or idx == len(patient_dirs):
                print(f"temporal extracted {idx}/{len(patient_dirs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(patient_temporal_features, str(p)): p for p in patient_dirs}
            for idx, fut in enumerate(as_completed(futures), 1):
                pid, feats = fut.result()
                rows[pid] = feats
                if idx % 10 == 0 or idx == len(patient_dirs):
                    print(f"temporal extracted {idx}/{len(patient_dirs)}", flush=True)

    patient_ids = np.asarray(sorted(rows), dtype=np.int32)
    feature_names = sorted({name for feats in rows.values() for name in feats if name != "patient_id"})
    X = np.empty((len(patient_ids), len(feature_names)), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        feats = rows[int(pid)]
        X[i] = [feats.get(name, np.nan) for name in feature_names]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, patient_ids=patient_ids, feature_names=np.asarray(feature_names), X=X)
    return patient_ids, feature_names, X


def align_matrix(target_ids, source_ids, X):
    index = {int(pid): i for i, pid in enumerate(source_ids)}
    return X[[index[int(pid)] for pid in target_ids]]


def load_combined_features(dataset_dir, out_dir, workers, force):
    out_dir = Path(out_dir)
    old_cache = out_dir / "pose_features.npz"
    old_ids, old_names, X_old = base.extract_pose_features(dataset_dir, old_cache, workers=workers, force=False)
    temp_ids, temp_names, X_temp = extract_temporal_features(
        dataset_dir, out_dir / "temporal_pose_features.npz", workers=workers, force=force
    )
    common = np.asarray(sorted(set(map(int, old_ids)) & set(map(int, temp_ids))), dtype=np.int32)
    X = np.hstack([align_matrix(common, old_ids, X_old), align_matrix(common, temp_ids, X_temp)])
    feature_names = [f"stat_{name}" for name in old_names] + [f"temp_{name}" for name in temp_names]
    return common, feature_names, X


def track1_score(y_true, prob, threshold):
    pred = (prob >= threshold).astype(np.float32)
    return base.track1_score(y_true, pred)


def sklearn_probs(pipe, X):
    z = pipe[:-1].transform(X)
    probs = []
    for est in pipe.named_steps["clf"].estimators_:
        p = est.predict_proba(z)
        if len(est.classes_) == 1:
            arr = np.full(z.shape[0], float(est.classes_[0]))
        else:
            arr = p[:, list(est.classes_).index(1)]
        probs.append(arr)
    return np.vstack(probs).T


def make_pipeline(model_name, seed):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=900,
            random_state=seed,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    elif model_name == "extra":
        clf = ExtraTreesClassifier(
            n_estimators=900,
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
            ("clf", MultiOutputClassifier(clf, n_jobs=1)),
        ]
    )


def evaluate_track1_models(X, y):
    from sklearn.model_selection import KFold

    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    results = {}
    for model_name in ["rf", "extra"]:
        probs = []
        y_valid = []
        kf = KFold(n_splits=5, shuffle=True, random_state=2026)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
            pipe = make_pipeline(model_name, 3100 + fold)
            pipe.fit(X[train_idx], y[train_idx].astype(int))
            probs.append(sklearn_probs(pipe, X[valid_idx]))
            y_valid.append(y[valid_idx])
            print(f"{model_name} fold {fold}/5", flush=True)
        prob = np.vstack(probs)
        truth = np.vstack(y_valid)
        scores = {str(th): track1_score(truth, prob, th) for th in thresholds}
        results[model_name] = scores
        print(model_name, scores, flush=True)
    return results


def old_track2_predictions(track2_rows):
    left = base.majority_label([row["left"]["gait_subtype"] for row in track2_rows])
    right = base.majority_label([row["right"]["gait_subtype"] for row in track2_rows])
    return {pid: {"left": left, "right": right} for pid in base.TRACK2_TEST_IDS}


def write_submission(track1_prob, track2_pred, path, thresholds):
    for threshold in thresholds:
        pred = (track1_prob >= threshold).astype(int)
        track1_pred = {int(pid): pred[i].tolist() for i, pid in enumerate(base.TRACK1_TEST_IDS)}
        out_path = Path(str(path).replace("{threshold}", f"{int(threshold * 100):03d}"))
        base.write_submission_csv(track1_pred, track2_pred, out_path)
        rows = list(csv.DictReader(out_path.open("r", encoding="utf-8")))
        totals = [int(row["Total"]) for row in rows if row["ID"].startswith("track1-")]
        print(out_path.name, "avg_total", round(sum(totals) / len(totals), 3), "totals", totals, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--track1-train", default="track1_train.json")
    parser.add_argument("--track2-train", default="track2_train.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--force-temporal-features", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    patient_ids, feature_names, X_all = load_combined_features(
        args.dataset_dir, out_dir, args.workers, args.force_temporal_features
    )
    print(f"combined features: {X_all.shape[0]} patients x {X_all.shape[1]} features", flush=True)

    track1_rows = load_json(args.track1_train)
    track2_rows = load_json(args.track2_train)
    train_ids, y_train = base.track1_targets(track1_rows)
    X_train = base.make_feature_matrix(train_ids, patient_ids, X_all)
    X_test = base.make_feature_matrix(base.TRACK1_TEST_IDS, patient_ids, X_all)
    X_train = np.hstack([X_train, (train_ids[:, None] / 110.0).astype(np.float32)])
    X_test = np.hstack([X_test, (np.asarray(base.TRACK1_TEST_IDS)[:, None] / 110.0).astype(np.float32)])

    cv = evaluate_track1_models(X_train, y_train)

    track2_pred = old_track2_predictions(track2_rows)
    candidate_files = []
    for model_name, seed in [("rf", 4201), ("extra", 4202)]:
        pipe = make_pipeline(model_name, seed)
        pipe.fit(X_train, y_train.astype(int))
        prob = sklearn_probs(pipe, X_test)
        pattern = out_dir / f"candidate_adv_{model_name}_t{{threshold}}_oldT2.csv"
        write_submission(prob, track2_pred, pattern, [0.25, 0.30, 0.35, 0.40, 0.45])
        candidate_files.extend(str(out_dir / f"candidate_adv_{model_name}_t{int(t * 100):03d}_oldT2.csv") for t in [0.25, 0.30, 0.35, 0.40, 0.45])

    report = {
        "combined_shape": [int(X_all.shape[0]), int(X_all.shape[1])],
        "feature_count": len(feature_names),
        "track1_cv": cv,
        "candidate_files": candidate_files,
        "sources_used": [
            "Automated EVGS systems: stride/event detection and keypoint filtering",
            "Skeleton gait recognition: multi-scale temporal pose features",
            "Skeleton action recognition: ST-GCN-style spatial-temporal modeling",
        ],
    }
    dump_json(report, out_dir / "advanced_pose_report.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
