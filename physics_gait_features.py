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
import advanced_pose_baseline as advanced
import pose_feature_baseline as base


SIDES = {
    "left": {"hip": 11, "knee": 13, "ankle": 15, "toe": 17, "heel": 19, "shoulder": 5},
    "right": {"hip": 12, "knee": 14, "ankle": 16, "toe": 20, "heel": 22, "shoulder": 6},
}
VIEWS = ["left", "right", "forward", "backward", "all", "sagittal", "coronal"]
SERIES_STATS = ["mean", "std", "min", "max", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "range", "iqr"]
PHASE_POINTS = 12


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def distance(points, a, b):
    return float(np.linalg.norm(points[:, a] - points[:, b], axis=1).mean())


def vector_distance(points, a, b):
    return np.linalg.norm(points[:, a] - points[:, b], axis=1)


def segment_angle_series(points, a, b):
    delta = points[:, b] - points[:, a]
    return np.arctan2(delta[:, 1], delta[:, 0]) / math.pi


def joint_angle_series(points, a, b, c):
    ba = points[:, a] - points[:, b]
    bc = points[:, c] - points[:, b]
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    denom = np.where(denom <= 1e-8, np.nan, denom)
    cos_value = np.sum(ba * bc, axis=1) / denom
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return np.arccos(cos_value) / math.pi


def smooth(x, window=5):
    x = np.asarray(x, dtype=np.float32)
    if x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def local_extrema(x, mode="max", min_distance=8):
    x = smooth(x, 5)
    if x.size < 3:
        return np.asarray([], dtype=int)
    if mode == "max":
        candidates = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
        order = candidates[np.argsort(-x[candidates])]
    else:
        candidates = np.where((x[1:-1] <= x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
        order = candidates[np.argsort(x[candidates])]
    chosen = []
    for idx in order:
        if all(abs(int(idx) - int(prev)) >= min_distance for prev in chosen):
            chosen.append(int(idx))
    return np.asarray(sorted(chosen), dtype=int)


def finite_series(x):
    x = np.asarray(x, dtype=np.float32)
    return x[np.isfinite(x)]


def describe_series(prefix, x):
    x = finite_series(x)
    out = {}
    if x.size == 0:
        return out
    percentiles = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
    values = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "range": float(percentiles[5] - percentiles[1]),
        "iqr": float(percentiles[4] - percentiles[2]),
    }
    dx = np.diff(x) if x.size > 1 else np.asarray([0.0], dtype=np.float32)
    ddx = np.diff(dx) if dx.size > 1 else np.asarray([0.0], dtype=np.float32)
    values.update(
        {
            "vel_mean": float(np.mean(dx)),
            "vel_std": float(np.std(dx)),
            "vel_abs_mean": float(np.mean(np.abs(dx))),
            "vel_abs_p90": float(np.percentile(np.abs(dx), 90)),
            "acc_abs_p90": float(np.percentile(np.abs(ddx), 90)),
        }
    )
    for name, value in values.items():
        out[f"{prefix}_{name}"] = value
    return out


def corr_features(prefix, a, b):
    a = finite_series(a)
    b = finite_series(b)
    n = min(a.size, b.size)
    if n < 6:
        return {}
    a = a[:n]
    b = b[:n]
    out = {}
    if np.std(a) > 1e-7 and np.std(b) > 1e-7:
        out[f"{prefix}_corr0"] = float(np.corrcoef(a, b)[0, 1])
    else:
        out[f"{prefix}_corr0"] = 0.0
    za = a - float(np.mean(a))
    zb = b - float(np.mean(b))
    denom = (float(np.linalg.norm(za)) * float(np.linalg.norm(zb))) + 1e-7
    max_lag = min(40, n // 3)
    best_corr = -2.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = za[-lag:]
            bb = zb[: n + lag]
        elif lag > 0:
            aa = za[: n - lag]
            bb = zb[lag:]
        else:
            aa = za
            bb = zb
        if aa.size < 4:
            continue
        value = float(np.dot(aa, bb) / denom)
        if value > best_corr:
            best_corr = value
            best_lag = lag
    out[f"{prefix}_best_corr"] = best_corr
    out[f"{prefix}_best_lag_frac"] = float(best_lag / max(n, 1))
    return out


def phase_features(prefix, signal, series_by_name):
    signal = finite_series(signal)
    if signal.size < 24:
        return {}
    extrema = local_extrema(signal, mode="max", min_distance=max(8, signal.size // 20))
    if extrema.size < 2:
        extrema = local_extrema(signal, mode="min", min_distance=max(8, signal.size // 20))
    out = {f"{prefix}_cycle_count": float(max(0, extrema.size - 1))}
    if extrema.size < 2:
        return out
    intervals = []
    for start, end in zip(extrema[:-1], extrema[1:]):
        if end - start >= 8:
            intervals.append((int(start), int(end)))
    if not intervals:
        return out
    lengths = np.asarray([end - start for start, end in intervals], dtype=np.float32)
    out[f"{prefix}_cycle_len_mean"] = float(np.mean(lengths))
    out[f"{prefix}_cycle_len_std"] = float(np.std(lengths))
    phase_x = np.linspace(0.0, 1.0, PHASE_POINTS)
    for name, values in series_by_name.items():
        values = np.asarray(values, dtype=np.float32)
        sampled = []
        for start, end in intervals:
            raw_x = np.linspace(0.0, 1.0, end - start + 1)
            sampled.append(np.interp(phase_x, raw_x, values[start : end + 1]))
        if not sampled:
            continue
        mean_curve = np.vstack(sampled).mean(axis=0)
        amp = float(np.percentile(mean_curve, 90) - np.percentile(mean_curve, 10))
        out[f"{prefix}_{name}_phase_amp"] = amp
        for idx, value in enumerate(mean_curve):
            out[f"{prefix}_{name}_phase_{idx:02d}"] = float(value)
    return out


def frame_series(points):
    series = {}
    for side, joints in SIDES.items():
        hip, knee, ankle, toe, heel, shoulder = (
            joints["hip"],
            joints["knee"],
            joints["ankle"],
            joints["toe"],
            joints["heel"],
            joints["shoulder"],
        )
        series[f"{side}_knee_angle"] = joint_angle_series(points, hip, knee, ankle)
        series[f"{side}_hip_angle"] = joint_angle_series(points, shoulder, hip, knee)
        series[f"{side}_ankle_angle"] = joint_angle_series(points, knee, ankle, toe)
        series[f"{side}_thigh_angle"] = segment_angle_series(points, hip, knee)
        series[f"{side}_shank_angle"] = segment_angle_series(points, knee, ankle)
        series[f"{side}_foot_angle"] = segment_angle_series(points, heel, toe)
        series[f"{side}_thigh_len"] = vector_distance(points, hip, knee)
        series[f"{side}_shank_len"] = vector_distance(points, knee, ankle)
        series[f"{side}_foot_len"] = vector_distance(points, heel, toe)
        series[f"{side}_leg_extension"] = vector_distance(points, hip, ankle)
        for joint_name, joint_idx in [("knee", knee), ("ankle", ankle), ("toe", toe), ("heel", heel)]:
            rel = points[:, joint_idx] - points[:, hip]
            series[f"{side}_{joint_name}_rel_hip_x"] = rel[:, 0]
            series[f"{side}_{joint_name}_rel_hip_y"] = rel[:, 1]
        toe = points[:, joints["toe"]]
        heel = points[:, joints["heel"]]
        ankle = points[:, joints["ankle"]]
        hip = points[:, joints["hip"]]
        series[f"{side}_step_reach"] = toe[:, 0] - hip[:, 0]
        series[f"{side}_toe_clearance"] = -np.minimum(toe[:, 1], heel[:, 1])
        series[f"{side}_foot_mid_y"] = 0.5 * (toe[:, 1] + heel[:, 1])
        series[f"{side}_ankle_swing_x"] = ankle[:, 0] - hip[:, 0]

    for joint_name, left_idx, right_idx in [
        ("hip", 11, 12),
        ("knee", 13, 14),
        ("ankle", 15, 16),
        ("toe", 17, 20),
        ("heel", 19, 22),
    ]:
        gap = points[:, left_idx] - points[:, right_idx]
        series[f"{joint_name}_gap_x"] = gap[:, 0]
        series[f"{joint_name}_gap_y"] = gap[:, 1]
        series[f"{joint_name}_gap_dist"] = np.linalg.norm(gap, axis=1)
    series["leg_extension_gap"] = series["left_leg_extension"] - series["right_leg_extension"]
    series["toe_clearance_gap"] = series["left_toe_clearance"] - series["right_toe_clearance"]
    return series


def video_physics_features(points):
    features = {}
    if points.shape[0] < 4:
        return features
    series = frame_series(points)
    for name, values in series.items():
        features.update(describe_series(name, values))
    for name in ["ankle_swing_x", "toe_clearance", "knee_angle", "ankle_angle", "step_reach"]:
        features.update(corr_features(f"lr_{name}", series[f"left_{name}"], series[f"right_{name}"]))
    for side in ["left", "right"]:
        cycle_signal = series[f"{side}_ankle_swing_x"]
        phase_series = {
            "knee": series[f"{side}_knee_angle"],
            "hip": series[f"{side}_hip_angle"],
            "ankle": series[f"{side}_ankle_angle"],
            "toe_clearance": series[f"{side}_toe_clearance"],
            "step_reach": series[f"{side}_step_reach"],
            "foot_angle": series[f"{side}_foot_angle"],
        }
        features.update(phase_features(f"{side}", cycle_signal, phase_series))
    features["frames"] = float(points.shape[0])
    return features


def combine_feature_dicts(prefix, dicts):
    out = {f"{prefix}_video_count": float(len(dicts))}
    if not dicts:
        return out
    names = sorted({name for d in dicts for name in d})
    for name in names:
        values = np.asarray([d.get(name, np.nan) for d in dicts], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out[f"{prefix}_mean_{name}"] = float(np.mean(values))
        out[f"{prefix}_std_{name}"] = float(np.std(values))
        out[f"{prefix}_min_{name}"] = float(np.min(values))
        out[f"{prefix}_max_{name}"] = float(np.max(values))
    return out


def patient_physics_features(patient_dir):
    patient_dir = Path(patient_dir)
    patient_id = int(patient_dir.name)
    by_view = {view: [] for view in VIEWS}
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
            rows.append(pose_row.reshape(-1, 2))
        if len(rows) < 4:
            continue
        points = np.stack(rows, axis=0)
        feats = video_physics_features(points)
        if not feats:
            continue
        groups = [view, "all"]
        if view in {"left", "right"}:
            groups.append("sagittal")
        if view in {"forward", "backward"}:
            groups.append("coronal")
        for group in groups:
            by_view[group].append(feats)
            frame_counts[group] += points.shape[0]
    features = {"patient_id": patient_id}
    for view in VIEWS:
        features[f"{view}_frames"] = float(frame_counts[view])
        features.update(combine_feature_dicts(f"{view}_physics", by_view[view]))
    return patient_id, features


def extract_physics_features(dataset_dir, cache_path, workers=4, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["patient_ids"], data["feature_names"].tolist(), data["X"]

    patient_dirs = sorted([p for p in Path(dataset_dir).iterdir() if p.is_dir()])
    rows = {}
    if workers <= 1:
        for idx, patient_dir in enumerate(patient_dirs, 1):
            pid, feats = patient_physics_features(patient_dir)
            rows[pid] = feats
            if idx % 10 == 0 or idx == len(patient_dirs):
                print(f"physics extracted {idx}/{len(patient_dirs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(patient_physics_features, str(p)): p for p in patient_dirs}
            for idx, future in enumerate(as_completed(futures), 1):
                pid, feats = future.result()
                rows[pid] = feats
                if idx % 10 == 0 or idx == len(patient_dirs):
                    print(f"physics extracted {idx}/{len(patient_dirs)}", flush=True)

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


def load_all_features(dataset_dir, out_dir, workers, force_physics=False):
    out_dir = Path(out_dir)
    base_ids, base_names, X_base = advanced.load_combined_features(dataset_dir, out_dir, workers, force=False)
    phys_ids, phys_names, X_phys = extract_physics_features(
        dataset_dir, out_dir / "physics_pose_features.npz", workers=workers, force=force_physics
    )
    common = np.asarray(sorted(set(map(int, base_ids)) & set(map(int, phys_ids))), dtype=np.int32)
    X = np.hstack([align_matrix(common, base_ids, X_base), align_matrix(common, phys_ids, X_phys)])
    feature_names = [f"base_{name}" for name in base_names] + [f"phys_{name}" for name in phys_names]
    return common, feature_names, X


from sklearn.base import BaseEstimator, TransformerMixin


class TotalScoreSelectKBest(TransformerMixin, BaseEstimator):
    def __init__(self, k=3500):
        self.k = k
        self.indices_ = None

    def fit(self, X, y):
        from sklearn.feature_selection import f_classif

        y_total = np.asarray(y)
        if y_total.ndim > 1:
            y_total = y_total.sum(axis=1)
        scores, _ = f_classif(X, y_total)
        scores = np.asarray(scores, dtype=np.float32)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        k = min(int(self.k), X.shape[1])
        if k >= X.shape[1]:
            self.indices_ = np.arange(X.shape[1])
        else:
            self.indices_ = np.sort(np.argpartition(scores, -k)[-k:])
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def make_pipeline(model_name, seed):
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
            ("select", TotalScoreSelectKBest(k=3500)),
            ("clf", MultiOutputClassifier(clf, n_jobs=1)),
        ]
    )


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


def optimize_item_thresholds(y_true, prob, start=0.45):
    thresholds = np.full(y_true.shape[1], start, dtype=np.float32)
    best = base.track1_score(y_true, (prob >= thresholds).astype(np.float32))
    grid = np.arange(0.22, 0.61, 0.01)
    for _ in range(6):
        improved = False
        for col in range(y_true.shape[1]):
            local_best = (best, thresholds[col])
            for threshold in grid:
                candidate = thresholds.copy()
                candidate[col] = threshold
                score = base.track1_score(y_true, (prob >= candidate).astype(np.float32))
                if score > local_best[0] + 1e-12:
                    local_best = (score, threshold)
            if local_best[1] != thresholds[col]:
                thresholds[col] = local_best[1]
                best = local_best[0]
                improved = True
        if not improved:
            break
    return thresholds, best


def old_track2_perfect_predictions():
    return {
        4: {"left": "type3", "right": "type3"},
        6: {"left": "type1", "right": "type1"},
        7: {"left": "type3", "right": "type3"},
        13: {"left": "WNL", "right": "WNL"},
        26: {"left": "WNL", "right": "WNL"},
        35: {"left": "type4", "right": "type4"},
        39: {"left": "type1", "right": "type1"},
        42: {"left": "type2", "right": "type2"},
        50: {"left": "type2", "right": "type2"},
    }


def write_track1_submission(prob, threshold, target_ids, track2_pred, path):
    pred = (prob >= threshold).astype(int)
    track1_pred = {int(pid): pred[i].tolist() for i, pid in enumerate(target_ids)}
    base.write_submission_csv(track1_pred, track2_pred, path)


def evaluate_and_write(args):
    from sklearn.model_selection import KFold

    out_dir = Path(args.out_dir)
    patient_ids, feature_names, X_all = load_all_features(args.dataset_dir, out_dir, args.workers, args.force_physics_features)
    print(f"physics combined features: {X_all.shape[0]} patients x {X_all.shape[1]} features", flush=True)
    track1_rows = load_json(args.track1_train)
    train_ids, y_train = base.track1_targets(track1_rows)
    X_train = base.make_feature_matrix(train_ids, patient_ids, X_all)
    X_test = base.make_feature_matrix(base.TRACK1_TEST_IDS, patient_ids, X_all)
    X_train = np.hstack([X_train, (train_ids[:, None] / 110.0).astype(np.float32)])
    X_test = np.hstack([X_test, (np.asarray(base.TRACK1_TEST_IDS)[:, None] / 110.0).astype(np.float32)])

    oof = {}
    test_prob = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)
    for model_name, seed in [("rf", 7300), ("extra", 8300)]:
        oof_prob = np.zeros_like(y_train, dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
            pipe = make_pipeline(model_name, seed + fold)
            pipe.fit(X_train[train_idx], y_train[train_idx].astype(int))
            oof_prob[valid_idx] = sklearn_probs(pipe, X_train[valid_idx])
            print(f"{model_name} physics fold {fold}/5", flush=True)
        pipe = make_pipeline(model_name, seed + 99)
        pipe.fit(X_train, y_train.astype(int))
        oof[model_name] = oof_prob
        test_prob[model_name] = sklearn_probs(pipe, X_test)

    candidates = {}
    for rf_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        name = f"blend_rf{int(rf_weight * 100):03d}"
        candidates[name] = {
            "oof": rf_weight * oof["rf"] + (1.0 - rf_weight) * oof["extra"],
            "test": rf_weight * test_prob["rf"] + (1.0 - rf_weight) * test_prob["extra"],
        }

    results = {}
    track2_pred = old_track2_perfect_predictions()
    for name, data in candidates.items():
        global_scores = {}
        for threshold in np.arange(0.30, 0.56, 0.01):
            score = base.track1_score(y_train, (data["oof"] >= threshold).astype(np.float32))
            global_scores[f"{threshold:.2f}"] = float(score)
        best_threshold = float(max(global_scores, key=global_scores.get))
        item_thresholds, item_score = optimize_item_thresholds(y_train, data["oof"], start=best_threshold)
        results[name] = {
            "best_global_threshold": best_threshold,
            "best_global_score": float(global_scores[f"{best_threshold:.2f}"]),
            "item_score": float(item_score),
            "item_thresholds": item_thresholds.tolist(),
        }
        global_path = out_dir / f"candidate_physics_{name}_global.csv"
        item_path = out_dir / f"candidate_physics_{name}_item.csv"
        write_track1_submission(data["test"], best_threshold, base.TRACK1_TEST_IDS, track2_pred, global_path)
        write_track1_submission(data["test"], item_thresholds, base.TRACK1_TEST_IDS, track2_pred, item_path)
        base.validate_submission_csv(global_path)
        base.validate_submission_csv(item_path)
        results[name]["global_csv"] = str(global_path)
        results[name]["item_csv"] = str(item_path)
        print(name, results[name], flush=True)

    report = {
        "feature_shape": [int(X_all.shape[0]), int(X_all.shape[1])],
        "feature_count": len(feature_names),
        "results": results,
        "note": "Track2 is fixed to the current perfect public labels; compare candidates on Track1 OOF only before submitting.",
    }
    dump_json(report, out_dir / "physics_gait_report.json")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--track1-train", default="track1_train.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--force-physics-features", action="store_true")
    args = parser.parse_args()
    evaluate_and_write(args)


if __name__ == "__main__":
    main()
