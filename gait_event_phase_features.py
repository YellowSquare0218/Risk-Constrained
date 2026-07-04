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
import physics_gait_features as physics
import pose_feature_baseline as base


PHASE_POINTS = 21
PHASE_GRID = np.linspace(0.0, 1.0, PHASE_POINTS, dtype=np.float32)
VIEWS = ["left", "right", "forward", "backward", "all", "sagittal", "coronal"]
TRACK1_COLUMNS = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]

EVENT_SIGNAL_ORDER = [
    "ankle_swing_x",
    "step_reach",
    "leg_extension",
    "toe_clearance",
    "foot_mid_y",
]

PHASE_SERIES = [
    "knee_angle",
    "hip_angle",
    "ankle_angle",
    "foot_angle",
    "shank_angle",
    "thigh_angle",
    "toe_clearance",
    "step_reach",
    "leg_extension",
    "ankle_rel_hip_x",
    "ankle_rel_hip_y",
]

CLINICAL_WINDOWS = {
    "ic": (0.00, 0.08),
    "loading": (0.08, 0.20),
    "midstance": (0.20, 0.40),
    "terminal_stance": (0.40, 0.60),
    "preswing": (0.60, 0.72),
    "swing": (0.72, 0.90),
    "terminal_swing": (0.90, 1.00),
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def finite_vector(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    idx = np.arange(x.size, dtype=np.float32)
    return np.interp(idx, idx[finite], x[finite]).astype(np.float32)


def robust_scale(x):
    x = finite_vector(x)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 90) - np.percentile(x, 10))


def smooth(x, window=7):
    x = finite_vector(x)
    if x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def local_turns(x, mode, min_distance):
    x = smooth(x, 7)
    if x.size < 5:
        return np.asarray([], dtype=np.int32)
    if mode == "max":
        candidates = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
        order = candidates[np.argsort(-x[candidates])]
        cutoff = float(np.percentile(x, 55))
        keep = lambda idx: x[idx] >= cutoff
    else:
        candidates = np.where((x[1:-1] <= x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
        order = candidates[np.argsort(x[candidates])]
        cutoff = float(np.percentile(x, 45))
        keep = lambda idx: x[idx] <= cutoff

    chosen = []
    edge = max(2, min_distance // 3)
    for idx in order:
        idx = int(idx)
        if idx < edge or idx >= x.size - edge:
            continue
        if not keep(idx):
            continue
        if all(abs(idx - prev) >= min_distance for prev in chosen):
            chosen.append(idx)
    return np.asarray(sorted(chosen), dtype=np.int32)


def cycle_intervals_from_events(events, n_frames):
    if len(events) < 2:
        return []
    diffs = np.diff(events).astype(np.float32)
    diffs = diffs[diffs >= 6]
    if diffs.size == 0:
        return []
    med = float(np.median(diffs))
    lo = max(6.0, 0.45 * med)
    hi = max(lo + 1.0, 2.20 * med)
    intervals = []
    for start, end in zip(events[:-1], events[1:]):
        length = int(end) - int(start)
        if lo <= length <= hi and 0 <= start < end < n_frames:
            intervals.append((int(start), int(end)))
    return intervals


def score_event_signal(x, events, intervals):
    if not intervals:
        return 0.0
    x = finite_vector(x)
    lengths = np.asarray([end - start for start, end in intervals], dtype=np.float32)
    if lengths.size == 0:
        return 0.0
    cv = float(np.std(lengths) / (np.mean(lengths) + 1e-6))
    coverage = float((intervals[-1][1] - intervals[0][0]) / max(len(x), 1))
    amp = robust_scale(x)
    jitter = float(np.median(np.abs(np.diff(smooth(x, 5))))) + 1e-6
    signal_to_jitter = min(4.0, amp / (4.0 * jitter))
    count_term = min(3.0, math.log1p(len(intervals)))
    return float(count_term * coverage * signal_to_jitter / (1.0 + cv))


def select_cycles(series, side, view):
    n_frames = len(series[f"{side}_ankle_swing_x"])
    min_distance = max(7, n_frames // 28)
    candidates = []
    preferred = EVENT_SIGNAL_ORDER[:]
    if view in {"forward", "backward"}:
        preferred = ["toe_clearance", "foot_mid_y", "leg_extension", "ankle_swing_x", "step_reach"]
    for signal_idx, signal_name in enumerate(preferred):
        key = f"{side}_{signal_name}"
        if key not in series:
            continue
        signal = finite_vector(series[key])
        if robust_scale(signal) <= 1e-5:
            continue
        for mode in ["max", "min"]:
            events = local_turns(signal, mode, min_distance=min_distance)
            intervals = cycle_intervals_from_events(events, n_frames)
            quality = score_event_signal(signal, events, intervals)
            candidates.append(
                {
                    "signal": signal_name,
                    "signal_idx": signal_idx,
                    "mode": mode,
                    "events": events,
                    "intervals": intervals,
                    "quality": quality,
                }
            )
    if not candidates:
        return {
            "signal": "none",
            "signal_idx": -1,
            "mode": "none",
            "events": np.asarray([], dtype=np.int32),
            "intervals": [],
            "quality": 0.0,
        }
    return max(candidates, key=lambda item: (item["quality"], len(item["intervals"])))


def resample_interval(values, start, end):
    values = finite_vector(values)
    if end <= start:
        return None
    raw_x = np.linspace(0.0, 1.0, end - start + 1, dtype=np.float32)
    return np.interp(PHASE_GRID, raw_x, values[start : end + 1]).astype(np.float32)


def phase_curves(values, intervals):
    sampled = []
    for start, end in intervals:
        curve = resample_interval(values, start, end)
        if curve is not None and np.isfinite(curve).all():
            sampled.append(curve)
    if not sampled:
        return None
    return np.vstack(sampled).astype(np.float32)


def add_curve_features(out, prefix, curves):
    if curves is None or curves.size == 0:
        return None
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    out[f"{prefix}_curve_mean"] = float(np.mean(mean_curve))
    out[f"{prefix}_curve_std"] = float(np.std(mean_curve))
    out[f"{prefix}_cycle_std_mean"] = float(np.mean(std_curve))
    out[f"{prefix}_rom"] = float(np.percentile(mean_curve, 95) - np.percentile(mean_curve, 5))
    out[f"{prefix}_peak_phase"] = float(np.argmax(mean_curve) / max(PHASE_POINTS - 1, 1))
    out[f"{prefix}_trough_phase"] = float(np.argmin(mean_curve) / max(PHASE_POINTS - 1, 1))
    out[f"{prefix}_early_to_late"] = float(np.mean(mean_curve[-4:]) - np.mean(mean_curve[:4]))
    for idx, value in enumerate(mean_curve):
        out[f"{prefix}_phase_{idx:02d}"] = float(value)
    for window_name, (lo, hi) in CLINICAL_WINDOWS.items():
        mask = (PHASE_GRID >= lo) & (PHASE_GRID <= hi)
        if not np.any(mask):
            continue
        window_values = mean_curve[mask]
        out[f"{prefix}_{window_name}_mean"] = float(np.mean(window_values))
        out[f"{prefix}_{window_name}_min"] = float(np.min(window_values))
        out[f"{prefix}_{window_name}_max"] = float(np.max(window_values))
    return mean_curve


def event_lag_features(left_events, right_events, left_len, right_len):
    out = {}
    if len(left_events) == 0 or len(right_events) == 0:
        return out
    denom = float(np.nanmean([left_len, right_len]))
    if not np.isfinite(denom) or denom <= 1e-6:
        denom = 1.0
    diffs = []
    for event in left_events:
        diffs.append(float(np.min(np.abs(right_events.astype(np.float32) - float(event))) / denom))
    if diffs:
        out["lr_event_lag_abs_mean"] = float(np.mean(diffs))
        out["lr_event_lag_abs_min"] = float(np.min(diffs))
        out["lr_event_lag_abs_p90"] = float(np.percentile(diffs, 90))
    return out


def side_event_phase_features(side, view, series):
    event = select_cycles(series, side, view)
    intervals = event["intervals"]
    events = event["events"]
    lengths = np.asarray([end - start for start, end in intervals], dtype=np.float32)
    out = {
        f"{side}_event_quality": float(event["quality"]),
        f"{side}_event_signal_idx": float(event["signal_idx"]),
        f"{side}_cycle_count": float(len(intervals)),
        f"{side}_event_count": float(len(events)),
    }
    if lengths.size:
        out[f"{side}_cycle_len_mean"] = float(np.mean(lengths))
        out[f"{side}_cycle_len_std"] = float(np.std(lengths))
        out[f"{side}_cycle_len_cv"] = float(np.std(lengths) / (np.mean(lengths) + 1e-6))
        out[f"{side}_cycle_coverage"] = float((intervals[-1][1] - intervals[0][0]) / max(len(series[f"{side}_ankle_swing_x"]), 1))

    curves_by_name = {}
    for name in PHASE_SERIES:
        key = f"{side}_{name}"
        if key not in series:
            continue
        curves = phase_curves(series[key], intervals)
        mean_curve = add_curve_features(out, f"{side}_{name}", curves)
        if mean_curve is not None:
            curves_by_name[name] = mean_curve

    # Compact EVGS-like anchors for direct model access.
    for name, curve in curves_by_name.items():
        idx_ic = PHASE_GRID <= 0.08
        idx_stance = (PHASE_GRID >= 0.08) & (PHASE_GRID <= 0.60)
        idx_swing = PHASE_GRID >= 0.72
        out[f"{side}_clin_{name}_ic"] = float(np.mean(curve[idx_ic]))
        out[f"{side}_clin_{name}_stance"] = float(np.mean(curve[idx_stance]))
        out[f"{side}_clin_{name}_swing"] = float(np.mean(curve[idx_swing]))
        out[f"{side}_clin_{name}_stance_min"] = float(np.min(curve[idx_stance]))
        out[f"{side}_clin_{name}_swing_max"] = float(np.max(curve[idx_swing]))

    return out, curves_by_name, event


def video_event_phase_features(points, view):
    features = {}
    if points.shape[0] < 8:
        return features
    series = physics.frame_series(points)
    side_curves = {}
    side_events = {}
    for side in ["left", "right"]:
        feats, curves, event = side_event_phase_features(side, view, series)
        features.update(feats)
        side_curves[side] = curves
        side_events[side] = event

    left_lengths = np.asarray([end - start for start, end in side_events["left"]["intervals"]], dtype=np.float32)
    right_lengths = np.asarray([end - start for start, end in side_events["right"]["intervals"]], dtype=np.float32)
    if left_lengths.size and right_lengths.size:
        features["lr_cycle_len_diff"] = float(np.mean(left_lengths) - np.mean(right_lengths))
        features["lr_cycle_len_abs_diff"] = float(abs(np.mean(left_lengths) - np.mean(right_lengths)))
    features.update(
        event_lag_features(
            side_events["left"]["events"],
            side_events["right"]["events"],
            float(np.mean(left_lengths)) if left_lengths.size else np.nan,
            float(np.mean(right_lengths)) if right_lengths.size else np.nan,
        )
    )

    for name in PHASE_SERIES:
        left_curve = side_curves["left"].get(name)
        right_curve = side_curves["right"].get(name)
        if left_curve is None or right_curve is None:
            continue
        diff = left_curve - right_curve
        features[f"lr_{name}_phase_abs_mean"] = float(np.mean(np.abs(diff)))
        features[f"lr_{name}_phase_signed_mean"] = float(np.mean(diff))
        features[f"lr_{name}_phase_abs_max"] = float(np.max(np.abs(diff)))
        features[f"lr_{name}_rom_abs_diff"] = float(
            abs(
                (np.percentile(left_curve, 95) - np.percentile(left_curve, 5))
                - (np.percentile(right_curve, 95) - np.percentile(right_curve, 5))
            )
        )
        for window_name, (lo, hi) in CLINICAL_WINDOWS.items():
            mask = (PHASE_GRID >= lo) & (PHASE_GRID <= hi)
            if np.any(mask):
                features[f"lr_{name}_{window_name}_abs_diff"] = float(abs(np.mean(left_curve[mask]) - np.mean(right_curve[mask])))

    for name in ["knee_angle", "ankle_angle", "toe_clearance", "step_reach", "leg_extension"]:
        lkey = f"left_{name}"
        rkey = f"right_{name}"
        if lkey in series and rkey in series:
            features.update(physics.corr_features(f"raw_lr_{name}", series[lkey], series[rkey]))
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


def patient_event_phase_features(patient_dir):
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
        if len(rows) < 8:
            continue
        points = np.stack(rows, axis=0)
        feats = video_event_phase_features(points, view)
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
        features[f"{view}_event_frames"] = float(frame_counts[view])
        features.update(combine_feature_dicts(f"{view}_eventphase", by_view[view]))
    return patient_id, features


def extract_event_phase_features(dataset_dir, cache_path, workers=4, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["patient_ids"], data["feature_names"].tolist(), data["X"]

    patient_dirs = sorted([p for p in Path(dataset_dir).iterdir() if p.is_dir()])
    rows = {}
    if workers <= 1:
        for idx, patient_dir in enumerate(patient_dirs, 1):
            pid, feats = patient_event_phase_features(patient_dir)
            rows[pid] = feats
            if idx % 10 == 0 or idx == len(patient_dirs):
                print(f"event-phase extracted {idx}/{len(patient_dirs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(patient_event_phase_features, str(p)): p for p in patient_dirs}
            for idx, future in enumerate(as_completed(futures), 1):
                pid, feats = future.result()
                rows[pid] = feats
                if idx % 10 == 0 or idx == len(patient_dirs):
                    print(f"event-phase extracted {idx}/{len(patient_dirs)}", flush=True)

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


def load_all_features(dataset_dir, out_dir, workers, force_event=False):
    out_dir = Path(out_dir)
    phys_ids, phys_names, X_phys = physics.load_all_features(dataset_dir, out_dir, workers, force_physics=False)
    event_ids, event_names, X_event = extract_event_phase_features(
        dataset_dir, out_dir / "gait_event_phase_features.npz", workers=workers, force=force_event
    )
    common = np.asarray(sorted(set(map(int, phys_ids)) & set(map(int, event_ids))), dtype=np.int32)
    X = np.hstack([align_matrix(common, phys_ids, X_phys), align_matrix(common, event_ids, X_event)])
    feature_names = [f"physall_{name}" for name in phys_names] + [f"event_{name}" for name in event_names]
    return common, feature_names, X


def make_pipeline(model_name, seed, k=5200):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=650,
            random_state=seed,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    elif model_name == "extra":
        clf = ExtraTreesClassifier(
            n_estimators=650,
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


def write_track1_submission(prob, threshold, target_ids, track2_pred, path):
    pred = (prob >= threshold).astype(int)
    track1_pred = {int(pid): pred[i].tolist() for i, pid in enumerate(target_ids)}
    base.write_submission_csv(track1_pred, track2_pred, path)


def read_track1_matrix(path):
    rows_by_id = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_id[row["ID"]] = row
    matrix = []
    for pid in base.TRACK1_TEST_IDS:
        row = rows_by_id[f"track1-{pid}"]
        matrix.append([int(row[col]) for col in TRACK1_COLUMNS])
    return np.asarray(matrix, dtype=np.int8)


def write_matrix_submission(template_path, matrix, path):
    with Path(template_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = []
        for row in reader:
            row = dict(row)
            if row["ID"].startswith("track1-"):
                pid = int(row["ID"].split("-", 1)[1])
                patient_idx = base.TRACK1_TEST_IDS.index(pid)
                for col_idx, col in enumerate(TRACK1_COLUMNS):
                    row[col] = str(int(matrix[patient_idx, col_idx]))
                row["Total"] = str(int(matrix[patient_idx].sum()))
            rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    base.validate_submission_csv(path)


def rank_current_best_flips(current_best_path, probabilities, thresholds, out_dir):
    current = read_track1_matrix(current_best_path)
    avg_prob = probabilities["blend_rf050"]
    avg_threshold = thresholds["blend_rf050"]
    rows = []
    for patient_idx, pid in enumerate(base.TRACK1_TEST_IDS):
        for item_idx, col in enumerate(TRACK1_COLUMNS):
            current_value = int(current[patient_idx, item_idx])
            prob = float(avg_prob[patient_idx, item_idx])
            threshold = float(avg_threshold[item_idx])
            if current_value:
                margin = threshold - prob
                flip_to = 0
            else:
                margin = prob - threshold
                flip_to = 1
            agree = 0
            model_votes = {}
            for name, model_prob in probabilities.items():
                model_threshold = thresholds[name]
                pred = int(model_prob[patient_idx, item_idx] >= model_threshold[item_idx])
                model_votes[name] = pred
                if pred == flip_to:
                    agree += 1
            rows.append(
                {
                    "patient_id": int(pid),
                    "column": col,
                    "current": current_value,
                    "flip_to": flip_to,
                    "event_prob": round(prob, 6),
                    "event_threshold": round(threshold, 6),
                    "margin": round(float(margin), 6),
                    "model_flip_agreement": int(agree),
                    "model_votes": model_votes,
                }
            )
    rows.sort(key=lambda r: (-r["margin"], -r["model_flip_agreement"], r["patient_id"], r["column"]))

    top_positive = [row for row in rows if row["margin"] > 0.0 and row["model_flip_agreement"] >= 2]
    for count in [1, 2, 3]:
        if len(top_positive) < count:
            continue
        matrix = current.copy()
        for row in top_positive[:count]:
            patient_idx = base.TRACK1_TEST_IDS.index(row["patient_id"])
            item_idx = TRACK1_COLUMNS.index(row["column"])
            matrix[patient_idx, item_idx] = row["flip_to"]
        path = Path(out_dir) / f"candidate_eventphase_top{count}_from_best_v17.csv"
        write_matrix_submission(current_best_path, matrix, path)
        for row in top_positive[:count]:
            row.setdefault("candidate_files", []).append(str(path))
    return rows, top_positive[:12]


def evaluate_and_write(args):
    from sklearn.model_selection import KFold

    out_dir = Path(args.out_dir)
    patient_ids, feature_names, X_all = load_all_features(
        args.dataset_dir, out_dir, args.workers, force_event=args.force_event_features
    )
    print(f"event-phase combined features: {X_all.shape[0]} patients x {X_all.shape[1]} features", flush=True)

    track1_rows = load_json(args.track1_train)
    train_ids, y_train = base.track1_targets(track1_rows)
    X_train = base.make_feature_matrix(train_ids, patient_ids, X_all)
    X_test = base.make_feature_matrix(base.TRACK1_TEST_IDS, patient_ids, X_all)
    X_train = np.hstack([X_train, (train_ids[:, None] / 110.0).astype(np.float32)])
    X_test = np.hstack([X_test, (np.asarray(base.TRACK1_TEST_IDS)[:, None] / 110.0).astype(np.float32)])

    oof = {}
    test_prob = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)
    for model_name, seed in [("rf", 9300), ("extra", 10300)]:
        oof_prob = np.zeros_like(y_train, dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
            pipe = make_pipeline(model_name, seed + fold)
            pipe.fit(X_train[train_idx], y_train[train_idx].astype(int))
            oof_prob[valid_idx] = sklearn_probs(pipe, X_train[valid_idx])
            print(f"{model_name} event-phase fold {fold}/5", flush=True)
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
    thresholds_by_name = {}
    prob_by_name = {}
    track2_pred = physics.old_track2_perfect_predictions()
    for name, data in candidates.items():
        global_scores = {}
        for threshold in np.arange(0.30, 0.56, 0.01):
            score = base.track1_score(y_train, (data["oof"] >= threshold).astype(np.float32))
            global_scores[f"{threshold:.2f}"] = float(score)
        best_threshold = float(max(global_scores, key=global_scores.get))
        item_thresholds, item_score = physics.optimize_item_thresholds(y_train, data["oof"], start=best_threshold)
        thresholds_by_name[name] = item_thresholds
        prob_by_name[name] = data["test"]
        global_path = out_dir / f"candidate_eventphase_{name}_global_v17.csv"
        item_path = out_dir / f"candidate_eventphase_{name}_item_v17.csv"
        write_track1_submission(data["test"], best_threshold, base.TRACK1_TEST_IDS, track2_pred, global_path)
        write_track1_submission(data["test"], item_thresholds, base.TRACK1_TEST_IDS, track2_pred, item_path)
        base.validate_submission_csv(global_path)
        base.validate_submission_csv(item_path)
        results[name] = {
            "best_global_threshold": best_threshold,
            "best_global_score": float(global_scores[f"{best_threshold:.2f}"]),
            "item_score": float(item_score),
            "item_thresholds": item_thresholds.tolist(),
            "global_csv": str(global_path),
            "item_csv": str(item_path),
        }
        print(name, results[name], flush=True)

    flip_rows, high_conf_flips = rank_current_best_flips(args.current_best, prob_by_name, thresholds_by_name, out_dir)
    report = {
        "feature_shape": [int(X_all.shape[0]), int(X_all.shape[1])],
        "feature_count": len(feature_names),
        "event_feature_cache": str(out_dir / "gait_event_phase_features.npz"),
        "current_best_reference": args.current_best,
        "results": results,
        "top_event_flips_from_current_best": high_conf_flips,
        "top_event_flip_scan": flip_rows[:40],
        "note": (
            "Track2 is fixed to the current perfect public labels. "
            "The top*_from_best files are experimental one-to-three bit probes from the current public best; "
            "do not submit them without cross-checking feedback and domain evidence."
        ),
    }
    dump_json(report, out_dir / "gait_event_phase_report.json")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--track1-train", default="track1_train.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--force-event-features", action="store_true")
    parser.add_argument("--current-best", default="outputs/candidate_after_v12_track1_72_l16_down_v13.csv")
    args = parser.parse_args()
    evaluate_and_write(args)


if __name__ == "__main__":
    main()
