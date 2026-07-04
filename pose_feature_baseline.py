import argparse
import csv
import json
import math
import os
import random
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


BODY_JOINTS = list(range(23))
VIEWS = ["forward", "backward", "left", "right", "all"]
TRACK1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
TRACK2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]
SUBTYPE_ORDER = ["WNL", "type1", "type2", "type3", "type4"]
CSV_COLUMNS = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)


PAIR_FEATURES = [
    ("shoulder_width", 5, 6),
    ("hip_width", 11, 12),
    ("knee_width", 13, 14),
    ("ankle_width", 15, 16),
    ("left_torso", 5, 11),
    ("right_torso", 6, 12),
    ("left_thigh", 11, 13),
    ("right_thigh", 12, 14),
    ("left_shank", 13, 15),
    ("right_shank", 14, 16),
    ("left_foot_span", 17, 19),
    ("right_foot_span", 20, 22),
    ("left_ankle_to_toe", 15, 17),
    ("right_ankle_to_toe", 16, 20),
]
ANGLE_FEATURES = [
    ("left_knee_angle", 11, 13, 15),
    ("right_knee_angle", 12, 14, 16),
    ("left_hip_angle", 5, 11, 13),
    ("right_hip_angle", 6, 12, 14),
    ("left_ankle_foot_angle", 13, 15, 17),
    ("right_ankle_foot_angle", 14, 16, 20),
]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def frame_feature_names():
    names = []
    for j in BODY_JOINTS:
        names.extend([f"j{j}_x", f"j{j}_y", f"j{j}_score"])
    names.extend([name for name, _, _ in PAIR_FEATURES])
    names.extend([name for name, _, _, _ in ANGLE_FEATURES])
    names.extend(["bbox_w", "bbox_h", "bbox_area", "bbox_aspect", "bbox_cx", "bbox_cy"])
    return names


BASE_FRAME_FEATURE_NAMES = frame_feature_names()
VELOCITY_FEATURE_NAMES = [f"j{j}_{axis}_vel" for j in BODY_JOINTS for axis in ("x", "y")]


def infer_view(video_name):
    for view in ["forward", "backward", "left", "right"]:
        if f"_{view}_" in video_name:
            return view
    return "unknown"


def safe_angle(points, a, b, c):
    ba = points[a] - points[b]
    bc = points[c] - points[b]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-8:
        return 0.0
    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.acos(cos_value) / math.pi


def read_frame_features(path):
    try:
        data = load_json(path)
        instances = data.get("instance_info") or []
        if not instances:
            return None
        inst = instances[0]
        keypoints = np.asarray(inst["keypoints"], dtype=np.float32)
        scores = np.asarray(inst.get("keypoint_scores", []), dtype=np.float32)
        if keypoints.shape[0] < max(BODY_JOINTS) + 1:
            return None
        points = keypoints[BODY_JOINTS, :2]
        if scores.shape[0] < max(BODY_JOINTS) + 1:
            scores = np.ones(keypoints.shape[0], dtype=np.float32)
        scores = scores[BODY_JOINTS]
        bbox = inst.get("gt_bbox_xywh_px") or [0, 0, 1, 1]
        x, y, w, h = [float(v) for v in bbox[:4]]
        width = float(data.get("video_info", {}).get("width", 1920))
        height = float(data.get("video_info", {}).get("height", 1080))
        scale = max(w, h, 1.0)
        center = np.asarray([x + w / 2.0, y + h / 2.0], dtype=np.float32)
        norm_points = (points - center) / scale

        values = []
        for idx in range(len(BODY_JOINTS)):
            values.extend([float(norm_points[idx, 0]), float(norm_points[idx, 1]), float(scores[idx])])
        for _, a, b in PAIR_FEATURES:
            values.append(float(np.linalg.norm(norm_points[a] - norm_points[b])))
        for _, a, b, c in ANGLE_FEATURES:
            values.append(safe_angle(norm_points, a, b, c))
        values.extend(
            [
                w / max(width, 1.0),
                h / max(height, 1.0),
                (w * h) / max(width * height, 1.0),
                w / max(h, 1.0),
                (x + w / 2.0) / max(width, 1.0),
                (y + h / 2.0) / max(height, 1.0),
            ]
        )
        return np.asarray(values, dtype=np.float32), norm_points.reshape(-1).astype(np.float32)
    except Exception:
        return None


def describe_matrix(prefix, names, matrix):
    out = {}
    if matrix.size == 0:
        for stat in ["mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90"]:
            for name in names:
                out[f"{prefix}_{stat}_{name}"] = np.nan
        return out

    stats = {
        "mean": np.nanmean(matrix, axis=0),
        "std": np.nanstd(matrix, axis=0),
        "min": np.nanmin(matrix, axis=0),
        "max": np.nanmax(matrix, axis=0),
        "p10": np.nanpercentile(matrix, 10, axis=0),
        "p25": np.nanpercentile(matrix, 25, axis=0),
        "p50": np.nanpercentile(matrix, 50, axis=0),
        "p75": np.nanpercentile(matrix, 75, axis=0),
        "p90": np.nanpercentile(matrix, 90, axis=0),
    }
    for stat, values in stats.items():
        for name, value in zip(names, values):
            out[f"{prefix}_{stat}_{name}"] = float(value)
    return out


def patient_features(patient_dir):
    patient_dir = Path(patient_dir)
    patient_id = int(patient_dir.name)
    by_view = {view: [] for view in VIEWS}
    vel_by_view = {view: [] for view in VIEWS}
    meta = {view: {"frames": 0, "videos": 0} for view in VIEWS}

    for video_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
        view = infer_view(video_dir.name)
        if view == "unknown":
            continue
        frame_values = []
        pose_values = []
        for frame_path in sorted(video_dir.glob("frame_*.json")):
            result = read_frame_features(frame_path)
            if result is None:
                continue
            feature_row, pose_row = result
            frame_values.append(feature_row)
            pose_values.append(pose_row)
        if not frame_values:
            continue
        frame_matrix = np.vstack(frame_values)
        pose_matrix = np.vstack(pose_values)
        by_view[view].append(frame_matrix)
        by_view["all"].append(frame_matrix)
        meta[view]["frames"] += frame_matrix.shape[0]
        meta[view]["videos"] += 1
        meta["all"]["frames"] += frame_matrix.shape[0]
        meta["all"]["videos"] += 1

        if pose_matrix.shape[0] > 1:
            velocity = np.diff(pose_matrix, axis=0)
            vel_by_view[view].append(velocity)
            vel_by_view["all"].append(velocity)

    features = {"patient_id": patient_id}
    for view in VIEWS:
        frame_matrix = np.vstack(by_view[view]) if by_view[view] else np.empty((0, len(BASE_FRAME_FEATURE_NAMES)))
        vel_matrix = np.vstack(vel_by_view[view]) if vel_by_view[view] else np.empty((0, len(VELOCITY_FEATURE_NAMES)))
        features[f"{view}_frames"] = float(meta[view]["frames"])
        features[f"{view}_videos"] = float(meta[view]["videos"])
        features.update(describe_matrix(f"{view}_pose", BASE_FRAME_FEATURE_NAMES, frame_matrix))
        features.update(describe_matrix(f"{view}_vel", VELOCITY_FEATURE_NAMES, vel_matrix))
    return patient_id, features


def extract_pose_features(dataset_dir, cache_path, workers=4, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return data["patient_ids"], data["feature_names"].tolist(), data["X"]

    dataset_dir = Path(dataset_dir)
    patient_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    rows = {}
    if workers <= 1:
        for idx, p in enumerate(patient_dirs, 1):
            pid, feats = patient_features(p)
            rows[pid] = feats
            if idx % 10 == 0 or idx == len(patient_dirs):
                print(f"extracted {idx}/{len(patient_dirs)} patients", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(patient_features, str(p)): p for p in patient_dirs}
            for idx, fut in enumerate(as_completed(futures), 1):
                pid, feats = fut.result()
                rows[pid] = feats
                if idx % 10 == 0 or idx == len(patient_dirs):
                    print(f"extracted {idx}/{len(patient_dirs)} patients", flush=True)

    patient_ids = np.asarray(sorted(rows), dtype=np.int32)
    feature_names = sorted({name for feats in rows.values() for name in feats if name != "patient_id"})
    X = np.empty((len(patient_ids), len(feature_names)), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        feats = rows[int(pid)]
        X[i] = [feats.get(name, np.nan) for name in feature_names]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, patient_ids=patient_ids, feature_names=np.asarray(feature_names), X=X)
    return patient_ids, feature_names, X


def as_by_id(rows):
    return {int(row["patient_id"]): row for row in rows}


def track1_targets(track1_rows):
    ids = []
    y = []
    for row in track1_rows:
        ids.append(int(row["patient_id"]))
        vals = []
        vals.extend(int(row["left"][str(i)]) for i in range(1, 18))
        vals.extend(int(row["right"][str(i)]) for i in range(1, 18))
        y.append(vals)
    return np.asarray(ids, dtype=np.int32), np.asarray(y, dtype=np.float32)


def make_feature_matrix(patient_ids, all_patient_ids, X):
    index = {int(pid): i for i, pid in enumerate(all_patient_ids)}
    return X[[index[int(pid)] for pid in patient_ids]]


def fit_preprocessor(X_train, X_apply):
    mean = np.nanmean(X_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    X_train_imp = np.where(np.isfinite(X_train), X_train, mean)
    X_apply_imp = np.where(np.isfinite(X_apply), X_apply, mean)
    std = X_train_imp.std(axis=0)
    keep = std > 1e-6
    std = std[keep]
    return (X_train_imp[:, keep] - mean[keep]) / std, (X_apply_imp[:, keep] - mean[keep]) / std, keep, mean, std


def transform_with_preprocessor(X, keep, mean, std):
    X_imp = np.where(np.isfinite(X), X, mean)
    return (X_imp[:, keep] - mean[keep]) / std


def ridge_probs(X_train, y_train, X_test, lam=10.0):
    y_mean = y_train.mean(axis=0, keepdims=True)
    yc = y_train - y_mean
    scale = max(1, X_train.shape[1])
    kernel = (X_train @ X_train.T) / scale
    alpha = np.linalg.solve(kernel + lam * np.eye(kernel.shape[0]), yc)
    test_kernel = (X_test @ X_train.T) / scale
    return np.clip(y_mean + test_kernel @ alpha, 0.0, 1.0)


def knn_binary_probs(X_train, y_train, X_test, k=7):
    probs = []
    for row in X_test:
        dist = np.sqrt(np.mean((X_train - row) ** 2, axis=1))
        order = np.argsort(dist)[: min(k, len(dist))]
        weights = 1.0 / (dist[order] + 1e-6)
        probs.append((weights[:, None] * y_train[order]).sum(axis=0) / weights.sum())
    return np.asarray(probs)


def track1_score(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    rmse = float(np.sqrt(np.mean((y_true.sum(axis=1) - y_pred.sum(axis=1)) ** 2)))
    return (acc + 1.0 - rmse / 34.0) / 2.0


def choose_track1_params(X, y, seed=2026):
    rng = random.Random(seed)
    ids = list(range(len(y)))
    candidates = []
    for lam in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        for k in [3, 5, 7, 11, 15]:
            for blend_ridge in [0.0, 0.35, 0.65, 1.0]:
                for threshold in [0.35, 0.4, 0.45, 0.5, 0.55]:
                    candidates.append((lam, k, blend_ridge, threshold))
    scores = {candidate: [] for candidate in candidates}

    for repeat in range(8):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        folds = [shuffled[i::5] for i in range(5)]
        for valid_idx in folds:
            train_idx = [i for i in ids if i not in valid_idx]
            Xtr_raw, Xva_raw = X[train_idx], X[valid_idx]
            ytr, yva = y[train_idx], y[valid_idx]
            Xtr, Xva, _, _, _ = fit_preprocessor(Xtr_raw, Xva_raw)
            prior = np.repeat(ytr.mean(axis=0, keepdims=True), len(valid_idx), axis=0)
            ridge_cache = {}
            knn_cache = {}
            for lam, k, blend_ridge, threshold in candidates:
                if lam not in ridge_cache:
                    ridge_cache[lam] = ridge_probs(Xtr, ytr, Xva, lam=lam)
                if k not in knn_cache:
                    knn_cache[k] = knn_binary_probs(Xtr, ytr, Xva, k=k)
                feature_prob = blend_ridge * ridge_cache[lam] + (1.0 - blend_ridge) * knn_cache[k]
                prob = 0.80 * feature_prob + 0.20 * prior
                pred = (prob >= threshold).astype(np.float32)
                scores[(lam, k, blend_ridge, threshold)].append(track1_score(yva, pred))

    ranked = sorted(scores.items(), key=lambda kv: (-np.mean(kv[1]), np.std(kv[1])))
    best, best_scores = ranked[0]
    return {
        "lambda": best[0],
        "k": best[1],
        "blend_ridge": best[2],
        "threshold": best[3],
        "cv_score": float(np.mean(best_scores)),
        "cv_std": float(np.std(best_scores)),
    }


def predict_track1(X_all, all_patient_ids, track1_rows, target_ids, params):
    train_ids, y_train = track1_targets(track1_rows)
    X_train_raw = make_feature_matrix(train_ids, all_patient_ids, X_all)
    X_target_raw = make_feature_matrix(target_ids, all_patient_ids, X_all)
    X_train, X_target, _, _, _ = fit_preprocessor(X_train_raw, X_target_raw)
    ridge = ridge_probs(X_train, y_train, X_target, lam=params["lambda"])
    knn = knn_binary_probs(X_train, y_train, X_target, k=params["k"])
    prior = np.repeat(y_train.mean(axis=0, keepdims=True), len(target_ids), axis=0)
    prob = 0.80 * (params["blend_ridge"] * ridge + (1.0 - params["blend_ridge"]) * knn) + 0.20 * prior
    pred = (prob >= params["threshold"]).astype(int)
    return {int(pid): pred[i].tolist() for i, pid in enumerate(target_ids)}


def macro_f1(y_true, y_pred):
    labels = SUBTYPE_ORDER
    scores = []
    for label in labels:
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == label and p == label)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != label and p == label)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == label and p != label)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def track2_score(y_true, y_pred):
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    return (acc + macro_f1(y_true, y_pred)) / 2.0


def track2_examples(track2_rows, all_patient_ids, X_all, ids=None):
    ids = set(ids) if ids is not None else None
    patient_ids = []
    sides = []
    labels = []
    for row in track2_rows:
        pid = int(row["patient_id"])
        if ids is not None and pid not in ids:
            continue
        for side_idx, side in enumerate(["left", "right"]):
            patient_ids.append(pid)
            sides.append(side_idx)
            labels.append(row[side]["gait_subtype"])
    X = make_feature_matrix(patient_ids, all_patient_ids, X_all)
    side_features = np.zeros((len(sides), 2), dtype=np.float32)
    for i, side_idx in enumerate(sides):
        side_features[i, side_idx] = 1.0
    return patient_ids, np.hstack([X, side_features]), labels


def knn_classify(X_train, y_train, X_test, k=5):
    out = []
    for row in X_test:
        dist = np.sqrt(np.mean((X_train - row) ** 2, axis=1))
        order = np.argsort(dist)[: min(k, len(dist))]
        weights_by_label = Counter()
        for idx in order:
            weights_by_label[y_train[idx]] += float(1.0 / (dist[idx] + 1e-6))
        out.append(sorted(weights_by_label, key=lambda label: (-weights_by_label[label], SUBTYPE_ORDER.index(label)))[0])
    return out


def choose_track2_k(track2_rows, all_patient_ids, X_all):
    train_patient_ids = [int(row["patient_id"]) for row in track2_rows]
    scores = {}
    for k in [1, 3, 5, 7, 9, 11, 15]:
        y_true = []
        y_pred = []
        for valid_pid in train_patient_ids:
            train_ids = [pid for pid in train_patient_ids if pid != valid_pid]
            _, Xtr_raw, ytr = track2_examples(track2_rows, all_patient_ids, X_all, ids=train_ids)
            _, Xva_raw, yva = track2_examples(track2_rows, all_patient_ids, X_all, ids=[valid_pid])
            Xtr, Xva, _, _, _ = fit_preprocessor(Xtr_raw, Xva_raw)
            pred = knn_classify(Xtr, ytr, Xva, k=k)
            y_true.extend(yva)
            y_pred.extend(pred)
        scores[k] = track2_score(y_true, y_pred)
    best_k = sorted(scores, key=lambda k: (-scores[k], k))[0]
    return {"k": best_k, "cv_score": float(scores[best_k]), "all_scores": {str(k): float(v) for k, v in scores.items()}}


def predict_track2(X_all, all_patient_ids, track2_rows, target_ids, params):
    _, Xtr_raw, ytr = track2_examples(track2_rows, all_patient_ids, X_all)
    out = {}
    for pid in target_ids:
        patient_ids = [pid, pid]
        side_features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        Xpid = make_feature_matrix(patient_ids, all_patient_ids, X_all)
        Xte_raw = np.hstack([Xpid, side_features])
        Xtr, Xte, _, _, _ = fit_preprocessor(Xtr_raw, Xte_raw)
        pred = knn_classify(Xtr, ytr, Xte, k=params["k"])
        out[int(pid)] = {"left": pred[0], "right": pred[1]}
    return out


def majority_label(labels):
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], SUBTYPE_ORDER.index(label)))[0]


def predict_track2_majority(track2_rows, target_ids):
    left = majority_label([row["left"]["gait_subtype"] for row in track2_rows])
    right = majority_label([row["right"]["gait_subtype"] for row in track2_rows])
    return {int(pid): {"left": left, "right": right} for pid in target_ids}


def write_submission_csv(track1_pred, track2_pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for pid in TRACK1_TEST_IDS:
            vals = track1_pred[pid]
            row = {"ID": f"track1-{pid}"}
            for i in range(17):
                row[f"L{i + 1}"] = int(vals[i])
            for i in range(17):
                row[f"R{i + 1}"] = int(vals[17 + i])
            row["Total"] = int(sum(vals))
            row["Left_gait_subtype"] = -1
            row["Right_gait_subtype"] = -1
            writer.writerow(row)
        for pid in TRACK2_TEST_IDS:
            row = {"ID": f"track2-{pid}"}
            for i in range(17):
                row[f"L{i + 1}"] = -1
                row[f"R{i + 1}"] = -1
            row["Total"] = -1
            row["Left_gait_subtype"] = track2_pred[pid]["left"]
            row["Right_gait_subtype"] = track2_pred[pid]["right"]
            writer.writerow(row)


def validate_submission_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected_ids = [f"track1-{pid}" for pid in TRACK1_TEST_IDS] + [f"track2-{pid}" for pid in TRACK2_TEST_IDS]
    got_ids = [row["ID"] for row in rows]
    if got_ids != expected_ids:
        raise ValueError(f"Unexpected IDs/order: {got_ids}")
    return {"rows": len(rows), "columns": len(CSV_COLUMNS)}


def write_zip(zip_path, files):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, arcname=Path(file_path).name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--track1-train", default="track1_train.json")
    parser.add_argument("--track2-train", default="track2_train.json")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--track1-threshold-override", type=float, default=None)
    parser.add_argument("--track2-mode", choices=["pose", "majority"], default="pose")
    parser.add_argument("--submission-name", default="submission_pose_baseline.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    feature_cache = out_dir / "pose_features.npz"
    patient_ids, feature_names, X_all = extract_pose_features(
        args.dataset_dir, feature_cache, workers=args.workers, force=args.force_features
    )
    print(f"feature matrix: {X_all.shape[0]} patients x {X_all.shape[1]} features", flush=True)

    track1_rows = load_json(args.track1_train)
    track2_rows = load_json(args.track2_train)

    train_ids, y_track1 = track1_targets(track1_rows)
    X_track1 = make_feature_matrix(train_ids, patient_ids, X_all)
    track1_params = choose_track1_params(X_track1, y_track1)
    if args.track1_threshold_override is not None:
        track1_params["threshold"] = args.track1_threshold_override
    print(f"track1 params: {track1_params}", flush=True)

    track2_params = choose_track2_k(track2_rows, patient_ids, X_all)
    print(f"track2 params: {track2_params}", flush=True)

    track1_pred = predict_track1(X_all, patient_ids, track1_rows, TRACK1_TEST_IDS, track1_params)
    if args.track2_mode == "majority":
        track2_pred = predict_track2_majority(track2_rows, TRACK2_TEST_IDS)
    else:
        track2_pred = predict_track2(X_all, patient_ids, track2_rows, TRACK2_TEST_IDS, track2_params)

    csv_path = out_dir / args.submission_name
    zip_path = out_dir / f"{csv_path.stem}.zip"
    write_submission_csv(track1_pred, track2_pred, csv_path)
    csv_validation = validate_submission_csv(csv_path)
    write_zip(zip_path, [csv_path])

    report = {
        "feature_cache": str(feature_cache),
        "feature_shape": [int(X_all.shape[0]), int(X_all.shape[1])],
        "track1_params": track1_params,
        "track2_params": track2_params,
        "track2_mode": args.track2_mode,
        "csv_validation": csv_validation,
        "outputs": {"csv": str(csv_path), "zip": str(zip_path)},
    }
    dump_json(report, out_dir / "pose_baseline_report.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
