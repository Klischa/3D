"""Evaluate once on held-out scenes, with geometric baselines and explicit gates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
from .data import DataConfig, SyntheticPairs, MODES
from .geometry import rigid_fit, matrix_pose, summarize_errors
from .model import BASE_SHA256, load_checkpoint


def centroid_pose(source, target):
    t = target.mean(axis=2)-source.mean(axis=2)
    q = np.zeros((len(source), 4), dtype=np.float32)
    q[:, 3] = 1
    return np.concatenate([t, q], axis=1)


def icp(source, target, initial, iterations=25, max_distance=.03):
    """Trimmed point-to-point ICP, same settings for both initializers."""
    source, target = source.T.astype(np.float64), target.T.astype(np.float64)
    r = Rotation.from_quat(initial[3:]).as_matrix()
    t = initial[:3].astype(np.float64)
    tree = cKDTree(target)
    used = 0
    for used in range(1, iterations+1):
        moved = source @ r.T + t
        distances, indices = tree.query(moved, workers=1)
        threshold = min(float(np.quantile(distances, .8)), max_distance)
        mask = distances <= threshold
        if mask.sum() < 12:
            break
        dr, dt = rigid_fit(moved[mask], target[indices[mask]])
        r, t = dr @ r, dr @ t + dt
        if np.linalg.norm(dr-np.eye(3)) < 1e-5 and np.linalg.norm(dt) < 1e-6:
            break
    return matrix_pose(r, t), used


def infer_torch(model, a, b, batch_size=8):
    result = []
    with torch.inference_mode():
        for i in range(0, len(a), batch_size):
            result.append(model(torch.from_numpy(a[i:i+batch_size]), torch.from_numpy(b[i:i+batch_size])).numpy())
    return np.concatenate(result)


def original_predictions(session, a, b):
    result = []
    for i in range(0, len(a), 8):
        x, y = a[i:i+8], b[i:i+8]
        output = session.run(["pose"], {"source": x-x.mean(2, keepdims=True), "target": y-y.mean(2, keepdims=True)})[0]
        norm = np.linalg.norm(output[:, 3:], axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Original model produced a zero quaternion")
        output[:, 3:] /= norm
        result.append(output)
    return np.concatenate(result)


def quality_gates(groups, identity):
    """Necessary experimental gates, NOT sufficient for a deployment approval."""
    learned, baseline = groups["all"]["neural"], groups["all"]["centroid"]
    rules = {
        "rotation_better_than_centroid_by_10pct": learned["rotation_mean_deg"] < .9*baseline["rotation_mean_deg"],
        "translation_better_than_centroid_by_10pct": learned["translation_mean_mm"] < .9*baseline["translation_mean_mm"],
        "identity_rotation_mean_below_0_5deg": identity["rotation_mean_deg"] < .5,
        "identity_translation_mean_below_3mm": identity["translation_mean_mm"] < 3,
        "icp_success_not_worse_by_more_than_2pct": groups["all"]["neural_icp"]["success_at_2deg_10mm"] >= groups["all"]["centroid_icp"]["success_at_2deg_10mm"]-.02,
    }
    return {"checks": rules, "synthetic_gates_passed": all(rules.values()),
            "deployment_approved": False, "reason": "No real held-out Astra scans with ground truth; synthetic success is not deployment validation."}


def evaluate(checkpoint, out, original_onnx=None, pairs_per_mode=64, split="test"):
    if pairs_per_mode < 1 or split not in ("validation", "test"):
        raise ValueError("Evaluation requires positive pair count and validation/test split")
    torch.set_num_threads(1)
    model, state = load_checkpoint(checkpoint)
    config = DataConfig(**state["config"]["data"])
    factory = SyntheticPairs(config)
    session = None
    if original_onnx:
        if hashlib.sha256(Path(original_onnx).read_bytes()).hexdigest() != BASE_SHA256:
            raise ValueError("Original model checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(original_onnx), options, providers=["CPUExecutionProvider"])
    groups, accumulated = {}, {}
    scene_keys = set()
    started = time.perf_counter()
    all_truth = []
    for mode in MODES:
        pairs = [factory.pair(split, i, mode=mode, identity=False) for i in range(pairs_per_mode)]
        scene_keys.update(p.scene for p in pairs)
        a, b, truth = [np.stack([getattr(p, key) for p in pairs]) for key in ("source", "target", "pose")]
        predictions = {"neural": infer_torch(model, a, b), "centroid": centroid_pose(a, b)}
        for name in ("centroid", "neural"):
            refined = [icp(x, y, initial)[0] for x, y, initial in zip(a, b, predictions[name])]
            predictions[name+"_icp"] = np.stack(refined)
        if session:
            predictions["original_cpp_contract"] = original_predictions(session, a, b)
        groups[mode] = {key: summarize_errors(value, truth) for key, value in predictions.items()}
        for key, value in predictions.items():
            accumulated.setdefault(key, []).append(value)
        all_truth.append(truth)
    truth = np.concatenate(all_truth)
    groups["all"] = {key: summarize_errors(np.concatenate(value), truth) for key, value in accumulated.items()}
    a, b, truth = factory.batch(split, range(pairs_per_mode), mode="clean", identity=True)
    identity = summarize_errors(infer_torch(model, a, b), truth)
    report = {
        "status": "experimental_synthetic_only", "real_world_validated": False,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_step": state["step"], "split": split,
        "scene_keys": sorted(scene_keys), "pairs_per_mode": pairs_per_mode,
        "points": config.points, "data": factory.description(),
        "metrics": groups, "identical_clouds": identity,
        "quality_gates": quality_gates(groups, identity),
        "elapsed_seconds": time.perf_counter()-started,
        "notes": ["No real Astra data or real-sensor noise calibration.",
                  "All metrics are source->target translation and sign-invariant quaternion angle.",
                  "Centroid baseline: identity rotation and difference of cloud centroids.",
                  "Both ICP baselines use identical 25-iteration trimmed point-to-point ICP.",
                  "Original ONNX is interpreted exactly as current C++ code (centered inputs; txyz+qxyzw). Its training-time contract is unknown.",
                  "Do not tune on this test report; select checkpoints using validation only."],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"all": groups["all"], "identity": identity, "gates": report["quality_gates"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--original-onnx")
    parser.add_argument("--pairs-per-mode", type=int, default=64)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.out, args.original_onnx, args.pairs_per_mode, args.split)


if __name__ == "__main__":
    main()
