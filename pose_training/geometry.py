"""Column-vector convention: target = R @ source + t; quaternions are xyzw."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    identity = torch.cat([torch.zeros_like(q[..., :3]), torch.ones_like(q[..., 3:4])], dim=-1)
    return torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)


def quaternion_matrix(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def quaternion_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sign-invariant squared chordal loss, locally equal to squared angle in radians."""
    pred, target = normalize_quaternion(pred), normalize_quaternion(target)
    positive = (pred - target).square().sum(-1)
    negative = (pred + target).square().sum(-1)
    return 4.0 * torch.minimum(positive, negative)


def pose_errors(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.shape[-1] != 7:
        raise ValueError("Expected matching [..., 7] poses in [txyz, qxyzw] order")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Non-finite pose")
    pn = np.linalg.norm(pred[..., 3:], axis=-1, keepdims=True)
    tn = np.linalg.norm(truth[..., 3:], axis=-1, keepdims=True)
    if np.any(pn < 1e-8) or np.any(tn < 1e-8):
        raise ValueError("Invalid zero quaternion")
    p, t = pred[..., 3:] / pn, truth[..., 3:] / tn
    dot = np.clip(np.abs(np.sum(p*t, axis=-1)), 0, 1)
    angle = np.rad2deg(2*np.arccos(dot))
    translation = np.linalg.norm(pred[..., :3] - truth[..., :3], axis=-1)
    return angle, translation


def summarize_errors(pred: np.ndarray, truth: np.ndarray) -> dict:
    angles, shifts = pose_errors(pred, truth)
    return {
        "count": int(angles.size),
        "rotation_mean_deg": float(angles.mean()),
        "rotation_median_deg": float(np.median(angles)),
        "rotation_p95_deg": float(np.quantile(angles, .95)),
        "translation_mean_mm": float(shifts.mean()*1000),
        "translation_median_mm": float(np.median(shifts)*1000),
        "translation_p95_mm": float(np.quantile(shifts, .95)*1000),
        "success_at_2deg_10mm": float(np.mean((angles < 2) & (shifts < .01))),
    }


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch fit for corresponding [N,3] points, with reflection rejection."""
    a, b = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("Need matching [N,3] arrays with at least three points")
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    if np.linalg.det(vt.T @ u.T) < 0:
        vt[-1] *= -1
    r = vt.T @ u.T
    return r, cb - r @ ca


def matrix_pose(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([t, Rotation.from_matrix(r).as_quat()]).astype(np.float32)
