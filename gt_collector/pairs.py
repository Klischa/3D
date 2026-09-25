"""Пары кадров с относительной GT-позой (контракт модели astra2).

Пара = два подряд идущих ГОДНЫХ кадра (flagn без fast/stale/no_board/
poor_fit/thin) с шагом, меньшим порогов. Относительная поза:

    T_target_from_source = inv(T_target) @ T_source

где source = текущий (более поздний) кадр, target = предыдущий — ровно
контракт pose_regressor (p_target = R·p_source + t в метрах, кватернион
(x,y,z,w)).
"""
from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def _matrix_from_row(row: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
    q = [float(row["qx"]), float(row["qy"]), float(float(row["qz"])), float(row["qw"])]
    m[:3, :3] = Rotation.from_quat(q).as_matrix()
    return m


def load_poses(session_dir: str | Path) -> list[dict]:
    path = Path(session_dir) / "poses.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_pairs(session_dir: str | Path,
                  max_step_translation_m: float = 0.2,
                  max_step_rotation_deg: float = 20.0) -> list[dict]:
    """Список пар: {source_frame, target_frame, T(4×4), step_mm, step_deg}."""
    rows = load_poses(session_dir)
    bad = ("no_board", "poor_fit", "fast", "stale", "thin")
    valid = [r for r in rows if not (set(r["flags"].split("|")) & set(bad))]
    pairs = []
    for a, b in zip(valid, valid[1:]):
        Ta = _matrix_from_row(a)  # target (старший)
        Tb = _matrix_from_row(b)  # source (новый)
        T = np.linalg.inv(Ta) @ Tb
        step_mm = float(np.linalg.norm(T[:3, 3])) * 1000.0
        step_deg = float(np.rad2deg(Rotation.from_matrix(T[:3, :3]).magnitude()))
        if step_mm / 1000.0 > max_step_translation_m or step_deg > max_step_rotation_deg:
            continue
        pairs.append({
            "source_frame": int(b["frame_idx"]),
            "target_frame": int(a["frame_idx"]),
            "T": T,
            "step_mm": step_mm,
            "step_deg": step_deg,
        })
    return pairs


def save_pairs(session_dir: str | Path, out_path: str | None = None) -> dict:
    """pairs.csv (развёрнутая 4×4 + шаг) + сводка; возвращает сводку."""
    pairs = extract_pairs(session_dir)
    out = Path(out_path) if out_path else Path(session_dir) / "pairs.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_frame", "target_frame", "step_mm", "step_deg"] +
                   [f"r{r}c{c}" for r in range(3) for c in range(4)])
        for p in pairs:
            w.writerow([p["source_frame"], p["target_frame"],
                        f"{p['step_mm']:.4f}", f"{p['step_deg']:.4f}"] +
                       [f"{v:.9f}" for v in p["T"].ravel()])
    steps = np.array([p["step_mm"] for p in pairs]) if pairs else np.array([0.0])
    angs = np.array([p["step_deg"] for p in pairs]) if pairs else np.array([0.0])
    summary = {
        "n_pairs": len(pairs),
        "step_mm": {"mean": float(steps.mean()), "p95": float(np.percentile(steps, 95)),
                    "max": float(steps.max())},
        "step_deg": {"mean": float(angs.mean()), "p95": float(np.percentile(angs, 95)),
                     "max": float(angs.max())},
    }
    (Path(session_dir) / "pairs_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary
