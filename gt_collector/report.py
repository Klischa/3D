"""Отчёт по сессии: качество GT-меток и пригодность данных.

Ключевые метрики:
* доля годных кадров и разброс флагов;
* резидуалы Umeyama (p50/p95/max) — по-кадровая уверенность GT;
* noise floor: на статичных участках (шаг < 2 мм и < 0.5°) дисперсия позы
  = нижняя граница ошибки GT; если статичного участка нет — оценка по
  самым тихим кадрам;
* шаги между кадрами (покрытие пространства поз для обучения).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .pairs import load_poses, extract_pairs
from .session import CSV_HEADER

_STATIC_STEP_MM = 2.0
_STATIC_STEP_DEG = 0.5


def build_report(session_dir: str | Path) -> dict:
    sdir = Path(session_dir)
    rows = load_poses(sdir)
    manifest = {}
    mp = sdir / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))

    bad = ("no_board", "poor_fit", "fast", "stale", "thin")
    flags_set = [set(r["flags"].split("|")) & set(bad) for r in rows]
    valid_idx = [i for i, fs in enumerate(flags_set) if not fs]
    n = len(rows)
    rep: dict = {
        "session": str(sdir),
        "frames": n,
        "valid_frames": len(valid_idx),
        "valid_ratio": (len(valid_idx) / n) if n else 0.0,
        "flags": manifest.get("flags_summary", {}),
        "detector_open_cv_errors": manifest.get("detector_open_cv_errors", 0),
    }

    if not rows:
        rep["error"] = "poses.csv пуст"
        return rep

    # --- резидуалы ---------------------------------------------------------
    resid = np.array([float(r["mean_residual_mm"]) for r in rows
                      if r["flags"].split("|") and "no_board" not in r["flags"]])
    if len(resid):
        rep["residual_mm"] = {
            "p50": float(np.percentile(resid, 50)),
            "p95": float(np.percentile(resid, 95)),
            "max": float(resid.max()),
        }
    tags = np.array([int(r["n_tags"]) for r in rows])
    rep["n_tags"] = {"min": int(tags.min()), "p50": int(np.percentile(tags, 50)),
                     "max": int(tags.max())}

    # --- шаги --------------------------------------------------------------
    pairs = extract_pairs(sdir)
    if pairs:
        steps = np.array([p["step_mm"] for p in pairs])
        angs = np.array([p["step_deg"] for p in pairs])
        rep["steps"] = {
            "n_pairs": len(pairs),
            "step_mm": {"mean": float(steps.mean()), "p95": float(np.percentile(steps, 95)),
                        "max": float(steps.max())},
            "step_deg": {"mean": float(angs.mean()), "p95": float(np.percentile(angs, 95)),
                         "max": float(angs.max())},
        }

    # --- noise floor по статичным участкам ---------------------------------
    T = [np.eye(4) for _ in rows]
    for i, r in enumerate(rows):
        if "no_board" in r["flags"]:
            continue
        T[i] = np.eye(4)
        T[i][:3, 3] = [float(r["tx"]), float(r["ty"]), float(r["tz"])]
        T[i][:3, :3] = Rotation.from_quat(
            [float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]).as_matrix()
    static = []
    for a, b in zip(range(len(rows) - 1), range(1, len(rows))):
        if "no_board" in rows[a]["flags"] or "no_board" in rows[b]["flags"]:
            continue
        M = np.linalg.inv(T[a]) @ T[b]
        step_mm = float(np.linalg.norm(M[:3, 3])) * 1000.0
        ang = float(np.rad2deg(Rotation.from_matrix(M[:3, :3]).magnitude()))
        if step_mm < _STATIC_STEP_MM and ang < _STATIC_STEP_DEG:
            static.append(b)
    if len(static) >= 5:
        # разброс поз в статике относительно среднего участка
        ts = np.array([[float(rows[i]["tx"]), float(rows[i]["ty"]), float(rows[i]["tz"])]
                       for i in static])
        mats = [Rotation.from_quat([float(rows[i]["qx"]), float(rows[i]["qy"]),
                                    float(rows[i]["qz"]), float(rows[i]["qw"])]).as_matrix()
                for i in static]
        t0 = ts.mean(0)
        # среднее вращение: SVD-проекция на SO(3)
        A = np.mean(mats, axis=0)
        U, _, Vt = np.linalg.svd(A)
        R0 = U @ Vt
        if np.linalg.det(R0) < 0:
            U[:, -1] *= -1
            R0 = U @ Vt
        dt = np.linalg.norm(ts - t0, axis=1) * 1000.0
        da = np.array([float(np.rad2deg(
            Rotation.from_matrix(R0.T @ mats[i]).magnitude())) for i in range(len(static))])
        rep["noise_floor"] = {
            "n_static_frames": int(len(static)),
            "translation_std_mm": float(dt.std()),
            "rotation_std_deg": float(da.std()),
            "translation_p95_mm": float(np.percentile(dt, 95)),
            "rotation_p95_deg": float(np.percentile(da, 95)),
        }
    else:
        rep["noise_floor"] = None
    return rep


def save_report(session_dir: str | Path) -> dict:
    rep = build_report(session_dir)
    (Path(session_dir) / "report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep
