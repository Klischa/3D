"""Сессия записи: бэкенд + детектор + позы → папка данных.

Структура сессии::

    session/
      manifest.json      # конфиг, калибровка, устройство, флаги-сводка
      poses.csv          # GT-поза по каждому кадру (4×4 в t+quat, резидуалы)
      report.json        # метрики (report.py)
      frames/
        0000000.jpg      # RGB (JPEG q90)
        0000000.cloud    # облак в системе камеры (ACGD v2: xyz int16 мм + RGB)

GT-поза — board_from_cam (мир = дошка). Флаги качества кадра:
  ok         — все проверки пройдены (кадр годен для пар)
  no_board   — маркеров меньше min_tags (поза в строке — placeholder)
  poor_fit   — средний репроекционный резидуал > max_residual_mm
  thin       — точек в облаке < min_points
  fast       — шаг > max_step_translation_m / max_step_rotation_deg
  stale      — разрыв времени > max_frame_gap_ms
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .board_pose import BoardPose, solve_board_pose
from .cloud import Cloud, make_cloud, save_cloud, write_jpeg
from .detectors import OpenCVAprilTagDetector

CSV_HEADER = ["frame_idx", "ts_us", "n_tags", "tx", "ty", "tz",
              "qx", "qy", "qz", "qw",
              "mean_residual_mm", "max_residual_mm", "normal_spread_deg",
              "n_points", "gap_ms", "flags"]


def pose_to_csv_row(idx: int, ts_us: int, pose: BoardPose, n_points: int,
                    gap_ms: float, flags: list[str]) -> list:
    q = Rotation.from_matrix(pose.R).as_quat()  # (x,y,z,w)
    return [idx, ts_us, pose.n_tags,
            f"{pose.t[0]:.9f}", f"{pose.t[1]:.9f}", f"{pose.t[2]:.9f}",
            f"{q[0]:.9f}", f"{q[1]:.9f}", f"{q[2]:.9f}", f"{q[3]:.9f}",
            f"{pose.mean_residual_mm:.4f}",
            f"{float(pose.per_tag_residual_mm.max() if len(pose.per_tag_residual_mm) else 0):.4f}",
            f"{pose.normal_spread_deg:.4f}",
            n_points, f"{gap_ms:.2f}", "|".join(flags)]


class Session:
    def __init__(self, out_dir: str | Path, cfg, backend, detector=None):
        cfg.validate()
        self.out = Path(out_dir)
        self.frames_dir = self.out / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.backend = backend
        self.detector = detector if detector is not None else backend.detector
        self._last_pose: BoardPose | None = None
        self._last_ts_us = 0
        self._rows: list[list] = []
        self._flags_count: dict[str, int] = {}
        self._start_wall = 0.0

    # ------------------------------------------------------------------
    def _flags(self, pose: BoardPose | None, n_points: int, gap_ms: float) -> list[str]:
        flags = []
        if pose is None:
            flags.append("no_board")
        else:
            if pose.mean_residual_mm > self.cfg.max_residual_mm:
                flags.append("poor_fit")
            if self._last_pose is not None:
                T = BoardPose.relative(pose, self._last_pose)
                d = float(np.linalg.norm(T[:3, 3]))
                ang = float(np.rad2deg(Rotation.from_matrix(T[:3, :3]).magnitude()))
                if d > self.cfg.max_step_translation_m or ang > self.cfg.max_step_rotation_deg:
                    flags.append("fast")
        if n_points < self.cfg.min_points:
            flags.append("thin")
        if self._last_ts_us and gap_ms > self.cfg.max_frame_gap_ms:
            flags.append("stale")
        return flags

    def _count_flags(self, flags: list[str]) -> None:
        for f in (flags or ["ok"]):
            self._flags_count[f] = self._flags_count.get(f, 0) + 1

    # ------------------------------------------------------------------
    def run(self) -> dict:
        self._start_wall = time.time()
        n_total = int(self.cfg.duration_s * self.cfg.fps) if self.cfg.frames is None \
            else self.cfg.frames
        self.backend.start()
        csv_path = self.out / "poses.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(CSV_HEADER)
                for idx in range(n_total):
                    frame = self.backend.wait_for_frame(1000)
                    if frame is None:
                        break
                    if idx < self.cfg.warmup_frames:
                        self._last_ts_us = frame.ts_us
                        continue
                    pose = self._process_frame(idx, frame, w)
        finally:
            self.backend.stop()

        manifest = {
            "version": 1,
            "created_wall": time.time(),
            "config": asdict(self.cfg) if hasattr(self.cfg, "board") else self.cfg.to_dict(),
            "backend": self.backend.metadata(),
            "detector": type(self.detector).__name__,
            "detector_open_cv_errors": getattr(self.detector, "error_count", 0),
            "frames_written": len(self._rows),
            "flags_summary": self._flags_count,
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
        return manifest

    def _process_frame(self, idx: int, frame, writer) -> BoardPose | None:
        gap_ms = (frame.ts_us - self._last_ts_us) / 1000.0 if self._last_ts_us else 0.0
        self._last_ts_us = frame.ts_us

        img_det = self.backend.undistort_bgr(frame.bgr) \
            if hasattr(self.backend, "undistort_bgr") else frame.bgr
        try:
            det = self.detector.detect(img_det)
        except Exception:
            det = []
        pose = None
        if len(det) >= 3:
            k_det = (self.backend.K_undistorted
                     if hasattr(self.backend, "K_undistorted") else self.backend.K)
            pose = solve_board_pose(det, self.cfg.board, np.asarray(k_det, np.float32), None)

        n_points = int(np.count_nonzero(frame.depth_mm > 0))
        flags = self._flags(pose, n_points, gap_ms) or ["ok"]
        self._count_flags(flags)
        row = pose_to_csv_row(idx, frame.ts_us, pose if pose is not None else BoardPose(
            np.zeros((3, 3)), np.zeros(3), len(det), 0.0),
            n_points, gap_ms, flags)
        writer.writerow(row)
        self._rows.append(row)

        # артефакты кадра (каждый save_frames_every-й; позы пишутся всегда)
        step = max(1, int(getattr(self.cfg, "save_frames_every", 1)))
        if idx % step == 0:
            stamp = f"{idx:07d}"
            write_jpeg(frame.bgr, str(self.frames_dir / f"{stamp}.jpg"))
            try:
                k_cloud = self.backend.K if hasattr(self.backend, "K") else None
                if k_cloud is not None:
                    cloud = make_cloud(frame.depth_mm, frame.bgr,
                                       np.asarray(k_cloud, np.float32),
                                       self.cfg.depth_min_m, self.cfg.depth_max_m)
                    if cloud.n:
                        save_cloud(cloud, str(self.frames_dir / f"{stamp}.cloud"))
            except Exception:
                pass  # облако — не критично для GT; флаги покажут thin
        if pose is not None:
            self._last_pose = pose
        return pose
