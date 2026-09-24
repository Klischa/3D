"""Калибровка и проверка дошки.

* save-calibration — сохранить заводскую калибровку Astra (K + дисторсии
  color/depth, depth→color extrinsic, baseline) в calibration.json.
  Используется как справочная; для GT важны K и дисторсии color.
* check-board — фото напечатанной дошки: детекция + раскладка + размер
  маркеров на экране (проверка перед сессией, с любого расстояния).
* print-board — PNG/SVG дошки для печати.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2

from .board import (board_bitmap, board_print_files, check_layout_matches_decode,
                    make_detector)
from .config import BoardSpec
from .detectors import OpenCVAprilTagDetector


def save_device_calibration(out_path: str, name_hint: str = "astra",
                            serial: str | None = None) -> dict:
    """Заводская калибровка устройства Orbbec (после первого frameset)."""
    from .backends import OrbbecBackend

    be = OrbbecBackend(name_hint=name_hint, serial=serial)
    be.start()
    try:
        calib = be._calib
        data = {
            "source": "device factory (pyorbbecsdk2 Pipeline.get_camera_param)",
            "device": be.metadata(),
            "rgb_intrinsic": {"fx": calib.rgb_intrinsic.fx, "fy": calib.rgb_intrinsic.fy,
                              "cx": calib.rgb_intrinsic.cx, "cy": calib.rgb_intrinsic.cy,
                              "width": calib.rgb_intrinsic.width,
                              "height": calib.rgb_intrinsic.height},
            "depth_intrinsic": {"fx": calib.depth_intrinsic.fx,
                                "fy": calib.depth_intrinsic.fy,
                                "cx": calib.depth_intrinsic.cx,
                                "cy": calib.depth_intrinsic.cy,
                                "width": calib.depth_intrinsic.width,
                                "height": calib.depth_intrinsic.height},
            "rgb_distortion": {"model": int(calib.rgb_distortion.model),
                               "k1": calib.rgb_distortion.k1, "k2": calib.rgb_distortion.k2,
                               "k3": calib.rgb_distortion.k3, "k4": calib.rgb_distortion.k4,
                               "k5": calib.rgb_distortion.k5, "k6": calib.rgb_distortion.k6,
                               "p1": calib.rgb_distortion.p1, "p2": calib.rgb_distortion.p2},
            "depth_distortion": {"model": int(calib.depth_distortion.model),
                                 "k1": calib.depth_distortion.k1,
                                 "k2": calib.depth_distortion.k2,
                                 "k3": calib.depth_distortion.k3,
                                 "k4": calib.depth_distortion.k4,
                                 "k5": calib.depth_distortion.k5,
                                 "k6": calib.depth_distortion.k6,
                                 "p1": calib.depth_distortion.p1,
                                 "p2": calib.depth_distortion.p2},
            "depth_to_color": {
                "rotation": np.asarray(calib.transform.rot, float).reshape(3, 3).tolist(),
                "translation_mm": np.asarray(calib.transform.transform, float).tolist(),
            },
        }
        try:
            data["baseline_mm"] = float(be.dev.get_baseline().baseline)
        except Exception:
            pass
        Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        return data
    finally:
        be.stop()


def check_board_image(image_path: str, board: BoardSpec | None = None,
                      calibration_path: str | None = None) -> dict:
    """Диагностика фото дошки: сколько маркеров, размер на экране, резидуалы."""
    board = board or BoardSpec.from_preset("a4")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {image_path}")
    if calibration_path:
        cal = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        k = cal["rgb_intrinsic"]
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], np.float32)
        d = cal["rgb_distortion"]
        dist = np.array([d["k1"], d["k2"], d["p1"], d["p2"],
                         d["k3"], d["k4"], d["k5"], d["k6"]], np.float32) \
            if d.get("model") == 1 else np.array([d["k1"], d["k2"], d["p1"], d["p2"]], np.float32)
        img = cv2.undistort(img, K, dist)
    det = OpenCVAprilTagDetector()
    tags = det.detect(img)
    sizes = []
    for t in tags:
        w = float(np.linalg.norm(t.corners[0] - t.corners[1]))
        h = float(np.linalg.norm(t.corners[1] - t.corners[2]))
        sizes.append((w + h) / 2.0)
    return {
        "image": image_path,
        "board": board.size(),
        "n_tags": len(tags),
        "expected_tags": board.tag_count,
        "ok": len(tags) == board.tag_count,
        "open_cv_errors": det.error_count,
        "tag_ids": sorted(t.tag_id for t in tags),
        "tag_size_px": ({"min": float(min(sizes)), "p50": float(np.median(sizes)),
                         "max": float(max(sizes))} if sizes else None),
    }


def print_board(board: BoardSpec, out_png: str, out_svg: str) -> tuple[str, str]:
    return board_print_files(board, out_png, out_svg)


def verify_board(board: BoardSpec) -> int:
    """Самодиагностика раскладки (декодирование собственного битмапа)."""
    return check_layout_matches_decode(board, detector=make_detector())
