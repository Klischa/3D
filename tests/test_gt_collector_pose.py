"""Тесты позы: umeyama, solve_board_pose, relative, конвенция углов."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import pytest
from scipy.spatial.transform import Rotation

from gt_collector.board import board_bitmap, make_detector
from gt_collector.board_pose import (BoardPose, solve_board_pose, umeyama)
from gt_collector.config import BoardSpec
from gt_collector.fake import FakeBackend, FakeConfig
from gt_collector.detectors import DetectedTag


def test_umeyama_recovers_rigid():
    rng = np.random.default_rng(1)
    R = Rotation.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix()
    t = rng.normal(0, 0.5, 3)
    src = rng.normal(0, 0.2, (20, 3))
    dst = src @ R.T + t
    Ru, tu = umeyama(src, dst)
    assert np.degrees(Rotation.from_matrix(Ru @ R.T).magnitude()) < 1e-6
    assert np.linalg.norm(tu - t) < 1e-6


def _pose4(R: np.ndarray, t: np.ndarray) -> BoardPose:
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return BoardPose.from_matrix(m)


def test_board_pose_relative():
    # T_board_from_cam для двух поз; relative(new, old) двигает точку new→old
    R1 = Rotation.from_rotvec([0.1, 0, 0]).as_matrix()
    R2 = Rotation.from_rotvec([0.2, 0.05, 0]).as_matrix()
    p1 = _pose4(R1, np.array([0.1, 0.2, 0.9]))
    p2 = _pose4(R2, np.array([0.12, 0.19, 0.92]))
    T = BoardPose.relative(p2, p1)  # source=p2 (new), target=p1 (old)
    pt_new = np.array([0.01, -0.02, 0.5])
    # вручную: точка в кадре NEW (source=p2) → board → OLD (target=p1)
    T2 = p2.matrix()
    T1 = p1.matrix()
    pt_board = T2[:3, :3] @ pt_new + T2[:3, 3]
    invT1 = np.linalg.inv(T1)
    pt_old_manual = invT1[:3, :3] @ pt_board + invT1[:3, 3]
    pt_old = T[:3, :3] @ pt_new + T[:3, 3]
    assert np.allclose(pt_old, pt_old_manual, atol=1e-12)


def test_solve_board_pose_zero_noise():
    """Scripted-детекции без шума: поза должна совпасть с GT точно."""
    board = BoardSpec.from_preset("a4")
    be = FakeBackend(FakeConfig(duration_s=1.0, fps=30, trajectory="orbit",
                                seed=3, board_preset="a4",
                                det_jitter_px=0.0), board)
    Rg, tg = be._pose(0)
    # точные scripted-детекции
    h = board.tag_mm / 2000
    off = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    K = be.cfg.K
    dets = []
    for k in range(board.tag_count):
        c_b = board.tag_centers_m()[k]
        corners = np.empty((4, 2), np.float32)
        for m, o in enumerate(off):
            pc = Rg @ (c_b + o) + tg
            corners[m] = [K[0, 0] * pc[0] / pc[2] + K[0, 2],
                          K[1, 1] * pc[1] / pc[2] + K[1, 2]]
        dets.append(DetectedTag(k, float(np.mean(corners[:, 0])),
                                float(np.mean(corners[:, 1])), corners))
    pose = solve_board_pose(dets, board, K, None)
    drot = np.degrees(Rotation.from_matrix(pose.R @ Rg.T).magnitude())
    dtr = np.linalg.norm(pose.t - tg) * 1000
    assert drot < 0.01, drot
    assert dtr < 1.0, dtr
    assert pose.mean_residual_mm < 0.01


def test_solve_board_pose_min_tags():
    board = BoardSpec.from_preset("a4")
    with pytest.raises(ValueError):
        solve_board_pose([DetectedTag(0, 100, 100, np.zeros((4, 2), np.float32))],
                         board, np.eye(3, dtype=np.float32))


def test_detector_smoke_real_opencv():
    """Смоук-тест реального OpenCV-детектора на чистом 1:1-битмапе дошки.

    (Содержимое со швом/шумом может попадать в баг привязки OpenCV+numpy —
    см. комментарии в detectors.py; чистый битмап стабilen.)
    """
    board = BoardSpec.from_preset("a4")
    bmp = board_bitmap(board)
    det = make_detector()
    corners, ids, _ = det.detectMarkers(bmp)
    assert ids is not None and len(ids) == board.tag_count
    assert sorted(ids.ravel()) == list(range(board.tag_count))
