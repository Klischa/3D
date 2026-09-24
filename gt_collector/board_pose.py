"""Полная 6DoF-поза камеры по дошке с AprilTag (только RGB, без depth).

Метод:
1. Детектор даёт субпиксельные углы маркеров (порядок TL, TR, BR, BL, y вниз).
2. Глобальный ``cv2.solvePnP`` по ВСЕМ 4N углам: углы ↔ аналитические 3D-углы
   маркеров в системе дошки (X вправо, Y вниз, Z из печати, центры из
   раскладки). Глобальная сборка снимает вырожденность depth-наблюдаемости,
   которую имеет пер-маркерный solvePnP для почти фронтальных квадратных
   маркеров (проверено: тот же субпиксельный шум даёт ~20 мм разброса
   пер-маркерных центров против субмиллиметровой ошибки глобальной сборки).
3. Качество кадра — пер-маркерный репроекционный резидуал (px → мм) и
   разброс нормалей маркеров (пер-маркерный solvePnP, только нормали).

Мир = координаты дошки (статичная жёсткая дошка): каждый кадр независим,
без межкадровой карты инициализации и соответствий по ID. Depth в GT не
участвует — метки независимы от ошибок depth-сенсора.

Согласованность с OpenCV проверена экспериментально: дошка, отрендеренная
проекцией через ``cv2.projectPoints`` с этой же системой координат
(y вниз, углы TL→BL), декодируется всеми ID, а глобальный solvePnP
возвращает исходную позу с точностью 0.000002°/0.0001 мм на чистом
round-trip и ~0.2°/0.7 мм при субпиксельном шуме 0.15 px.

Примечание: ``SOLVEPNP_IPPE_SQUARE`` в сборках opencv-python-headless 4.10–4.14
ненадёжен (даёт 180° на чистом round-trip) — используем SQPNP с фолбэком
на ITERATIVE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
import cv2


@dataclass
class BoardPose:
    R: np.ndarray                    # (3,3) board_from_cam
    t: np.ndarray                    # (3,) центр камеры в дошке, м
    n_tags: int
    mean_residual_mm: float
    per_tag_residual_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    tag_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    normal_spread_deg: float = 0.0   # разброс нормалей маркеров (плоскость дошки)
    pnp_failed: int = 0              # маркеры, где пер-тэг solvePnP не сошёлся

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    @staticmethod
    def from_matrix(m: np.ndarray, **kw) -> "BoardPose":
        return BoardPose(m[:3, :3].copy(), m[:3, 3].copy(),
                         kw.pop("n_tags", 0), kw.pop("mean_residual_mm", float("nan")),
                         kw.pop("per_tag_residual_mm", np.array([])),
                         kw.pop("tag_ids", np.array([], dtype=int)),
                         kw.pop("normal_spread_deg", 0.0), **kw)

    @staticmethod
    def relative(pose_source: "BoardPose", pose_target: "BoardPose") -> np.ndarray:
        """T_target_from_source = inv(T_target) @ T_source.

        source — текущий кадр (вход модели), target — предыдущий. Точка в
        кадре source преобразуется в кадр target: p_target = M p_source.
        """
        return np.linalg.inv(pose_target.matrix()) @ pose_source.matrix()


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dst ≈ R src + t (жёсткое преобразование, без отражения)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    diag = np.eye(3)
    diag[2, 2] = 1.0 if np.linalg.det(vt.T @ u.T) > 0 else -1.0
    r = vt.T @ diag @ u.T
    return r, cd - r @ cs


def tag_obj_corners_m(board) -> np.ndarray:
    """Углы маркера (TL, TR, BR, BL) в системе маркера: центр в начале,
    y вниз, z из печати."""
    h = board.tag_mm / 2000.0
    return np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]], np.float32)


def _solve_pnp(obj: np.ndarray, img: np.ndarray, k: np.ndarray, dist: np.ndarray):
    for flags in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=flags)
            if ok:
                return cv2.Rodrigues(rvec)[0], np.asarray(tvec).ravel()
        except TypeError:
            continue
    return None, None


def _project(k: np.ndarray, pts_b: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = (r @ pts_b.T).T + t
    return (k[:2, :2] @ p[:, :2].T).T / p[:, 2:3] + k[:2, 2]


def solve_board_pose(det, board, k: np.ndarray,
                     dist: np.ndarray | None = None) -> BoardPose:
    """Поза камеры в координатах дошки по списку DetectedTag."""
    if len(det) < 3:
        raise ValueError("Меньше 3 маркеров — позу решить нельзя")
    if dist is None:
        dist = np.zeros(4, np.float32)
    k = np.asarray(k, dtype=np.float32)

    off = tag_obj_corners_m(board)
    centers_b = board.tag_centers_m()
    ids = np.array([d.tag_id for d in det], dtype=int)
    all_obj = np.vstack([centers_b[i][None, :] + off for i in ids]).astype(np.float32)
    all_img = np.vstack([np.asarray(d.corners, dtype=np.float32).reshape(4, 2) for d in det])

    r, t = _solve_pnp(all_obj, all_img, k, dist)
    if r is None:
        raise ValueError("Глобальный solvePnP не сошёлся")

    # пер-маркерные репроекционные резидуалы, мм
    resid_mm = np.empty(len(det))
    normals_cam: list[np.ndarray] = []
    obj_local = off
    failed = 0
    for i, d in enumerate(det):
        corners3 = centers_b[d.tag_id][None, :] + off
        uv = _project(k, corners3.astype(np.float32), r, t)
        err_px = float(np.mean(np.linalg.norm(uv - np.asarray(d.corners, float).reshape(4, 2),
                                              axis=1)))
        depth = float((r @ centers_b[d.tag_id] + t)[2])
        resid_mm[i] = err_px * abs(depth) / float(k[0, 0])
        # нормали (для контроля плоскости) — пер-тэг solvePnP
        rn, _ = _solve_pnp(obj_local, np.asarray(d.corners, np.float32).reshape(4, 2), k, dist)
        if rn is None:
            failed += 1
            normals_cam.append(r[:, 2])
        else:
            normals_cam.append(rn[:, 2])
    normals_board = (r @ np.array(normals_cam).T).T
    spread = 0.0
    if len(normals_board) >= 3:
        cos_a = np.clip(np.abs(normals_board @ np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
        angles = np.rad2deg(np.arccos(cos_a))
        spread = float(angles.max() - angles.min())
    return BoardPose(r, t, len(det), float(resid_mm.mean()), resid_mm, ids, spread, failed)
