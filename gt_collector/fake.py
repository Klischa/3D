"""Синтетический бэкенд Astra: рендерит RGB + depth из GT-позы камеры.

Мир = координаты дошки. Камера движется по траектории (orbit / static /
sweep), дошка статична. RGB — 4-угловая perspective-warp битмапа дошки
(точное OpenCV-согласие: углы проецируются тем же ``projectPoints``),
шум сенсора + JPEG. Depth — аналитическое пересечение лучей с плоскостью
дошки и «стеной» фона, шум + dropout (как у реального ToF-сенсора).
Детекции — scripted из GT-позы (субпиксельный джиттер + дропаут): фейк
валидирует весь пайплайн после детектора, а не сам OpenCV (OpenCV отдельно
смок-тестится на чистых кадрах в тестах).
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from dataclasses import dataclass, field

from .board import BASE_PPM, board_bitmap, board_corners_m
from .detectors import DetectedTag, ScriptedDetector


@dataclass
class FakeConfig:
    width: int = 1280
    height: int = 720
    fx: float = 620.0
    fy: float = 620.0
    cx: float = 640.0
    cy: float = 360.0
    board_preset: str = "a4"
    duration_s: float = 60.0
    fps: int = 30
    seed: int = 11

    # траектория: 'orbit' | 'static' | 'sweep'
    trajectory: str = "orbit"
    orbit_radius_m: float = 1.0
    orbit_period_s: float = 45.0
    orbit_elev_offset_m: float = -0.10
    sweep_amplitude_deg: float = 30.0

    # сенсор
    noise_sigma: float = 2.0        # std RGB-шума, уровней
    jpeg_quality: int = 90
    depth_noise_mm: float = 1.5
    depth_dropout: float = 0.008
    bg_depth_m: float = -2.4        # плоскость фона в дошке (за дошкой)

    # scripted-детектор
    det_jitter_px: float = 0.12
    det_tag_dropout: float = 0.02

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float32)

    def num_frames(self) -> int:
        return int(round(self.duration_s * self.fps))


def look_at(P: np.ndarray, C: np.ndarray = np.zeros(3)) -> tuple[np.ndarray, np.ndarray]:
    """R, t (board_from_cam) для камеры в точке P, смотрящей на C."""
    P = np.asarray(P, float)
    C = np.asarray(C, float)
    Zc = C - P
    Zc /= np.linalg.norm(Zc)
    up = np.array([0.0, -1.0, 0.0])  # «вверх» мира = -Y дошки (Y дошки вниз)
    Xc = np.cross(up, Zc)
    Xc /= np.linalg.norm(Xc)
    Yc = np.cross(Zc, Xc)
    R = np.vstack([Xc, Yc, Zc])
    t = -R @ P
    return R, t


class FakeBackend:
    """Бэкенд-фейк с тем же интерфейсом, что и OrbbecBackend."""

    def __init__(self, cfg: FakeConfig, board=None):
        if cfg.trajectory not in ("orbit", "static", "sweep"):
            raise ValueError(f"Неизвестная траектория: {cfg.trajectory!r}")
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._idx = -1
        self._board = board
        if board is None:
            from .config import BoardSpec
            self._board = BoardSpec.from_preset(cfg.board_preset)
        self._board_bmp = board_bitmap(self._board, BASE_PPM)
        self._corners_b = board_corners_m(self._board)
        self._tag_centers = self._board.tag_centers_m()
        self._det_fn = None

    # ---- траектория -----------------------------------------------------
    def _pose(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        t = idx / c.fps
        if c.trajectory == "static":
            wob = (0.0015 * np.array([math.sin(2 * math.pi * t / 3.7),
                                      math.sin(2 * math.pi * t / 5.1 + 1.0),
                                      math.sin(2 * math.pi * t / 4.3 + 2.0)]))
            P = np.array([0.0, -0.2, 1.0]) + wob
        elif c.trajectory == "orbit":
            w = 2 * math.pi / c.orbit_period_s
            th = w * t
            r = c.orbit_radius_m
            P = np.array([r * math.cos(th),
                          c.orbit_elev_offset_m + 0.05 * math.sin(2 * math.pi * t / 9.0),
                          r * math.sin(th)])
        else:  # sweep — качание по дуге
            w = 2 * math.pi / c.orbit_period_s
            a = math.radians(c.sweep_amplitude_deg) * math.sin(w * t)
            P = np.array([c.orbit_radius_m * math.cos(a),
                          c.orbit_elev_offset_m,
                          c.orbit_radius_m * math.sin(a)])
        return look_at(P)

    # ---- RGB ------------------------------------------------------------
    def _render_rgb(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        c = self.cfg
        K = c.K
        rvec = cv2.Rodrigues(R.astype(np.float64))[0].astype(np.float32)
        pts, _ = cv2.projectPoints(self._corners_b.reshape(-1, 1, 3).astype(np.float32),
                                   rvec, t.astype(np.float32), K, np.zeros(4, np.float32))
        pts = pts.reshape(-1, 2).astype(np.float32)
        # размер дошки на экране → разрешение источника ≈ 3× (антиалиасинг)
        screen_w = float(np.linalg.norm(pts[0] - pts[1]))
        scale = max(1.0, 3.0 * screen_w / (self._board.width_m * BASE_PPM))
        bmp = self._board_bmp
        if scale != 1.0:
            bmp = cv2.resize(bmp, (int(bmp.shape[1] * scale), int(bmp.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        src = np.array([[0, 0], [bmp.shape[1], 0], [bmp.shape[1], bmp.shape[0]], [0, bmp.shape[0]]],
                       np.float32)
        M = cv2.getPerspectiveTransform(src, pts)
        img = cv2.warpPerspective(bmp, M, (c.width, c.height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=128)
        if c.noise_sigma > 0:
            img = (img.astype(np.float32) +
                   self._rng.normal(0.0, c.noise_sigma, img.shape)).clip(0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), c.jpeg_quality])
        if not ok:
            raise RuntimeError("Не удалось закодировать JPEG в фейке")
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ---- depth ----------------------------------------------------------
    def _render_depth(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        c = self.cfg
        fx, fy, cx, cy = c.fx, c.fy, c.cx, c.cy
        u = np.arange(c.width, dtype=np.float32)
        v = np.arange(c.height, dtype=np.float32)
        UU, VV = np.meshgrid(u, v)
        dx = (UU - cx) / fx
        dy = (VV - cy) / fy
        # лучи в системе дошки: p_b(s) = o_b + s * d_b
        Rcb = R  # board_from_cam
        o_b = -Rcb.T @ t                      # (3,)
        d_b = (Rcb.T @ np.stack([dx, dy, np.ones_like(dx)]).reshape(3, -1)).reshape(3, c.height, c.width)
        norm = np.sqrt(d_b[0] ** 2 + d_b[1] ** 2 + d_b[2] ** 2)

        hw, hh = self._board.width_m / 2.0 + 0.002, self._board.height_m / 2.0 + 0.002
        dbz = d_b[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s_board = -o_b[2] / dbz
        pb = o_b[:, None, None] + s_board * d_b
        on_board = (dbz > 0) & (s_board > 0) & (np.abs(pb[0]) <= hw) & (np.abs(pb[1]) <= hh)
        # фон: бесконечный задник на bg_depth_m впереди камеры (⊥ оптической оси) —
        # виден с любой точки орбиты, как в студии
        depth_mm = np.where(on_board,
                            s_board * norm * 1000.0,
                            np.abs(c.bg_depth_m) * norm * 1000.0)
        # шум + dropout только там, где есть точка
        if c.depth_noise_mm > 0:
            depth_mm = depth_mm + np.where(depth_mm > 0,
                                           self._rng.normal(0.0, c.depth_noise_mm, depth_mm.shape), 0.0)
        if c.depth_dropout > 0:
            depth_mm = np.where(self._rng.random(depth_mm.shape) < c.depth_dropout, 0.0, depth_mm)
        return np.clip(depth_mm, 0.0, 65535.0).astype(np.float32)

    # ---- scripted-детекция ----------------------------------------------
    def _detections(self, R: np.ndarray, t: np.ndarray) -> list[DetectedTag]:
        c = self.cfg
        K = c.K
        out: list[DetectedTag] = []
        half = self._board.tag_mm / 2000.0
        offs = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
        for k in range(self._board.tag_count):
            if self._rng.random() < c.det_tag_dropout:
                continue
            center_b = self._tag_centers[k]
            pc = R @ center_b + t
            if pc[2] < 0.05:
                continue
            u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
            v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
            if not (0 <= u < c.width and 0 <= v < c.height):
                continue
            if self._rng.random() < c.det_tag_dropout:
                continue
            j = self._rng.normal(0.0, c.det_jitter_px, 2)
            corners = np.empty((4, 2), np.float32)
            for m in range(4):
                q = R @ (center_b + offs[m]) + t
                qu = K[0, 0] * q[0] / q[2] + K[0, 2] + j[0] + self._rng.normal(0.0, 0.04)
                qv = K[1, 1] * q[1] / q[2] + K[1, 2] + j[1] + self._rng.normal(0.0, 0.04)
                corners[m] = (qu, qv)
            out.append(DetectedTag(k, float(u + j[0]), float(v + j[1]), corners))
        return out

    # ---- интерфейс бэкенда ----------------------------------------------
    @property
    def K(self) -> np.ndarray:
        return self.cfg.K

    @property
    def K_undistorted(self) -> np.ndarray:
        return self.cfg.K

    @property
    def detector(self):
        if self._det_fn is None:
            self._det_fn = ScriptedDetector(lambda _bgr: list(self._last_dets))
        return self._det_fn

    def start(self) -> None:
        self._idx = -1
        self._last_dets: list[DetectedTag] = []

    def wait_for_frame(self, timeout_ms: int = 1000):
        from .backends import FrameData
        self._idx += 1
        if self._idx >= self.cfg.num_frames():
            return None
        R, t = self._pose(self._idx)
        bgr = self._render_rgb(R, t)
        depth = self._render_depth(R, t)
        self._last_dets = self._detections(R, t)
        return FrameData(bgr=bgr, depth_mm=depth,
                         ts_us=self._idx * int(1_000_000 / self.cfg.fps),
                         extra={"gt_R": R, "gt_t": t, "backend": "fake"})

    def stop(self) -> None:
        pass

    def metadata(self) -> dict:
        c = self.cfg
        return {"backend": "fake", "board": self._board.size(),
                "K": c.K.tolist(), "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "trajectory": c.trajectory, "seed": c.seed,
                "fps": c.fps, "width": c.width, "height": c.height,
                "calibration_source": "synthetic"}
