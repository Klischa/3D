"""Детекция AprilTag в RGB-кадре (OpenCV) и интерфейс для фейка.

Примечание о надёжности: в сборках opencv-python-headless 4.10–4.14 в паре с
numpy 2.x отдельные вызовы ``detectMarkers``/``solvePnP`` на содержимом
изображения, которое содержит кандидатов, могут бросать
``TypeError: return arrays must be of ArrayType`` (баг привязки Python,
зависит от состояния процесса). Обход: явная CORNER_REFINE_SUBPIX,
пересоздание детектора и повтор на альтернативном представлении кадра.
В FakeBackend детекция scripted (без OpenCV), поэтому сбор на фейке
детерминирован.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cv2
from .board import make_detector


@dataclass(frozen=True)
class DetectedTag:
    tag_id: int
    cx: float          # пиксели, субпиксельный центр
    cy: float
    corners: np.ndarray  # (4,2) пиксели, порядок TL, TR, BR, BL (y вниз)


class TagDetector:
    """Интерфейс: RGB-кадр (BGR) → список маркеров."""

    def detect(self, bgr: np.ndarray) -> list[DetectedTag]:
        raise NotImplementedError

    @property
    def error_count(self) -> int:
        """Количество кадров, где OpenCV бросил исключение (статистика)."""
        return 0


class OpenCVAprilTagDetector(TagDetector):
    """cv2.aruco с SUBPIX-уточнением углов и защитой от бага привязки."""

    def __init__(self, max_image_width: int = 0):
        self._max_width = max_image_width  # 0 = без даунскейла
        self._detector = make_detector()
        self._errors = 0

    @property
    def error_count(self) -> int:
        return self._errors

    def _detect_once(self, img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._detector.detectMarkers(img)[:2]

    def detect(self, bgr: np.ndarray) -> list[DetectedTag]:
        img = bgr
        pix_scale = 1.0
        if self._max_width and img.shape[1] > self._max_width:
            scale = self._max_width / img.shape[1]
            img = cv2.resize(img, (self._max_width, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
            pix_scale = 1.0 / scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # варианты: исходный → BGR-«круг» (сбрасывает состояние привязки) → сдвиг
        variants = [gray,
                    cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY),
                    np.roll(gray, 1, axis=1)]
        corners, ids = None, None
        for k, v in enumerate(variants):
            try:
                corners, ids = self._detect_once(v)
                if ids is not None:
                    break
            except TypeError:
                self._errors += 1
                self._detector = make_detector()
                if k == len(variants) - 1:
                    return []
        if ids is None:
            return []
        out: list[DetectedTag] = []
        for c, id_i in zip(corners, ids.ravel()):
            cc = c.reshape(4, 2) * pix_scale
            out.append(DetectedTag(int(id_i), float(cc[:, 0].mean()),
                                   float(cc[:, 1].mean()), cc))
        return out


class ScriptedDetector(TagDetector):
    """Детектор по callback'у (FakeBackend: детекции из GT-позы кадра)."""

    def __init__(self, fn):
        self._fn = fn

    def detect(self, bgr: np.ndarray) -> list[DetectedTag]:
        return self._fn(bgr)
