"""Генерация дошки с AprilTag (PNG/SVG) и её геометрия.

Словарь: ``cv2.aruco.DICT_APRILTAG_25H9`` — семейство AprilTag 3.6 (25
маркеров, ориентация закодирована в битах — нет неоднозначности поворота,
в отличие от классических ArUco).

Сборка дошки — поштучно через ``cv2.aruco.generateImageMarker`` (PyPI-сборки
OpenCV 4.9–4.14 не содержат ``ArucoGeneratorImage``): каждый маркер рендерим
с ``markerImageBorderSize=0`` (внешнее чёрное кольцо маркера прилегает к
краю битмапа — проверено) и вклеиваем на белое полотно в строгом порядке
сетки. После сборки ``check_layout_matches_decode`` декодирует результат и
убеждается, что ID идут 0..N-1 по строкам — защита от любой смены порядка.

Система координат дошки (мировая для сессии): начало — в центре дошки,
X вправо, Y вниз, Z из лицевой (печатной) стороны. Битмап рисуется в той же
системе (строка 0 битмапа = верх листа, y вниз) — это же соглашение даёт
правильную детекцию после проекции (проверено: 4-угловая warp через
projectPoints декодирует все 12 ID).
"""
from __future__ import annotations

import base64
import io
import numpy as np
import cv2

DICT_ID = cv2.aruco.DICT_APRILTAG_25H9
DICT_NAME = "DICT_APRILTAG_25H9"

#: базовое разрешение битмапа дошки, px/м (рендеринг фейка и проверка раскладки)
BASE_PPM = 4000.0
#: разрешение печатного PNG, px/м (~305 DPI)
PRINT_PPM = 12000.0


def dictionary() -> "cv2.aruco.Dictionary":
    return cv2.aruco.getPredefinedDictionary(DICT_ID)


def make_detector() -> "cv2.aruco.ArucoDetector":
    """Детектор с явной CORNER_REFINE_SUBPIX.

    Важно: OpenCV по умолчанию выбирает CORNER_REFINE_APRILTAG для словаря
    AprilTag; в сборках opencv-python-headless 4.10–4.14 + numpy 2.x эта
    ветка нестабильна (``TypeError: return arrays must be of ArrayType``) и
    детектирует хуже. SUBPIX проверен: 12/12 на всех рабочих расстояниях.
    """
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary(), dp)


def tag_bitmap(board, tag_id: int, side_px: int = 200) -> np.ndarray:
    """Ч/б битмап одного маркера (чёрное кольцо на краю, margin=0)."""
    m = cv2.aruco.generateImageMarker(dictionary(), int(tag_id), 25, 0)
    if m is None:
        raise RuntimeError(f"OpenCV не смог сгенерировать маркер {tag_id}")
    if side_px == 25:
        return m
    return cv2.resize(m, (side_px, side_px), interpolation=cv2.INTER_AREA)


def board_bitmap(board, ppm: float = BASE_PPM) -> np.ndarray:
    """Битмап дошки (ч/б), верхний левый пиксель = верхний левый угол листа.

    Физический размер изображения = width_m × height_m при ppm px/м.
    """
    tag_px = int(round(board.tag_mm / 1000.0 * ppm))
    gap_px = int(round(board.gap_mm / 1000.0 * ppm))
    margin_px = int(round(board.margin_mm / 1000.0 * ppm))
    if tag_px < 25:
        raise ValueError("Разрешение слишком мало: маркер < 25 px")
    W = margin_px * 2 + board.cols * tag_px + (board.cols - 1) * gap_px
    H = margin_px * 2 + board.rows * tag_px + (board.rows - 1) * gap_px
    out = np.full((H, W), 255, np.uint8)
    for i in range(board.rows):
        for j in range(board.cols):
            k = i * board.cols + j
            y0 = margin_px + i * (tag_px + gap_px)
            x0 = margin_px + j * (tag_px + gap_px)
            out[y0:y0 + tag_px, x0:x0 + tag_px] = tag_bitmap(board, k, tag_px)
    return out


def _tag_png_b64(board, tag_id: int, side_px: int = 400) -> str:
    m = tag_bitmap(board, tag_id, side_px)
    ok, buf = cv2.imencode(".png", m)
    if not ok:
        raise RuntimeError("Не удалось закодировать PNG маркера")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def board_print_files(board, out_png: str, out_svg: str) -> tuple[str, str]:
    """Печатные файлы: PNG высокого разрешения и SVG (масштаб 1:1 в мм).

    SVG встраивает PNG каждого маркера base64-ом — печатается без потерь
    при любом масштабе, размеры указаны в миллиметрах.
    """
    bmp = board_bitmap(board, PRINT_PPM)
    ok, buf = cv2.imencode(".png", bmp)
    if not ok:
        raise RuntimeError("Не удалось закодировать PNG дошки")
    with open(out_png, "wb") as f:
        f.write(buf.tobytes())

    w_mm, h_mm = board.width_m * 1000.0, board.height_m * 1000.0
    tag_mm, gap_mm, margin_mm = board.tag_mm, board.gap_mm, board.margin_mm
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_mm:g}mm" height="{h_mm:g}mm" '
        f'viewBox="0 0 {w_mm:g} {h_mm:g}">',
        f'  <rect x="0" y="0" width="{w_mm:g}" height="{h_mm:g}" fill="#ffffff"/>',
        "  <!-- AprilTag-дошка: мир = координаты дошки (X вправо, Y вниз, Z из печати). "
        "Маркеры нумеруются 0..N-1 по строкам слева направо. -->",
    ]
    for i in range(board.rows):
        for j in range(board.cols):
            k = i * board.cols + j
            x = margin_mm + j * (tag_mm + gap_mm)
            y = margin_mm + i * (tag_mm + gap_mm)
            b64 = _tag_png_b64(board, k)
            parts.append(
                f'  <image x="{x:g}" y="{y:g}" width="{tag_mm:g}" height="{tag_mm:g}" '
                f'href="data:image/png;base64,{b64}" '
                f'preserveAspectRatio="none"/>'
            )
    parts.append("</svg>")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    return out_png, out_svg


def board_corners_m(board) -> np.ndarray:
    """4 угла дошки (TL, TR, BR, BL) в координатах дошки — для проекции."""
    hw, hh = board.width_m / 2.0, board.height_m / 2.0
    return np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]])


def check_layout_matches_decode(board, ppm: float = BASE_PPM, detector=None) -> int:
    """Декодирует битмап дошки и проверяет раскладку ID 0..N-1 по строкам.

    Возвращает количество распознанных маркеров. Бросает RuntimeError при
    любом несоответствии — эту функцию запускает selftest перед сессией.
    """
    bmp = board_bitmap(board, ppm)
    det = detector if detector is not None else make_detector()
    corners, ids, _ = det.detectMarkers(bmp)
    if ids is None or len(ids) != board.tag_count:
        n = 0 if ids is None else len(ids)
        raise RuntimeError(f"Декодирование дошки: найдено {n}, ожидалось {board.tag_count}")
    centers = np.array([c.reshape(4, 2).mean(axis=0) for c in corners])
    ids = ids.ravel()
    # пиксели → метры в системе дошки (центр — (0,0), y вниз)
    scale = board.width_m / bmp.shape[1]
    centers_m = (centers - np.array([bmp.shape[1] / 2.0, bmp.shape[0] / 2.0])) * scale
    expected = board.tag_centers_m()
    for k in range(board.tag_count):
        pos = int(np.argmin(np.linalg.norm(centers_m - expected[k][:2], axis=1)))
        if ids[pos] != k:
            raise RuntimeError(
                f"Маркер на позиции ID {k} декодирован как {ids[pos]} — раскладка нарушена")
        if np.linalg.norm(centers_m[pos] - expected[k][:2]) > 1.5 * scale:
            raise RuntimeError(f"Позиция маркера {k} не совпадает с раскладкой")
    return int(board.tag_count)
