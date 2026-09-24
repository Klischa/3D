"""Тесты дошки: геометрия, генерация, декодирование раскладки, печатные файлы."""
import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")  # gt_collector требует OpenCV: пропуск без него

from gt_collector.config import BoardSpec
from gt_collector.board import (BASE_PPM, board_bitmap, board_corners_m,
                                check_layout_matches_decode, make_detector,
                                tag_bitmap)
from gt_collector.calibrate import print_board


def test_preset_a4_geometry():
    b = BoardSpec.from_preset("a4")
    assert b.tag_count == 12
    # 2*15 + 4*50 + 3*10 = 260 мм; 2*15 + 3*50 + 2*10 = 200 мм
    assert b.width_m == pytest.approx(0.26)
    assert b.height_m == pytest.approx(0.20)
    c = b.tag_centers_m()
    assert c.shape == (12, 3)
    assert np.allclose(c[0], [-0.09, -0.06, 0.0])
    assert np.allclose(c[11], [0.09, 0.06, 0.0])
    assert np.allclose(c[:, 2], 0.0)


def test_too_many_tags_rejected():
    with pytest.raises(ValueError):
        BoardSpec(6, 5, 30, 10, 10)  # 30 > 25 маркеров в словаре
    with pytest.raises(ValueError):
        BoardSpec.from_preset("нет-такого")


def test_tag_bitmap_black_border():
    # margin=0: внешнее чёрное кольцо прилегает к краю битмапа
    t = tag_bitmap(None, 3, 25)
    assert t.shape == (25, 25)
    assert t.min() == 0
    for row in (0, 1, 23, 24):
        assert t[row, :].min() == 0
        assert t[row, :].max() == 255 or t[row, :].mean() < 128
    for col in (0, 1, 23, 24):
        assert t[:, col].min() == 0


def test_board_bitmap_size_and_decode():
    b = BoardSpec.from_preset("a4")
    bmp = board_bitmap(b)
    assert bmp.shape[1] == pytest.approx(b.width_m * BASE_PPM, abs=2)
    assert bmp.shape[0] == pytest.approx(b.height_m * BASE_PPM, abs=2)
    # раскладка: все 12 ID, 0..11 по строкам
    n = check_layout_matches_decode(b, detector=make_detector())
    assert n == 12


def test_corners_order():
    b = BoardSpec.from_preset("a4")
    c = board_corners_m(b)
    assert c.shape == (4, 3)
    assert np.allclose(c[0], [-0.13, -0.10, 0.0])
    assert np.allclose(c[1], [0.13, -0.10, 0.0])
    assert np.allclose(c[2], [0.13, 0.10, 0.0])
    assert np.allclose(c[3], [-0.13, 0.10, 0.0])


def test_check_board_image(tmp_path):
    """check_board_image на собственном 1:1-рендре (реальный OpenCV-детектор)."""
    from gt_collector.calibrate import check_board_image
    b = BoardSpec.from_preset("a4")
    bmp = board_bitmap(b)
    img = cv2.cvtColor(bmp, cv2.COLOR_GRAY2BGR)
    p = str(tmp_path / "board_photo.jpg")
    cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    res = check_board_image(p, b)
    assert res["ok"]
    assert res["n_tags"] == 12
    assert res["tag_size_px"]["p50"] > 150  # 50 мм при 4000 px/м = 200 px


def test_print_files(tmp_path):
    b = BoardSpec.from_preset("a4")
    png = str(tmp_path / "board.png")
    svg = str(tmp_path / "board.svg")
    p, s = print_board(b, png, svg)
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    # ~305 DPI: 260 мм → ~3120 px
    assert 3000 < img.shape[1] < 3300
    data = open(s, encoding="utf-8").read()
    assert data.count("<image ") == 12
    assert 'width="260mm"' in data
