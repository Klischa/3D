"""Тесты формата облака ACGD и make_cloud."""
import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")  # gt_collector требует OpenCV: пропуск без него

from gt_collector.cloud import (Cloud, load_cloud, make_cloud, pack_cloud,
                                save_cloud, unpack_cloud)


def _synthetic_cloud(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n, 3), np.float32)
    xyz[:, 0] = rng.normal(0, 0.2, n)
    xyz[:, 1] = rng.normal(0, 0.2, n)
    xyz[:, 2] = rng.uniform(0.4, 3.5, n)
    rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    return Cloud(xyz, rgb)


def test_pack_unpack_roundtrip():
    c = _synthetic_cloud()
    c2 = unpack_cloud(pack_cloud(c))
    assert c2.n == c.n
    # 1-мм квантование
    assert np.allclose(c2.xyz, np.round(c.xyz * 1000) / 1000, atol=0.001)
    assert np.array_equal(c2.rgb, c.rgb)


def test_file_roundtrip(tmp_path):
    c = _synthetic_cloud()
    p = str(tmp_path / "x.cloud")
    save_cloud(c, p)
    c2 = load_cloud(p)
    assert c2.n == c.n
    assert np.allclose(c2.xyz, c.xyz, atol=0.002)


def test_bad_magic(tmp_path):
    bad = b"XXXX" + pack_cloud(_synthetic_cloud(10))[4:]
    with pytest.raises(ValueError):
        unpack_cloud(bad)


def test_truncated(tmp_path):
    data = pack_cloud(_synthetic_cloud(100))
    with pytest.raises(ValueError):
        unpack_cloud(data[: len(data) - 20])


def test_empty_rejected():
    with pytest.raises(ValueError):
        pack_cloud(Cloud(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)))


def test_make_cloud_projection():
    # плоскость z=1 м на всю картинку; точка (640,360) → (0,0,1)
    h, w = 240, 320
    depth = np.full((h, w), 1000.0, np.float32)
    color = np.zeros((h, w, 3), np.uint8)
    color[h // 2, w // 2] = (10, 20, 30)  # BGR
    k = np.array([[620, 0, 160], [0, 620, 120], [0, 0, 1]], np.float32)
    cloud = make_cloud(depth, color, k, 0.3, 4.0)
    assert cloud.n == h * w
    # цвет центра
    idx = (cloud.xyz[:, 0] ** 2 + cloud.xyz[:, 1] ** 2).argmin()
    assert cloud.xyz[idx] == pytest.approx([0, 0, 1.0], abs=1e-3)
    assert list(cloud.rgb[idx]) == [30, 20, 10]  # RGB
    # точка в углу (0,0): x = (0-160)/620, y = (0-120)/620
    i2 = (np.arange(w)[None, :] == 0).nonzero()
    x0 = cloud.xyz[i2[0][0], 0]
    assert x0 == pytest.approx(-160 / 620, abs=1e-3)


def test_make_cloud_size_mismatch():
    depth = np.zeros((10, 10), np.float32)
    color = np.zeros((9, 9, 3), np.uint8)
    k = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        make_cloud(depth, color, k, 0.3, 4.0)
