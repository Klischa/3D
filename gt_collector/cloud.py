"""Формат облака в кадре камеры и его компактная упаковка.

`.cloud` — бинарный файл:
  magic  b'ACGD'      (4)
  version uint16 LE   (2)  = 2
  n_points uint32 LE  (4)
  per point: x,y,z int16 LE (мм, от оптического центра камеры; диапазон
  ±32,7 м — хватает на любую сцену; z > 0), r,g,b uint8
Итого 9 байт на точку (1-мм квантование достаточно для noise floor Astra).

v1 хранил uint16 (отрицательные координаты обрезались) — несовместимо.
"""
from dataclasses import dataclass
import struct
import numpy as np
import cv2

MAGIC = b"ACGD"
VERSION = 2


@dataclass
class Cloud:
    xyz: np.ndarray   # (N,3) float32, метры, координаты камеры
    rgb: np.ndarray   # (N,3) uint8

    @property
    def n(self) -> int:
        return len(self.xyz)


def make_cloud(depth_mm: np.ndarray, color_bgr: np.ndarray, k: np.ndarray,
               z_min_m: float, z_max_m: float) -> Cloud:
    """Дискретизированный RGB-D → облако в координатах камеры (aligned depth).

    depth_mm — (H,W) float32 в мм, уже выравненный по цвету; k — интросики
    цветного кадра (fx,fy,cx,cy). Пустые/неконечные пиксели отбрасываются.
    """
    h, w = depth_mm.shape
    if color_bgr.shape[:2] != (h, w):
        raise ValueError("Размеры color и depth не совпадают (нужен aligned depth)")
    valid = (depth_mm > 0) & np.isfinite(depth_mm)
    z = depth_mm[valid] / 1000.0
    mask = (z >= z_min_m) & (z <= z_max_m)
    z = z[mask]
    ys, xs = np.nonzero(valid)
    xs, ys = xs[mask], ys[mask]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    xyz = np.empty((len(z), 3), dtype=np.float32)
    xyz[:, 0] = (xs - cx) * z / fx
    xyz[:, 1] = (ys - cy) * z / fy
    xyz[:, 2] = z
    rgb = color_bgr[ys, xs][:, ::-1].astype(np.uint8)
    return Cloud(xyz, rgb)


def pack_cloud(cloud: Cloud) -> bytes:
    n = cloud.n
    if n == 0:
        raise ValueError("Пустое облако")
    if np.abs(cloud.xyz).max() > 32.767:
        raise ValueError("Координаты вне диапазона int16-мм (±32,7 м)")
    xyz_mm = np.rint(cloud.xyz * 1000.0).astype(np.int16)
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<HI", VERSION, n)
    # [x,y,z,r,g,b] по 9 байт: кладём xyz-плоскости и rgb-плоскости блоками.
    buf += xyz_mm.tobytes()
    buf += cloud.rgb.tobytes()
    return bytes(buf)


def unpack_cloud(data: bytes) -> Cloud:
    if len(data) < 10 or data[:4] != MAGIC:
        raise ValueError("Неверный magic .cloud")
    version, n = struct.unpack("<HI", data[4:10])
    if version != VERSION:
        raise ValueError(f"Неподдерживаемая версия .cloud: {version}")
    expected = 10 + n * 9
    if len(data) < expected:
        raise ValueError(f"Обрезанный .cloud: {len(data)} < {expected}")
    xyz = np.frombuffer(data[10:10 + n * 6], dtype=np.int16).reshape(n, 3).astype(np.float32) / 1000.0
    rgb = np.frombuffer(data[10 + n * 6:expected], dtype=np.uint8).reshape(n, 3)
    return Cloud(xyz, rgb)


def save_cloud(cloud: Cloud, path: str) -> None:
    with open(path, "wb") as f:
        f.write(pack_cloud(cloud))


def load_cloud(path: str) -> Cloud:
    with open(path, "rb") as f:
        return unpack_cloud(f.read())


def write_jpeg(bgr: np.ndarray, path: str, quality: int = 90) -> None:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Не удалось закодировать JPEG")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
