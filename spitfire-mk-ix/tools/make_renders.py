# -*- coding: utf-8 -*-
"""
Финальные превью-рендеры модели (оффлайн, без браузера) + контактный лист.

Запуск из папки spitfire-mk-ix:
    python3 tools/make_renders.py

Пишет в renders/:
    01_hero.png       — вид 3/4 (главная картинка)
    02_side.png       — профиль (левый борт)
    03_top.png        — план
    04_front.png      — вид спереди
    05_bottom.png     — вид снизу
    06_engine.png     — нос, винт, выхлоп
    07_cockpit.png    — кабина и центр крыла
    08_tail.png       — хвостовое оперение и шасси
    model_sheet.png   — контактный лист 2×2
"""

import os
import sys
import zlib
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from render_preview import load_obj, load_mtl, render  # noqa: E402

OUT = os.path.join(ROOT, "renders")


# ---------------------------------------------------------------- PNG-утилиты
def read_png(path):
    data = open(path, "rb").read()
    pos, w, h, idat = 8, None, None, b""
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + ln]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", chunk[:8])
        elif tag == b"IDAT":
            idat += chunk
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * 3
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        f = raw[i]
        i += 1
        line = bytearray(raw[i:i + stride])
        i += stride
        if f == 1:
            for x in range(3, stride):
                line[x] = (line[x] + line[x - 3]) & 255
        elif f == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif f == 3:
            for x in range(stride):
                a = line[x - 3] if x >= 3 else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = line[x - 3] if x >= 3 else 0
                b = prev[x]
                c = prev[x - 3] if x >= 3 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pr) & 255
        rows.append(line)
        prev = line
    return w, h, rows


def write_png(path, w, h, rows):
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    comp = zlib.compress(raw, 6)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", comp))
        fh.write(chunk(b"IEND", b""))


def stitch(paths, out, cols=2, gutter=10, bg=(18, 22, 28)):
    imgs = [read_png(p) for p in paths]
    cw = max(i[0] for i in imgs)
    ch = max(i[1] for i in imgs)
    rows_n = (len(imgs) + cols - 1) // cols
    W = cw * cols + gutter * (cols + 1)
    H = ch * rows_n + gutter * (rows_n + 1)
    rows = [bytearray(bytes(bg) * W) for _ in range(H)]
    for k, (w, h, src) in enumerate(imgs):
        cx = k % cols
        cy = k // cols
        ox = gutter + cx * (cw + gutter) + (cw - w) // 2
        oy = gutter + cy * (ch + gutter) + (ch - h) // 2
        for y in range(h):
            rows[oy + y][ox * 3:(ox + w) * 3] = src[y]
    write_png(out, W, H, rows)
    return out


# --------------------------------------------------------------------------
def main():
    obj = os.path.join(ROOT, "models", "spitfire.obj")
    mats = load_mtl(os.path.splitext(obj)[0] + ".mtl")
    verts, norms, faces = load_obj(obj)
    print("модель: %d вершин, %d треугольников" % (len(verts), len(faces)))
    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    jobs = [
        ("01_hero.png", "persp", (1600, 980), 32.0, None, None, {}),
        ("02_side.png", "side", (1500, 820), 32.0, None, None, {}),
        ("03_top.png", "top", (1400, 860), 32.0, None, None, {}),
        ("04_front.png", "front", (1400, 820), 32.0, None, None, {}),
        ("05_bottom.png", "bottom", (1400, 860), 32.0, None, None, {}),
        ("06_engine.png", "persp", (1200, 800), 34.0, 3.4, (3.3, 0.6, -0.1), {}),
        ("07_cockpit.png", "persp2", (1200, 800), 34.0, 3.2, (-0.2, 0.0, 0.1), {}),
        ("08_tail.png", "persp", (1200, 800), 32.0, 3.4, (-2.4, 0.8, 0.15), {}),
    ]
    for name, view, size, fov, dist, center, kw in jobs:
        out = os.path.join(OUT, name)
        render(verts, norms, faces, mats, out, size, view, fov, dist, **kw)
        print("  -> %s (%.0f КБ)" % (name, os.path.getsize(out) / 1024.0))

    sheet = stitch([os.path.join(OUT, n) for n in
                    ("01_hero.png", "02_side.png", "03_top.png", "05_bottom.png")],
                   os.path.join(OUT, "model_sheet.png"))
    print("  -> model_sheet.png (%.0f КБ)" % (os.path.getsize(sheet) / 1024.0))


if __name__ == "__main__":
    main()
