# -*- coding: utf-8 -*-
"""
Оффлайн-рендер превью модели (программный растеризатор, только stdlib).

Читает models/spitfire.obj + spitfire.mtl, рисует картинку с z-буфером,
пер-пиксельным освещением (ключевой свет + небесный ambient + блик)
и записывает PNG.

Запуск:
    python3 tools/render_preview.py --view persp  --out renders/01_persp.png
    python3 tools/render_preview.py --view side   --out renders/02_side.png
    python3 tools/render_preview.py --view top    --out renders/03_top.png
    python3 tools/render_preview.py --view front  --out renders/04_front.png
"""

import argparse
import math
import os
import struct
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------------------
# загрузка OBJ/MTL
# --------------------------------------------------------------------------
def load_mtl(path):
    mats = {}
    cur = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "newmtl":
                cur = {"kd": (0.8, 0.8, 0.8), "ks": (0.2, 0.2, 0.2), "ns": 40.0,
                       "d": 1.0}
                mats[p[1]] = cur
            elif cur is None:
                continue
            elif p[0] == "Kd":
                cur["kd"] = (float(p[1]), float(p[2]), float(p[3]))
            elif p[0] == "Ks":
                cur["ks"] = (float(p[1]), float(p[2]), float(p[3]))
            elif p[0] == "Ns":
                cur["ns"] = float(p[1])
            elif p[0] == "d":
                cur["d"] = float(p[1])
    return mats


def load_obj(path):
    verts, norms, faces = [], [], []
    mats = {}
    cur = "default"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif p[0] == "vn":
                norms.append((float(p[1]), float(p[2]), float(p[3])))
            elif p[0] == "usemtl":
                cur = p[1]
            elif p[0] == "f":
                idx = []
                for tok in p[1:]:
                    a = tok.split("/")
                    vi = int(a[0]) - 1
                    ni = int(a[2]) - 1 if len(a) > 2 and a[2] else vi
                    idx.append((vi, ni))
                for k in range(1, len(idx) - 1):
                    faces.append(([idx[0], idx[k], idx[k + 1]], cur))
    return verts, norms, faces


# --------------------------------------------------------------------------
# PNG
# --------------------------------------------------------------------------
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


def srgb(c):
    c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
    v = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return int(v * 255.0 + 0.5)


# --------------------------------------------------------------------------
# рендер
# --------------------------------------------------------------------------
def v_norm(v):
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / l, v[1] / l, v[2] / l) if l > 1e-12 else (0.0, 0.0, 1.0)


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# view -> (yaw, pitch, up_hint) в градусах
VIEWS = {
    "persp": (36.0, 17.0, (0.0, 0.0, 1.0)),
    "persp2": (-148.0, 21.0, (0.0, 0.0, 1.0)),
    "side": (91.0, 3.0, (0.0, 0.0, 1.0)),
    "side_r": (-89.0, 3.0, (0.0, 0.0, 1.0)),
    "front": (0.0, 6.0, (0.0, 0.0, 1.0)),
    "rear": (180.0, 8.0, (0.0, 0.0, 1.0)),
    "top": None,      # особый случай: строго сверху, нос вверх кадра
    "bottom": None,
}
NO_GROUND = ("top", "bottom")


def render(verts, norms, faces, mats, out, size, view, fov=32.0, dist=None,
           bg_top=(0.13, 0.16, 0.21), bg_bot=(0.04, 0.05, 0.07), ground=True,
           center=None, zc=-1.925):
    W, H = size
    spec = VIEWS[view]
    if view in ("top", "bottom"):
        yaw, pitch = 0.0, 0.0
        up_hint = (1.0, 0.0, 0.0)
        vert = 1.0 if view == "top" else -1.0
    else:
        yaw, pitch, up_hint = spec
        yaw, pitch = math.radians(yaw), math.radians(pitch)
        vert = 0.0

    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    if center is None:
        center = ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5,
                  (lo[2] + hi[2]) * 0.5)
    span = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    if dist is None:
        dist = span * 1.35

    if view in ("top", "bottom"):
        eye = (center[0] + 0.001, center[1], center[2] + vert * dist)
    else:
        eye = (center[0] + dist * math.cos(pitch) * math.cos(yaw),
               center[1] + dist * math.cos(pitch) * math.sin(yaw),
               center[2] + dist * math.sin(pitch))
    fwd = v_norm((center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]))
    if abs(v_dot(fwd, up_hint)) > 0.999:
        up_hint = (0.0, 0.0, 1.0) if abs(fwd[2]) < 0.9 else (1.0, 0.0, 0.0)
    right = v_norm(v_cross(fwd, up_hint))
    up = v_cross(right, fwd)
    f = (H * 0.5) / math.tan(math.radians(fov) * 0.5)

    # освещение
    key = v_norm((0.42, 0.52, 0.74))
    key_c = (1.10, 1.06, 0.98)
    sky_c = (0.30, 0.35, 0.42)
    gnd_c = (0.13, 0.12, 0.11)
    half = v_norm((key[0] + fwd[0], key[1] + fwd[1], key[2] + fwd[2]))
    view_c = (-fwd[0], -fwd[1], -fwd[2])

    # фон
    px = [[0, 0, 0] for _ in range(W * H)]
    for y in range(H):
        t = y / float(H - 1)
        col = [srgb(bg_top[i] + (bg_bot[i] - bg_top[i]) * t) for i in range(3)]
        row0 = y * W
        for x in range(W):
            p = px[row0 + x]
            p[0], p[1], p[2] = col

    # горизонт/земля
    if ground and view not in NO_GROUND and zc is not None:
        for y in range(H):
            for x in range(W):
                rx = (x + 0.5 - W * 0.5) / f
                ry = (H * 0.5 - (y + 0.5)) / f
                d = v_norm((fwd[0] + right[0] * rx + up[0] * ry,
                            fwd[1] + right[1] * rx + up[1] * ry,
                            fwd[2] + right[2] * rx + up[2] * ry))
                if d[2] < -1e-6:
                    t = (zc - eye[2]) / d[2]
                    if t > 0:
                        hx = eye[0] + d[0] * t
                        hy = eye[1] + d[1] * t
                        fade = max(0.0, 1.0 - math.sqrt(hx * hx + hy * hy) / 26.0)
                        c = [srgb((0.10 + 0.06 * fade) * s) for s in (1.0, 1.04, 1.10)]
                        p = px[y * W + x]
                        p[0], p[1], p[2] = c

    zbuf = [1e30] * (W * H)
    t0 = time.time()

    # проекция вершин
    proj = []
    for v in verts:
        rx, ry, rz = v[0] - eye[0], v[1] - eye[1], v[2] - eye[2]
        cz = v_dot((rx, ry, rz), fwd)
        cx = v_dot((rx, ry, rz), right)
        cy = v_dot((rx, ry, rz), up)
        if cz < 0.05:
            proj.append(None)
            continue
        # w = cz — для перспективно-корректной интерполяции глубины и атрибутов
        proj.append((W * 0.5 + cx / cz * f, H * 0.5 - cy / cz * f, cz, cz))

    # порядок: сначала непрозрачные, потом стекло
    opaque, glass = [], []
    for tri, mname in faces:
        (opaque if mats.get(mname, {}).get("d", 1.0) >= 0.999 else glass).append((tri, mname))

    def draw(passes, write_z, blend):
        for tri, mname in passes:
            p = [proj[i[0]] for i in tri]
            n = [norms[i[1]] for i in tri]
            if p[0] is None or p[1] is None or p[2] is None:
                continue
            (x0, y0, w0, z0), (x1, y1, w1, z1), (x2, y2, w2, z2) = p
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-9:
                continue
            inv_area = 1.0 / area
            minx = max(0, int(min(x0, x1, x2)))
            maxx = min(W - 1, int(max(x0, x1, x2)) + 1)
            miny = max(0, int(min(y0, y1, y2)))
            maxy = min(H - 1, int(max(y0, y1, y2)) + 1)
            if minx > maxx or miny > maxy:
                continue
            m = mats.get(mname)
            if m is None:
                m = {"kd": (0.6, 0.6, 0.6), "ks": (0.2, 0.2, 0.2), "ns": 40.0, "d": 1.0}
            kd = [c ** 2.2 for c in m["kd"]]
            ks = (m["ks"][0] + m["ks"][1] + m["ks"][2]) / 3.0
            ns = m["ns"]
            alpha = m["d"]
            e0x, e0y = y1 - y2, x2 - x1
            e1x, e1y = y2 - y0, x0 - x2
            e2x, e2y = y0 - y1, x1 - x0
            for y in range(miny, maxy + 1):
                py = y + 0.5
                row = y * W
                for x in range(minx, maxx + 1):
                    pxx = x + 0.5
                    b0 = (e0x * (pxx - x1) + e0y * (py - y1)) * inv_area
                    if b0 < 0.0:
                        continue
                    b1 = (e1x * (pxx - x2) + e1y * (py - y2)) * inv_area
                    if b1 < 0.0:
                        continue
                    b2 = 1.0 - b0 - b1
                    if b2 < 0.0:
                        continue
                    i = row + x
                    wsum = b0 / w0 + b1 / w1 + b2 / w2
                    zz = 1.0 / wsum
                    if zz >= zbuf[i]:
                        continue
                    pb0 = (b0 / w0) / wsum
                    pb1 = (b1 / w1) / wsum
                    pb2 = (b2 / w2) / wsum
                    nx = n[0][0] * pb0 + n[1][0] * pb1 + n[2][0] * pb2
                    ny = n[0][1] * pb0 + n[1][1] * pb1 + n[2][1] * pb2
                    nz = n[0][2] * pb0 + n[1][2] * pb1 + n[2][2] * pb2
                    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
                    if ln < 1e-9:
                        continue
                    nx /= ln
                    ny /= ln
                    nz /= ln
                    ndv = nx * view_c[0] + ny * view_c[1] + nz * view_c[2]
                    if ndv < 0.0:
                        nx, ny, nz = -nx, -ny, -nz
                        ndv = -ndv
                    ndl = nx * key[0] + ny * key[1] + nz * key[2]
                    if ndl < 0.0:
                        ndl = 0.0
                    skyf = 0.5 + 0.5 * nz
                    r = kd[0] * (ndl * key_c[0] + sky_c[0] * skyf + gnd_c[0] * (1.0 - skyf) + 0.06 * ndv)
                    g = kd[1] * (ndl * key_c[1] + sky_c[1] * skyf + gnd_c[1] * (1.0 - skyf) + 0.06 * ndv)
                    b = kd[2] * (ndl * key_c[2] + sky_c[2] * skyf + gnd_c[2] * (1.0 - skyf) + 0.06 * ndv)
                    ndh = nx * half[0] + ny * half[1] + nz * half[2]
                    if ndh > 0.0 and ks > 0.01:
                        sp = 0.45 * ks * (ndh ** ns)
                        r += sp
                        g += sp
                        b += sp
                    if write_z:
                        zbuf[i] = zz
                    p = px[i]
                    if blend < 1.0:
                        p[0] = int(p[0] * (1.0 - blend) + srgb(r) * blend)
                        p[1] = int(p[1] * (1.0 - blend) + srgb(g) * blend)
                        p[2] = int(p[2] * (1.0 - blend) + srgb(b) * blend)
                    else:
                        p[0], p[1], p[2] = srgb(r), srgb(g), srgb(b)

    draw(opaque, True, 1.0)
    if glass:
        draw(glass, False, 0.55)
    print("  рендер %s: %.1f с" % (view, time.time() - t0))

    rows = []
    for y in range(H):
        row = bytearray()
        base = y * W
        for x in range(W):
            p = px[base + x]
            row += bytes((p[0], p[1], p[2]))
        rows.append(row)
    write_png(out, W, H, rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="persp", choices=sorted(VIEWS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", default="1000x620")
    ap.add_argument("--fov", type=float, default=32.0)
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--center", default=None, help="x,y,z — точка взгляда")
    ap.add_argument("--noground", action="store_true")
    ap.add_argument("--obj", default=os.path.join(ROOT, "models", "spitfire.obj"))
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.lower().split("x"))
    out = args.out or os.path.join(ROOT, "renders", "preview_%s.png" % args.view)
    if not os.path.isdir(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    mtl = os.path.splitext(args.obj)[0] + ".mtl"
    mats = load_mtl(mtl)
    verts, norms, faces = load_obj(args.obj)
    print("  загружено: %d вершин, %d треугольников, %d материалов"
          % (len(verts), len(faces), len(mats)))
    center = tuple(float(v) for v in args.center.split(",")) if args.center else None
    render(verts, norms, faces, mats, out, (w, h), args.view, args.fov, args.dist,
           ground=not args.noground, center=center)
    print("  -> %s" % out)


if __name__ == "__main__":
    main()
