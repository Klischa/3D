# -*- coding: utf-8 -*-
"""
meshkit — мини-библиотека для процедурной генерации меша (только stdlib).

Возможности:
  * Mesh   — вершины/грани(полигоны) с тегами частей, UV и материалами;
  * loft   — сшивание набора сечений ("ring") в гладкую оболочку;
  * нормали — сглаженные вершинные + фасетные;
  * auto-orient — по знаку объёма разворачивает грани части наружу;
  * покраска — материал грани вычисляется правилом по тегу/центроиду/нормали;
  * экспорт — OBJ+MTL, glTF 2.0 (.glb, PBR, без внешних зависимостей), STL.

Используется генератором spitfire-mk-ix/tools/generate_spitfire.py.
"""

import json
import math
import os
import struct
from collections import defaultdict


# --------------------------------------------------------------------------
# векторная математика над кортежами
# --------------------------------------------------------------------------
def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_len(a):
    return math.sqrt(v_dot(a, a))


def v_norm(a):
    l = v_len(a)
    if l < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / l, a[1] / l, a[2] / l)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0) if e1 != e0 else 0.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def catmull_rom(pts, t):
    """pts: [(t_i, v_i), ...] с возрастающим t. Интерполяция значения в t."""
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    i = 0
    while i < len(pts) - 2 and t > pts[i + 1][0]:
        i += 1
    t0, v0 = pts[i]
    t1, v1 = pts[i + 1]
    dt = t1 - t0
    if dt <= 0:
        return v1
    u = (t - t0) / dt
    pm1 = pts[i - 1][1] if i > 0 else v0 - (v1 - v0)
    pp2 = pts[i + 2][1] if i + 2 < len(pts) else v1 + (v1 - v0)
    m0 = 0.5 * (v1 - pm1)
    m1 = 0.5 * (pp2 - v0)
    u2 = u * u
    u3 = u2 * u
    return ((2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * m0 +
            (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * m1)


class Curve1D(object):
    """Непрерывная кривая по таблице опорных точек (Catmull-Rom)."""

    def __init__(self, pts):
        self.pts = sorted(pts, key=lambda p: p[0])

    def __call__(self, t):
        return catmull_rom(self.pts, t)


# --------------------------------------------------------------------------
# материалы
# --------------------------------------------------------------------------
class Material(object):
    """Цвет задаётся в sRGB 0..1 (как в дизайне), в glTF пишется линейный."""

    def __init__(self, name, color, metallic=0.0, roughness=0.7,
                 alpha=1.0, double_sided=False, specular=0.25):
        self.name = name
        self.color = color
        self.metallic = metallic
        self.roughness = roughness
        self.alpha = alpha
        self.double_sided = double_sided
        self.specular = specular


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# --------------------------------------------------------------------------
# меш
# --------------------------------------------------------------------------
class Face(object):
    __slots__ = ("v", "tag", "uv", "cen", "nrm", "mat", "part")

    def __init__(self, v, tag, uv, part=0):
        self.v = v
        self.tag = tag
        self.uv = uv
        self.part = part
        self.cen = None
        self.nrm = None
        self.mat = None


class Mesh(object):
    def __init__(self):
        self.verts = []
        self.faces = []
        self.tag_faces = defaultdict(list)
        self.part_faces = defaultdict(list)
        self.parts = 0
        self._part = 0
        self.new_part()

    def new_part(self):
        """Начинает новую «часть» — замкнутую оболочку (для контроля ориентации)."""
        self.parts += 1
        self._part = self.parts
        return self.parts

    # -- примитивы ---------------------------------------------------------
    def add_vert(self, p):
        self.verts.append((float(p[0]), float(p[1]), float(p[2])))
        return len(self.verts) - 1

    def add_ring(self, pts):
        return [self.add_vert(p) for p in pts]

    def add_face(self, idx, tag, uv=(0.0, 0.0)):
        idx = [i for i in idx]
        # убираем вырожденные (склеенные) вершины
        uniq = []
        for i in idx:
            if not uniq or uniq[-1] != i:
                uniq.append(i)
        if len(uniq) > 1 and uniq[0] == uniq[-1]:
            uniq.pop()
        if len(uniq) < 3:
            return None
        f = Face(uniq, tag, uv, self._part)
        self.faces.append(f)
        self.tag_faces[tag].append(len(self.faces) - 1)
        self.part_faces[self._part].append(len(self.faces) - 1)
        return f

    # -- лофтинг -----------------------------------------------------------
    def loft(self, rings, tag, closed=True, cap_start=False, cap_end=False,
             uv_from=0.0, uv_to=1.0, cap_tag=None, part=None):
        """rings: список сечений (списки точек) одинаковой длины.

        tag — строка (все грани) либо список тегов по индексу j (по «поясам»).
        Правило ориентации торцов: начало — обратный порядок, конец — прямой;
        при этом весь лофт получается согласованным (см. fix_orientation).
        """
        if part is None:
            self.new_part()
        else:
            self._part = part
        per_segment = isinstance(tag, (list, tuple))
        if per_segment:
            if cap_tag is None:
                cap_tag = tag[0]
        else:
            cap_tag = tag if cap_tag is None else cap_tag
        n = len(rings)
        m = len(rings[0])
        idx = [self.add_ring(r) for r in rings]
        for i in range(n - 1):
            u0 = lerp(uv_from, uv_to, i / max(1, n - 1))
            u1 = lerp(uv_from, uv_to, (i + 1) / max(1, n - 1))
            jm = m if closed else m - 1
            for j in range(jm):
                j2 = (j + 1) % m
                t = tag[j] if per_segment else tag
                self.add_face(
                    [idx[i][j], idx[i][j2], idx[i + 1][j2], idx[i + 1][j]],
                    t, ((u0 + u1) * 0.5, (j + 0.5) / m))
        if cap_start:
            self.add_face(list(reversed(idx[0])), cap_tag, (uv_from, 0.0))
        if cap_end:
            self.add_face(list(idx[-1]), cap_tag, (uv_to, 0.0))
        return idx

    def torus_loft(self, rings, tag, closed_j=True, part=None):
        """Лофт, замкнутый и по i (перебор сечений): труба/кольцо-тор.

        rings: список сечений, каждое — замкнутый контур поперечного разреза.
        """
        if part is None:
            self.new_part()
        else:
            self._part = part
        per_segment = isinstance(tag, (list, tuple))
        n = len(rings)
        m = len(rings[0])
        idx = [self.add_ring(r) for r in rings]
        for i in range(n):
            i2 = (i + 1) % n
            jm = m if closed_j else m - 1
            for j in range(jm):
                j2 = (j + 1) % m
                t = tag[j] if per_segment else tag
                self.add_face([idx[i][j], idx[i][j2], idx[i2][j2], idx[i2][j]], t)
        return idx

    # -- преобразования ----------------------------------------------------
    def mirror_y(self, v0, f0):
        """Добавляет зеркальную копию всего, что построено после снимка (v0, f0).

        Оригинал остаётся на месте, копия получает обратный обход вершин
        (чтобы нормали тоже остались наружу).
        """
        self.new_part()
        n = len(self.verts)
        shift = n - v0
        for i in range(v0, n):
            x, y, z = self.verts[i]
            self.verts.append((x, -y, z))
        for fi in range(f0, len(self.faces)):
            f = self.faces[fi]
            self.add_face(list(reversed([i + shift for i in f.v])), f.tag, f.uv)

    def transform(self, v0, fn):
        for i in range(v0, len(self.verts)):
            self.verts[i] = fn(self.verts[i])

    def mark(self):
        return (len(self.verts), len(self.faces))

    def fan(self, ring_idx, apex, tag, uv=(0.0, 0.0), reverse=False):
        a = self.add_vert(apex)
        m = len(ring_idx)
        for j in range(m):
            j2 = (j + 1) % m
            tri = [ring_idx[j], ring_idx[j2], a]
            if reverse:
                tri = list(reversed(tri))
            self.add_face(tri, tag, uv)
        return a

    def flip_tag(self, tag):
        for fi in self.tag_faces[tag]:
            self.faces[fi].v = list(reversed(self.faces[fi].v))

    # -- финализация -------------------------------------------------------
    def finalize(self):
        """Считает центроиды и нормали граней, вершинные нормали."""
        self.vnrm = [(0.0, 0.0, 0.0)] * len(self.verts)
        for f in self.faces:
            pts = [self.verts[i] for i in f.v]
            c = (sum(p[0] for p in pts) / len(pts),
                 sum(p[1] for p in pts) / len(pts),
                 sum(p[2] for p in pts) / len(pts))
            f.cen = c
            n = (0.0, 0.0, 0.0)
            for k in range(1, len(pts) - 1):
                n = v_add(n, v_cross(v_sub(pts[k], pts[0]),
                                     v_sub(pts[k + 1], pts[0])))
            f.nrm = v_norm(n)
            # площадное накопление вершинных нормалей
            for i in f.v:
                self.vnrm[i] = v_add(self.vnrm[i], n)
        self.vnrm = [v_norm(n) for n in self.vnrm]

    def part_volume(self, tag):
        vol = 0.0
        for fi in self.tag_faces[tag]:
            f = self.faces[fi]
            pts = [self.verts[i] for i in f.v]
            for k in range(1, len(pts) - 1):
                a, b, c = pts[0], pts[k], pts[k + 1]
                vol += v_dot(a, v_cross(b, c)) / 6.0
        return vol

    def _volume_of(self, face_ids):
        vol = 0.0
        for fi in face_ids:
            f = self.faces[fi]
            pts = [self.verts[i] for i in f.v]
            for k in range(1, len(pts) - 1):
                a, b, c = pts[0], pts[k], pts[k + 1]
                vol += v_dot(a, v_cross(b, c)) / 6.0
        return vol

    def part_volume(self, tag):
        return self._volume_of(self.tag_faces[tag])

    def flip_part(self, pid):
        for fi in self.part_faces[pid]:
            self.faces[fi].v = list(reversed(self.faces[fi].v))

    def fix_orientation(self):
        """Разворачивает оболочки, у которых объём отрицательный (нормали внутрь)."""
        flipped = 0
        for pid in sorted(self.part_faces):
            if self._volume_of(self.part_faces[pid]) < 0:
                self.flip_part(pid)
                flipped += 1
        return flipped

    def open_shells(self):
        """Диагностика: оболочки с незакрытыми краями (каждое ребро должно быть 2 раза)."""
        report = []
        for pid in sorted(self.part_faces):
            edges = defaultdict(int)
            for fi in self.part_faces[pid]:
                vv = self.faces[fi].v
                for k in range(len(vv)):
                    a, b = vv[k], vv[(k + 1) % len(vv)]
                    edges[(a, b)] += 1
            bad = sum(1 for e, n in edges.items()
                      if n != 1 or edges.get((e[1], e[0]), 0) != 1)
            if bad:
                report.append((pid, bad, len(self.part_faces[pid]),
                               self.faces[self.part_faces[pid][0]].tag))
        return report

    def paint(self, painter):
        for f in self.faces:
            f.mat = painter(f)

    def stats(self):
        tri = sum(len(f.v) - 2 for f in self.faces)
        return len(self.verts), len(self.faces), tri

    def bounds(self):
        xs = [p[0] for p in self.verts]
        ys = [p[1] for p in self.verts]
        zs = [p[2] for p in self.verts]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# --------------------------------------------------------------------------
# экспорт
# --------------------------------------------------------------------------
def _triangles(mesh):
    for f in mesh.faces:
        pts = f.v
        for k in range(1, len(pts) - 1):
            yield (pts[0], pts[k], pts[k + 1]), f


def export_obj(mesh, path, materials, comment=""):
    base = os.path.splitext(path)[0]
    mtl_name = os.path.basename(base) + ".mtl"
    mat_index = {}
    for name, m in materials.items():
        mat_index[name] = "m%s" % name

    with open(base + ".mtl", "w", encoding="utf-8") as fh:
        for name, m in materials.items():
            r, g, b = m.color
            fh.write("newmtl %s\n" % mat_index[name])
            fh.write("Kd %.5f %.5f %.5f\n" % (r, g, b))
            fh.write("Ka %.5f %.5f %.5f\n" % (r * 0.1, g * 0.1, b * 0.1))
            spec = 0.1 + 0.9 * (1.0 - m.roughness)
            fh.write("Ks %.5f %.5f %.5f\n" % (spec, spec, spec))
            fh.write("Ns %.2f\n" % (2.0 + 400.0 * (1.0 - m.roughness) ** 2))
            fh.write("d %.4f\n" % m.alpha)
            fh.write("illum 2\n\n")

    with open(path, "w", encoding="utf-8") as fh:
        if comment:
            fh.write("# %s\n" % comment)
        fh.write("# vertices: %d, faces: %d\n" % (len(mesh.verts), len(mesh.faces)))
        fh.write("mtllib %s\n" % mtl_name)
        fh.write("o Spitfire\n")
        for p in mesh.verts:
            fh.write("v %.5f %.5f %.5f\n" % p)
        for n in mesh.vnrm:
            fh.write("vn %.5f %.5f %.5f\n" % n)
        cur = None
        for f in mesh.faces:
            if f.mat != cur:
                fh.write("usemtl %s\n" % mat_index[f.mat])
                cur = f.mat
            vv = " ".join("%d//%d" % (i + 1, i + 1) for i in f.v)
            fh.write("f %s\n" % vv)
    return path


def export_glb(mesh, path, materials, name="Supermarine Spitfire"):
    """Минимальный, но корректный glTF 2.0 (binary) без внешних зависимостей."""
    by_mat = defaultdict(list)
    for f in mesh.faces:
        by_mat[f.mat].append(f)
    order = [m for m in materials if m in by_mat]

    # буферы
    pos_blob = b""
    nrm_blob = b""
    idx_blob = b""
    views = []
    accessors = []
    prims = []
    offset = 0

    def add_view(blob, target=None):
        nonlocal offset
        # выравнивание по 4 байта
        pad = (-offset) % 4
        return pad

    chunks = bytearray()

    def push(data, target=None):
        nonlocal offset
        pad = (-offset) % 4
        if pad:
            chunks.extend(b"\x00" * pad)
            offset += pad
        off = offset
        chunks.extend(data)
        offset += len(data)
        views.append({"buffer": 0, "byteOffset": off,
                      "byteLength": len(data), **({"target": target} if target else {})})
        return len(views) - 1

    # все вершины пишем один раз
    pos_data = b"".join(struct.pack("<3f", *p) for p in mesh.verts)
    nrm_data = b"".join(struct.pack("<3f", *n) for n in mesh.vnrm)
    pos_view = push(pos_data, 34962)
    nrm_view = push(nrm_data, 34962)
    xs = [p[0] for p in mesh.verts]
    ys = [p[1] for p in mesh.verts]
    zs = [p[2] for p in mesh.verts]
    accessors.append({"bufferView": pos_view, "componentType": 5126,
                      "count": len(mesh.verts), "type": "VEC3",
                      "min": [min(xs), min(ys), min(zs)],
                      "max": [max(xs), max(ys), max(zs)]})
    pos_acc = len(accessors) - 1
    accessors.append({"bufferView": nrm_view, "componentType": 5126,
                      "count": len(mesh.verts), "type": "VEC3"})
    nrm_acc = len(accessors) - 1

    for mat_name in order:
        faces = by_mat[mat_name]
        idx = []
        for f in faces:
            for k in range(1, len(f.v) - 1):
                idx.extend([f.v[0], f.v[k], f.v[k + 1]])
        idx_data = struct.pack("<%dI" % len(idx), *idx)
        iv = push(idx_data, 34963)
        accessors.append({"bufferView": iv, "componentType": 5125,
                          "count": len(idx), "type": "SCALAR"})
        prims.append({"attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                      "indices": len(accessors) - 1,
                      "material": order.index(mat_name),
                      "mode": 4})

    gl_materials = []
    for mat_name in order:
        m = materials[mat_name]
        col = [srgb_to_linear(c) for c in m.color]
        g = {"name": mat_name,
             "doubleSided": bool(m.double_sided),
             "pbrMetallicRoughness": {
                 "baseColorFactor": col + [m.alpha],
                 "metallicFactor": m.metallic,
                 "roughnessFactor": m.roughness}}
        if m.alpha < 1.0:
            g["alphaMode"] = "BLEND"
        gl_materials.append(g)

    gltf = {
        "asset": {"version": "2.0", "generator": "meshkit (github.com/Klischa/3D/spitfire-mk-ix)"},
        "scene": 0,
        "scenes": [{"name": name, "nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": prims}],
        "materials": gl_materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    bin_chunk = bytes(chunks)
    bin_chunk += b"\x00" * ((-len(bin_chunk)) % 4)

    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        fh.write(bin_chunk)
    return path


def export_stl(mesh, path, name="Spitfire"):
    with open(path, "wb") as fh:
        header = ("%s — procedural model (github.com/Klischa/3D/spitfire-mk-ix)" % name).encode("ascii", "replace")
        fh.write(header.ljust(80, b" ")[:80])
        tris = list(_triangles(mesh))
        fh.write(struct.pack("<I", len(tris)))
        for (a, b, c), f in tris:
            pa, pb, pc = mesh.verts[a], mesh.verts[b], mesh.verts[c]
            n = v_norm(v_cross(v_sub(pb, pa), v_sub(pc, pa)))
            fh.write(struct.pack("<3f", *n))
            for p in (pa, pb, pc):
                fh.write(struct.pack("<3f", *p))
            fh.write(struct.pack("<H", 0))
    return path


def export_materials_json(materials, path):
    data = {}
    for name, m in materials.items():
        data[name] = {"color": list(m.color), "metallic": m.metallic,
                      "roughness": m.roughness, "alpha": m.alpha}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    return path
