# -*- coding: utf-8 -*-
"""
Процедурный генератор 3D-модели Supermarine Spitfire Mk IX (RAF, 1942-44).
Только стандартная библиотека Python — никаких внешних зависимостей.

Запуск из папки spitfire-mk-ix:
    python3 tools/generate_spitfire.py

Результат в models/:
    spitfire.obj + spitfire.mtl  — для Blender / 3ds Max / Maya / Sketchfab
    spitfire.glb                 — glTF 2.0 binary (PBR) для Three.js / Unity / Godot
    spitfire.stl                 — для 3D-печати
    materials.json               — палитра материалов

Система координат (правая): X — вперёд (в сторону носа), Y — влево (по размаху),
Z — вверх. Единицы — метры. Начало координат — на оси винта, в плоскости симметрии.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from meshkit import (Mesh, Material, Curve1D, v_add, v_sub, v_mul, v_cross,
                     v_norm, v_len, lerp, clamp, export_obj, export_glb,
                     export_stl, export_materials_json)

TAU = math.pi * 2.0
D2R = math.pi / 180.0
OUT = os.path.join(os.path.dirname(HERE), "models")


# ==========================================================================
# 1. МАТЕРИАЛЫ (sRGB, дневной истребитель RAF 1943-44:
#    Dark Green / Ocean Grey сверху, Medium Sea Grey снизу)
# ==========================================================================
MATERIALS = {}


def mat(name, color, metallic=0.0, roughness=0.7, alpha=1.0, double_sided=False):
    MATERIALS[name] = Material(name, color, metallic, roughness, alpha, double_sided)
    return name


M_GREEN = mat("raf_dark_green", (0.243, 0.290, 0.180), roughness=0.62)
M_OCEAN = mat("raf_ocean_grey", (0.419, 0.439, 0.408), roughness=0.64)
M_SEA = mat("raf_sea_grey", (0.615, 0.647, 0.667), roughness=0.60)
M_SPIN = mat("sky_spinner", (0.730, 0.780, 0.685), roughness=0.45, metallic=0.10)
M_GLASS = mat("canopy_glass", (0.784, 0.855, 0.886), roughness=0.06, alpha=0.32,
              double_sided=True)
M_FRAME = mat("canopy_frame", (0.196, 0.208, 0.204), roughness=0.55)
M_METAL = mat("metal_strut", (0.596, 0.612, 0.627), metallic=0.85, roughness=0.34)
M_TYRE = mat("tyre_rubber", (0.075, 0.075, 0.086), roughness=0.95)
M_HUB = mat("wheel_hub", (0.549, 0.565, 0.580), metallic=0.70, roughness=0.40)
M_BLADE = mat("prop_blade", (0.149, 0.133, 0.118), roughness=0.48)
M_TIP = mat("prop_tip_yellow", (0.859, 0.702, 0.161), roughness=0.50)
M_EXH = mat("exhaust_steel", (0.361, 0.318, 0.290), metallic=0.55, roughness=0.62)
M_SOOT = mat("exhaust_soot", (0.129, 0.122, 0.118), roughness=0.85)
M_GUN = mat("gunmetal", (0.184, 0.192, 0.204), metallic=0.90, roughness=0.38)
M_BLACK = mat("matte_black", (0.086, 0.090, 0.094), roughness=0.85)
M_RED = mat("roundel_red", (0.706, 0.129, 0.149), roughness=0.62)
M_WHITE = mat("roundel_white", (0.918, 0.914, 0.878), roughness=0.62)
M_BLUE = mat("roundel_blue", (0.129, 0.259, 0.475), roughness=0.62)
M_YELLOW = mat("roundel_yellow", (0.878, 0.718, 0.176), roughness=0.62)
M_MATRIX = mat("radiator_matrix", (0.055, 0.058, 0.061), roughness=0.90)


# ==========================================================================
# 2. РАЗМЕРЫ (Supermarine Spitfire Mk IX)
# ==========================================================================
SPEC = {
    "length": 9.47,       # м, длина (носовой обтекатель -> руль направления)
    "span": 11.23,        # м, размах
    "height": 3.86,       # м, высота в стояночном положении (по справочнику)
    "wing_area": 22.48,   # м², площадь крыла
    "c_root": 2.56,       # м, хорда в корне (центр эллипса крыла)
    "dihedral": 6.0,      # град, поперечное V
    "incidence": 2.5,     # град, угол установки в корне
    "prop_dia": 3.28,     # м, диаметр четырёхлопастного винта Rotol (10'9\")
    "prop_blades": 4,
}

# Mk IX длиннее Mk I за счёт моторамы под двухступенчатый Merlin 61:
# носовая часть (x >= 2.40) сдвинута вперёд на NOSE_DX.
NOSE_DX = 0.30
X_NOSE = 5.30
X_TAIL = -4.14
X_LE = 1.83
Y_TIP = 5.615
Z_WING_ROOT = -0.50
GEAR_PIVOT = (0.58, 0.70)
GEAR_DOWN = (-0.05, 0.86, -1.55)
GEAR_UP = (0.42, 0.70, -0.60)
X_PROP = 5.10
SUPERE = 2.4   # показатель суперэллипса сечения фюзеляжа (плоская «палуба»)

# полуширина (Y) и полувысота (Z) фюзеляжа + смещение оси по длине
FUS_RY = Curve1D([
    (-4.05, 0.042), (-3.80, 0.085), (-3.50, 0.140), (-3.10, 0.195),
    (-2.60, 0.245), (-2.10, 0.290), (-1.60, 0.335), (-1.10, 0.380),
    (-0.60, 0.420), (-0.10, 0.462), (0.40, 0.492), (0.90, 0.515),
    (1.40, 0.530), (1.90, 0.535), (2.70, 0.500), (3.20, 0.480),
    (3.70, 0.450), (4.20, 0.410), (4.60, 0.375), (4.85, 0.335),
    (4.92, 0.330),
])
FUS_RZ = Curve1D([
    (-4.05, 0.050), (-3.80, 0.100), (-3.50, 0.175), (-3.10, 0.255),
    (-2.60, 0.325), (-2.10, 0.385), (-1.60, 0.435), (-1.10, 0.480),
    (-0.60, 0.520), (-0.10, 0.556), (0.40, 0.582), (0.90, 0.596),
    (1.40, 0.585), (1.90, 0.570), (2.70, 0.530), (3.20, 0.480),
    (3.70, 0.445), (4.20, 0.430), (4.60, 0.372), (4.85, 0.335),
    (4.92, 0.330),
])
FUS_ZC = Curve1D([
    (-4.05, 0.015), (-3.50, 0.042), (-2.60, 0.040), (-1.60, 0.030),
    (-0.60, 0.060), (0.40, 0.090), (0.90, 0.090), (1.40, 0.080),
    (2.70, 0.020), (3.20, -0.010), (3.70, -0.020), (4.20, 0.000),
    (4.60, -0.012), (4.85, -0.003), (4.92, 0.000),
])
FUS_STATIONS = [p[0] for p in FUS_RY.pts]


def fus_ry(x):
    return max(FUS_RY(x), 0.02)


def fus_rz(x):
    return max(FUS_RZ(x), 0.02)


def fus_zc(x):
    return FUS_ZC(x)


def fus_point(x, phi):
    """phi = 0 — верх, +pi/2 — левый борт, pi — низ (суперэллипс по бортам)."""
    ry, rz = fus_ry(x), fus_rz(x)
    cy, cz = math.sin(phi), math.cos(phi)
    s = (abs(cy) ** SUPERE + abs(cz) ** SUPERE) ** (-1.0 / SUPERE)
    return (x, ry * cy * s, fus_zc(x) + rz * cz * s)


def fus_top_z(x, y):
    """Высота «палубы» (верхней обшивки) при данном y."""
    ry, rz = fus_ry(x), fus_rz(x)
    k = clamp(abs(y) / ry, 0.0, 0.999)
    return fus_zc(x) + rz * (1.0 - k ** SUPERE) ** (1.0 / SUPERE)


# --------------------------------------------------------------------------
# крыло
# --------------------------------------------------------------------------
WING_EA = 5.63       # полуось эллипса хорд (чуть больше полуразмаха: кончик узкий, но не нулевой)
WING_XC_ROOT = 0.55  # центр корневой хорды (1.83 - 2.56 / 2)
WING_XC_TIP = 0.82   # центр хорды законцовки: задняя кромка заметает сильнее передней


def _wing_ell(y):
    t = min(abs(y), WING_EA - 1e-6) / WING_EA
    return math.sqrt(max(0.0, 1.0 - t * t))


def wing_chord(y):
    return max(SPEC["c_root"] * _wing_ell(y), 0.02)


def wing_center_x(y):
    t = 1.0 - _wing_ell(y)
    return lerp(WING_XC_ROOT, WING_XC_TIP, t ** 0.9)


def wing_le_x(y):
    return wing_center_x(y) + wing_chord(y) / 2.0


def wing_te_x(y):
    return wing_center_x(y) - wing_chord(y) / 2.0


def wing_zref(y):
    a = abs(y)
    z = Z_WING_ROOT
    if a > 0.55:
        z += (a - 0.55) * math.tan(SPEC["dihedral"] * D2R)
    return z


def wing_tc(y):
    return lerp(0.135, 0.098, clamp(abs(y) / 5.6, 0.0, 1.0))


def wing_inc(y):
    return lerp(SPEC["incidence"], 1.0, clamp(abs(y) / 5.6, 0.0, 1.0))


def naca_thickness(t, tc):
    return (tc / 0.2) * (0.2969 * math.sqrt(t) - 0.1260 * t
                         - 0.3516 * t * t + 0.2843 * t ** 3 - 0.1036 * t ** 4)


def naca_camber(t, camber=0.025, p=0.4):
    if t < p:
        return camber / (p * p) * (2 * p * t - t * t)
    return camber / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * t - t * t)


TS = [0.0, 0.007, 0.025, 0.06, 0.11, 0.18, 0.27, 0.38, 0.50,
      0.62, 0.73, 0.83, 0.91, 0.965, 1.0]
NTS = len(TS)


def airfoil_section(x_le, chord, z_ref, tc, inc_deg, camber=0.025):
    """Сечение в плоскости (x, z): носок в x_le, хорда вдоль -X, угол установки."""
    xqc = x_le - 0.25 * chord
    a = inc_deg * D2R
    up, lo = [], []
    for t in TS:
        yt = chord * naca_thickness(t, tc)
        yc = chord * naca_camber(t, camber)
        x = x_le - t * chord
        dx = x - xqc
        for zz, lst in ((z_ref + yc + yt, up), (z_ref + yc - yt, lo)):
            dz = zz - z_ref
            lst.append((xqc + dx * math.cos(a) + dz * math.sin(a),
                        z_ref - dx * math.sin(a) + dz * math.cos(a)))
    return up + list(reversed(lo))


def airfoil_range(x_le, chord_full, z_ref, tc, inc_deg, camber, t0, t1, n=8):
    """Участок профиля (t0..t1 по хорде) как замкнутый контур — для рулей."""
    xqc = x_le - 0.25 * chord_full
    a = inc_deg * D2R
    up, lo = [], []
    for k in range(n + 1):
        t = lerp(t0, t1, k / float(n))
        yt = chord_full * naca_thickness(t, tc)
        yc = chord_full * naca_camber(t, camber)
        x = x_le - t * chord_full
        dx = x - xqc
        for zz, lst in ((z_ref + yc + yt, up), (z_ref + yc - yt, lo)):
            dz = zz - z_ref
            lst.append((xqc + dx * math.cos(a) + dz * math.sin(a),
                        z_ref - dx * math.sin(a) + dz * math.cos(a)))
    return up + list(reversed(lo))


def wing_surface_z(x, y, lower=False):
    chord = wing_chord(y)
    t = clamp((wing_le_x(y) - x) / chord, 0.0, 1.0)
    s = -1.0 if lower else 1.0
    return (wing_zref(y) + chord * naca_camber(t)
            + s * chord * naca_thickness(t, wing_tc(y)))


def wing_normal(x, y, lower=False, eps=0.03):
    tx = (wing_surface_z(x + eps, y, lower) - wing_surface_z(x - eps, y, lower)) / (2 * eps)
    ty = (wing_surface_z(x, y + eps, lower) - wing_surface_z(x, y - eps, lower)) / (2 * eps)
    n = v_norm((-tx, -ty, 1.0))
    return v_mul(n, -1.0) if lower else n


# ==========================================================================
# 3. ГЕОМЕТРИЧЕСКИЕ ХЕЛПЕРЫ
# ==========================================================================
def frame_from_axis(d):
    d = v_norm(d)
    ref = (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = v_norm(v_cross(ref, d))
    v = v_cross(d, u)
    return u, v


def add_tube(mesh, p0, p1, r0, r1=None, tag="metal", seg=10, caps=True,
             stations=None, nstations=4):
    """Цилиндр/конус (или тело вращения с профилем radii=[(s, r)...])."""
    d = v_sub(p1, p0)
    L = v_len(d)
    if L < 1e-7:
        return
    d = v_mul(d, 1.0 / L)
    u, v = frame_from_axis(d)
    if stations is None:
        r1 = r0 if r1 is None else r1
        stations = [(k / float(nstations), lerp(r0, r1, k / float(nstations)))
                    for k in range(nstations + 1)]
    rings = []
    for s, r in stations:
        c = v_add(p0, v_mul(d, L * s))
        rings.append([v_add(c, v_add(v_mul(u, r * math.cos(TAU * i / seg)),
                                     v_mul(v, r * math.sin(TAU * i / seg))))
                      for i in range(seg)])
    mesh.loft(rings, tag, closed=True, cap_start=caps, cap_end=caps)


def add_plate(mesh, p0, p1, wdir, half_w, tdir, half_t, tag):
    """Плоская панель (щиток ниши, створка) между p0 и p1."""
    rings = []
    for p in (p0, p1):
        corners = []
        for sw, st in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            corners.append(v_add(p, v_add(v_mul(wdir, half_w * sw),
                                          v_mul(tdir, half_t * st))))
        rings.append(corners)
    mesh.loft(rings, tag, closed=True, cap_start=True, cap_end=True)


def add_box(mesh, c, size, tag, eps=0.0):
    mesh.new_part()
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    pts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
           (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    idx = [mesh.add_vert(v_add(c, p)) for p in pts]
    for q in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        mesh.add_face([idx[i] for i in q], tag)


def add_fairing(mesh, x0, x1, y, r, z_top, tag, seg=10):
    """Полуобтекатель под крылом: плоский верх, скруглённый низ, сглаженные концы."""
    rings = []
    n = 6
    for k in range(n + 1):
        s01 = k / float(n)
        x = lerp(x0, x1, s01)
        shrink = 1.0 - 0.34 * max(0.0, (abs(s01 - 0.5) * 2 - 0.70) / 0.30) ** 2
        rr = r * shrink
        ring = [(x, y + rr, z_top), (x, y - rr, z_top)]
        for i in range(1, seg):
            a = math.pi * i / seg
            ring.append((x, y - rr * math.cos(a), z_top - rr * math.sin(a)))
        rings.append(ring)
    mesh.loft(rings, tag, closed=True, cap_start=True, cap_end=True)


def add_swept(mesh, path, r, tag, seg=8):
    for i in range(len(path) - 1):
        add_tube(mesh, path[i], path[i + 1], r, r, tag, seg=seg, caps=True)


# ==========================================================================
# 4. ЧАСТИ ПЛАНЕРА
# ==========================================================================
def build_fuselage(mesh):
    mesh.new_part()
    seg = 30
    xs = []
    for i in range(len(FUS_STATIONS) - 1):
        x0, x1 = FUS_STATIONS[i], FUS_STATIONS[i + 1]
        n = 3 if abs(x1 - x0) > 0.35 else 1
        for k in range(n):
            xs.append(lerp(x0, x1, k / float(n)))
    xs.append(FUS_STATIONS[-1])
    rings, jtags = [], []
    for x in xs:
        rings.append([fus_point(x, TAU * j / seg) for j in range(seg)])
        jtags.append(["fus_top" if math.cos(TAU * (j + 0.5) / seg) > -0.34
                      else "fus_bot" for j in range(seg)])
    idx = [mesh.add_ring(r) for r in rings]
    for i in range(len(rings) - 1):
        for j in range(seg):
            j2 = (j + 1) % seg
            mesh.add_face([idx[i][j], idx[i][j2], idx[i + 1][j2], idx[i + 1][j]],
                          jtags[i][j])
    mesh.add_face(list(reversed(idx[0])), "fus_bot")
    mesh.add_face(list(idx[-1]), "fus_top")


def build_spinner(mesh):
    # удлинённый кок Mk IX (сдвинут на NOSE_DX, чуть большего диаметра)
    prof = [(4.84, 0.305), (4.90, 0.320), (4.96, 0.325), (5.04, 0.317),
            (5.12, 0.289), (5.20, 0.234), (5.255, 0.152), (5.29, 0.061),
            (5.30, 0.010)]
    seg = 24
    rings = [[(x, r * math.sin(TAU * j / seg), r * math.cos(TAU * j / seg))
              for j in range(seg)] for x, r in prof]
    mesh.loft(rings, "spinner", closed=True, cap_start=True, cap_end=True)


def build_blade(mesh):
    """Лопасть винта: ось Z — по радиусу, хорда вдоль X, толщина вдоль Y."""
    R = SPEC["prop_dia"] * 0.5
    prof = [(0.08, 0.080, 0.026), (0.16, 0.160, 0.034), (0.28, 0.255, 0.034),
            (0.42, 0.310, 0.031), (0.58, 0.320, 0.028), (0.74, 0.302, 0.024),
            (0.88, 0.258, 0.019), (0.96, 0.208, 0.016), (1.0, 0.150, 0.013)]
    ts = [0.0, 0.06, 0.16, 0.30, 0.5, 0.7, 0.84, 0.94, 1.0]
    rings = []
    for u, chord, thick in prof:
        chord *= 1.08  # широкие лопасти Rotol Jablo (Mk IX)
        r = u * R
        twist = (40.0 - 30.0 * u) * D2R
        ring = []
        for sgn in (1, -1):
            seq = ts if sgn == 1 else list(reversed(ts))
            for t in seq:
                du = (t - 0.35) * chord
                f = max(0.05, 1.0 - ((t - 0.35) / 0.66) ** 2)
                hz = 0.5 * thick * math.sqrt(f)
                ring.append((X_PROP + du * math.cos(twist),
                             du * math.sin(twist) + hz * sgn, r))
        rings.append(ring)
    mesh.loft(rings, "prop_blade", closed=True, cap_start=True, cap_end=True)


def build_propeller(mesh):
    for k in range(SPEC["prop_blades"]):
        m = mesh.mark()
        build_blade(mesh)
        ang = TAU * k / SPEC["prop_blades"]
        ca, sa = math.cos(ang), math.sin(ang)
        mesh.transform(m[0], lambda p: (p[0], p[1] * ca - p[2] * sa,
                                        p[1] * sa + p[2] * ca))


CANOPY = [
    (0.80, 0.105, 0.705), (0.70, 0.180, 0.742), (0.58, 0.245, 0.768),
    (0.40, 0.300, 0.786), (0.10, 0.335, 0.796), (-0.30, 0.345, 0.798),
    (-0.70, 0.342, 0.792), (-1.05, 0.335, 0.778), (-1.35, 0.320, 0.752),
    (-1.62, 0.292, 0.710), (-1.85, 0.245, 0.642), (-2.02, 0.172, 0.552),
    (-2.12, 0.085, 0.460),
]


def build_canopy(mesh):
    seg = 14
    rings = []
    for x, w, ztop in CANOPY:
        zbase = fus_top_z(x, w) - 0.035
        h = max(ztop - zbase, 0.02)
        ring = [(x, w * math.cos(math.pi * i / seg),
                 zbase + h * math.sin(math.pi * i / seg)) for i in range(seg + 1)]
        for i in range(1, 4):
            ring.append((x, lerp(-w, w, i / 4.0), zbase - 0.02))
        rings.append(ring)
    mesh.loft(rings, "canopy_glass", closed=True, cap_start=True, cap_end=True)
    # рамы фонаря
    for xf in (0.70, 0.40, -0.42, -1.35):
        w = ztop = None
        for (xa, wa, za), (xb, wb, zb) in zip(CANOPY, CANOPY[1:]):
            if xa >= xf >= xb:
                t = (xa - xf) / (xa - xb)
                w, ztop = lerp(wa, wb, t), lerp(za, zb, t)
                break
        if w is None:
            continue
        zbase = fus_top_z(xf, w) - 0.035
        h = max(ztop - zbase, 0.02) + 0.012
        path = [(xf, (w + 0.012) * math.cos(math.pi * i / seg),
                 zbase + h * math.sin(math.pi * i / seg)) for i in range(seg + 1)]
        add_swept(mesh, path, 0.015, "canopy_frame", seg=6)
    # зеркало заднего вида и его кронштейн
    add_tube(mesh, (0.585, 0.0, 0.762), (0.560, 0.0, 0.805), 0.020, 0.014,
             "canopy_frame", seg=8)
    add_box(mesh, (0.552, 0.0, 0.828), (0.030, 0.145, 0.052), "mirror")


FIN_LE = [(-2.78, 0.05), (-2.71, 0.54), (-2.86, 0.64), (-2.93, 0.80),
          (-3.01, 0.95), (-3.08, 1.10), (-3.16, 1.25), (-3.26, 1.40),
          (-3.40, 1.52), (-3.48, 1.56)]
FIN_TE = [(-3.62, 1.56), (-3.76, 1.43), (-3.90, 1.24), (-4.01, 1.04),
          (-4.08, 0.84), (-4.13, 0.64), (-4.14, 0.50), (-4.14, 0.30),
          (-4.13, 0.08)]
FIN_HINGE = 0.62          # доля хорды до шарнира руля направления


def _interp_curve(pts, z):
    """Интерполяция x по z для таблиц кромок киля."""
    pts = sorted(pts, key=lambda p: p[1])
    if z <= pts[0][1]:
        return pts[0][0]
    if z >= pts[-1][1]:
        return pts[-1][0]
    for i in range(len(pts) - 1):
        z0, z1 = pts[i][1], pts[i + 1][1]
        if z0 <= z <= z1:
            t = (z - z0) / max(1e-6, z1 - z0)
            return lerp(pts[i][0], pts[i + 1][0], t)
    return pts[-1][0]


def fin_chord(z):
    return _interp_curve(FIN_LE, z), _interp_curve(FIN_TE, z)


def fin_half_thick(z):
    return lerp(0.050, 0.016, clamp((z - 0.02) / 1.54, 0.0, 1.0))


def fin_half_thick_t(z, t, shrink_top=1.0):
    ht = fin_half_thick(z) * shrink_top
    return ht * math.sqrt(max(0.05, 1.0 - ((t - 0.30) / 0.72) ** 2))


def fin_section(z, t0, t1, n, shrink_top=1.0):
    """Сечение киля/руля: обвод по (t0..t1) хорды, толщина по Y."""
    x_le, x_te = fin_chord(z)
    chord = x_le - x_te
    up, lo = [], []
    for k in range(n + 1):
        t = lerp(t0, t1, k / float(n))
        x = x_le - t * chord
        th = fin_half_thick_t(z, t, shrink_top)
        up.append((x, th, z))
        lo.append((x, -th, z))
    return up + list(reversed(lo))


def build_fin(mesh):
    zs = [0.05, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.35, 1.47, 1.56]
    rings = []
    for i, z in enumerate(zs):
        shrink = 0.45 if i == len(zs) - 1 else 1.0
        rings.append(fin_section(z, 0.0, FIN_HINGE, 9, shrink))
    jtags = ["fin_left" if j < 10 else "fin_right" for j in range(len(rings[0]))]
    mesh.loft(rings, jtags, closed=True, cap_start=True, cap_end=True,
              cap_tag="fin_left")


def build_rudder(mesh):
    zs = [0.08, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.35, 1.47, 1.56]
    rings = []
    for i, z in enumerate(zs):
        shrink = 0.45 if i == len(zs) - 1 else 1.0
        rings.append(fin_section(z, FIN_HINGE - 0.012, 1.0, 9, shrink))
    jtags = ["rud_left" if j < 9 else "rud_right" for j in range(len(rings[0]))]
    mesh.loft(rings, jtags, closed=True, cap_start=True, cap_end=True,
              cap_tag="rud_left")


WING_HINGE_IN = 1.90    # где начинается элерон по размаху
WING_HINGE_T = 0.68     # доля хорды до шарнира элерона


def wing_hinge_t(y):
    u = clamp((abs(y) - WING_HINGE_IN) / 0.40, 0.0, 1.0)
    return lerp(1.0, WING_HINGE_T, u)


def build_wing(mesh, side=1):
    ys = [0.0, 0.5, 1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.0,
          5.2, 5.35, 5.47, 5.56, 5.615]
    rings = []
    for y in ys:
        sec = airfoil_range(wing_le_x(y), wing_chord(y), wing_zref(y),
                            wing_tc(y), wing_inc(y), 0.025, 0.0,
                            min(wing_hinge_t(y), 1.0), n=14)
        rings.append([(px, side * y, pz) for (px, pz) in sec])
    jtags = ["wing_top" if j < 15 else "wing_bot" for j in range(len(rings[0]))]
    mesh.loft(rings, jtags, closed=True, cap_start=True, cap_end=True,
              cap_tag="wing_top")


def build_aileron(mesh, side=1):
    ys = [1.98, 2.30, 2.80, 3.40, 4.00, 4.60, 5.00, 5.20, 5.35, 5.47, 5.56, 5.615]
    rings = []
    for y in ys:
        t0 = min(wing_hinge_t(y) - 0.012, 0.97)
        sec = airfoil_range(wing_le_x(y), wing_chord(y), wing_zref(y),
                            wing_tc(y), wing_inc(y), 0.025, t0, 1.0, n=13)
        rings.append([(px, side * y, pz) for (px, pz) in sec])
    jtags = ["ail_top" if j < 14 else "ail_bot" for j in range(len(rings[0]))]
    mesh.loft(rings, jtags, closed=True, cap_start=True, cap_end=True,
              cap_tag="ail_top")


TAIL = {
    "span": 1.72,
    "xc_root": -3.22, "xc_tip": -3.28,
    "chord_root": 1.84, "ell_a": 1.75,
    "hinge_root": 0.62, "hinge_tip": 0.60,   # доля хорды до шарнира руля
}


def tail_geom(y):
    u = clamp(abs(y) / TAIL["span"], 0.0, 1.0)
    a = min(abs(y), TAIL["ell_a"] - 1e-6) / TAIL["ell_a"]
    e = math.sqrt(max(0.0, 1.0 - a * a))
    chord = TAIL["chord_root"] * e
    xc = lerp(TAIL["xc_root"], TAIL["xc_tip"], (1.0 - e) ** 0.9)
    x_le = xc + chord / 2.0
    z = 0.115 + 0.020 * u
    tc = lerp(0.115, 0.078, u)
    return x_le, chord, z, tc, u


def build_tailplane(mesh, side=1):
    ys = [0.0, 0.35, 0.70, 1.05, 1.35, 1.58, 1.72]
    rings = []
    for y in ys:
        x_le, chord, z, tc, u = tail_geom(y)
        t1 = lerp(TAIL["hinge_root"], TAIL["hinge_tip"], u)
        sec = airfoil_range(x_le, chord, z, tc, -1.2, 0.0, 0.0, t1, n=7)
        rings.append([(px, side * y, pz) for (px, pz) in sec])
    jtags = ["tail_top" if j < 8 else "tail_bot" for j in range(len(rings[0]))]
    mesh.loft(rings, jtags, closed=True, cap_start=True, cap_end=True,
              cap_tag="tail_top")


def build_elevator(mesh, side=1):
    ys = [0.10, 0.45, 0.80, 1.12, 1.40, 1.60, 1.72]
    rings = []
    for y in ys:
        x_le, chord, z, tc, u = tail_geom(y)
        t0 = lerp(TAIL["hinge_root"], TAIL["hinge_tip"], u) - 0.012
        sec = airfoil_range(x_le, chord, z, tc, -1.2, 0.0, t0, 1.0, n=8)
        rings.append([(px, side * y, pz) for (px, pz) in sec])
    jtags = ["elev_top" if j < 8 else "elev_bot" for j in range(len(rings[0]))]
    mesh.loft(rings, jtags, closed=True, cap_start=True, cap_end=True,
              cap_tag="elev_top")


def build_main_gear(mesh, side=1, extended=True):
    hip = (GEAR_PIVOT[0], side * GEAR_PIVOT[1],
           wing_surface_z(GEAR_PIVOT[0], GEAR_PIVOT[1], lower=True) + 0.05)
    src = GEAR_DOWN if extended else GEAR_UP
    ax = (src[0], side * src[1], src[2])
    knee = tuple(lerp(hip[i], ax[i], 0.46) for i in range(3))
    # амортизационная стойка: кожух сверху, шток снизу
    add_tube(mesh, hip, knee, 0.082, 0.068, "oleo_sleeve", seg=14)
    add_tube(mesh, knee, ax, 0.058, 0.052, "gear_leg", seg=14)
    # подкос из ниши крыла
    add_tube(mesh, (hip[0] + 0.30, hip[1] - side * 0.03, hip[2] + 0.05),
             (knee[0] + 0.05, knee[1] + side * 0.03, knee[2] - 0.03),
             0.028, 0.022, "gear_link", seg=8)
    # тормозная магистраль
    add_tube(mesh, (hip[0] - 0.09, hip[1] + side * 0.04, hip[2] - 0.02),
             (ax[0] - 0.03, ax[1] + side * 0.08, ax[2] + 0.19), 0.015, 0.012,
             "gear_link", seg=6)
    # колесо (профиль с округлым протектором)
    wprof = [(0.00, 0.250), (0.09, 0.345), (0.30, 0.372), (0.50, 0.375),
             (0.70, 0.372), (0.91, 0.345), (1.00, 0.250)]
    add_tube(mesh, (ax[0], ax[1] - side * 0.082, ax[2]),
             (ax[0], ax[1] + side * 0.082, ax[2]), 0.375, stations=wprof,
             tag="tyre", seg=24)
    add_tube(mesh, (ax[0], ax[1] - side * 0.072, ax[2]),
             (ax[0], ax[1] + side * 0.072, ax[2]), 0.205, 0.205,
             "wheel_hub", seg=20)
    add_tube(mesh, (ax[0], ax[1] - side * 0.120, ax[2]),
             (ax[0], ax[1] + side * 0.120, ax[2]), 0.075, 0.075, "wheel_hub",
             seg=10)
    # щиток ниши шасси — вертикальная панель вдоль стойки
    ydoor = hip[1] + side * 0.150
    add_plate(mesh, (hip[0] + 0.26, ydoor, hip[2] + 0.02),
              (ax[0] + 0.20, ydoor, ax[2] + 0.20), (1.0, 0.0, 0.0), 0.20,
              (0.0, 1.0, 0.0), 0.012, "gear_door")


def build_tailwheel(mesh):
    z0 = fus_zc(-3.60) - fus_rz(-3.60) + 0.02
    add_tube(mesh, (-3.62, 0.0, z0), (-3.58, 0.0, -0.44), 0.036, 0.028,
             "gear_leg", seg=8)
    add_tube(mesh, (-3.58, -0.055, -0.50), (-3.58, 0.055, -0.50), 0.115, 0.115,
             "tyre", seg=14)
    add_tube(mesh, (-3.58, -0.065, -0.50), (-3.58, 0.065, -0.50), 0.055, 0.055,
             "wheel_hub", seg=10)


def build_details(mesh):
    # антенная мачта
    xm = -0.95
    add_tube(mesh, (xm, 0.0, fus_top_z(xm, 0.0) - 0.04),
             (xm - 0.30, 0.0, fus_top_z(xm - 0.30, 0.0) + 0.19), 0.036, 0.014,
             "canopy_frame", seg=8)
    # выхлопные патрубки Merlin 61: 6 на борт, эжекторные, увеличенные
    for side in (1, -1):
        for k in range(6):
            x = 3.82 - k * 0.108
            p0 = fus_point(x, side * 0.62 * math.pi)
            p1 = v_add(p0, (0.0, side * 0.095, -0.045))
            add_tube(mesh, (p0[0] + 0.045, p0[1], p0[2]), p1, 0.060, 0.048,
                     "exhaust", seg=8)
    # крыло типа C (Mk IXc): 1 × Hispano 20 мм + 2 × Browning .303 на консоль
    for side in (1, -1):
        # пушка Hispano: ствол выступает перед передней кромкой
        yc = 2.32
        xc = wing_le_x(yc)
        zc = wing_zref(yc) - 0.01
        add_tube(mesh, (xc - 1.10, side * yc, zc), (xc + 0.52, side * yc, zc),
                 0.042, 0.030, "gun", seg=10)
        # обтекатель ствола у передней кромки + выпуклость барабана сверху
        add_tube(mesh, (xc - 0.28, side * yc, zc), (xc + 0.16, side * yc, zc),
                 0.072, 0.052, "cannon_fairing", seg=10)
        zt = wing_surface_z(xc - 0.35, yc) + 0.005
        add_tube(mesh, (xc - 0.62, side * yc, zt), (xc - 0.08, side * yc, zt),
                 0.095, 0.055, "cannon_bulge", seg=10)
        # пулемёты Browning: внутренний и внешний от пушки
        for k, y in enumerate((1.98, 2.62)):
            xm = wing_le_x(y) + 0.12 - 0.012 * k
            z = wing_zref(y) - 0.01
            add_tube(mesh, (xm - 0.95, side * y, z), (xm, side * y, z),
                     0.030, 0.022, "gun", seg=8)


def build_chin_intake(mesh):
    # воздухозаборник карбюратора под носом: широкий «ковш» сразу за коком
    # (как на фото NH341/MH434), под Merlin 61
    x0, x1, r, ztop = 4.00, 4.80, 0.20, -0.31
    add_fairing(mesh, x0, x1, 0.0, r, ztop, "chin", seg=10)
    # тёмное входное окно спереди (выступает из обтекателя)
    add_plate(mesh, (x1 - 0.035, 0.0, ztop - r * 0.42),
              (x1 + 0.006, 0.0, ztop - r * 0.42),
              (0.0, 1.0, 0.0), r * 0.62, (0.0, 0.0, 1.0), r * 0.40,
              "radiator_face")


def build_radiators(mesh):
    # Mk IX: интеркулер наддува — левый борт, гликолевый радиатор — правый;
    # оба увеличены относительно Mk I
    spec = [(+1, 0.50, 1.02, 0.620, 0.255), (-1, -0.46, 0.50, 0.740, 0.300)]
    for side, x0, x1, y_c, r in spec:
        y = side * y_c
        rings = []
        n = 6
        for k in range(n + 1):
            s01 = k / float(n)
            x = lerp(x0, x1, s01)
            shrink = 1.0 - 0.34 * max(0.0, (abs(s01 - 0.5) * 2 - 0.70) / 0.30) ** 2
            rr = r * shrink
            zl = wing_surface_z(x, y + rr, lower=True)
            zr = wing_surface_z(x, y - rr, lower=True)
            z0 = min(zl, zr) + 0.004
            ring = [(x, y + rr, zl + 0.004), (x, y - rr, zr + 0.004)]
            for i in range(1, 10):
                a = math.pi * i / 10
                ring.append((x, y - rr * math.cos(a), z0 - rr * math.sin(a)))
            rings.append(ring)
        mesh.loft(rings, "radiator", closed=True, cap_start=True, cap_end=True)
        # входная решётка радиатора (тёмная пластина впереди)
        zc = 0.5 * (wing_surface_z(x0, y, lower=True) + wing_surface_z(x0 + 0.2, y, lower=True))
        add_plate(mesh, (x0 + 0.012, y, zc - r * 0.30), (x0 + 0.030, y, zc - r * 0.30),
                  (0.0, 1.0, 0.0), r * 0.72, (0.0, 0.0, 1.0), r * 0.62,
                  "radiator_face")


def build_antiglare(mesh):
    rings = []
    for x in (0.76, 1.05, 1.35, 1.62):
        ws = (-0.30, -0.16, 0.0, 0.16, 0.30)
        ring = [(x, w, fus_top_z(x, w) + 0.010) for w in ws]
        ring.append((x, 0.30, fus_top_z(x, 0.30) - 0.05))
        ring.append((x, -0.30, fus_top_z(x, -0.30) - 0.05))
        rings.append(ring)
    mesh.loft(rings, "antiglare", closed=True, cap_start=True, cap_end=True)


# ==========================================================================
# 5. ОПОЗНАВАТЕЛЬНЫЕ ЗНАКИ
# ==========================================================================
def ring_on_surface(mesh, outer_pts, inner_pts, normal, tag, off=0.016, deep=0.05):
    """Кольцо-накладка, повторяющее кривизну обшивки (замкнутый тор)."""
    pid = mesh.new_part()
    n = v_norm(normal)
    o1 = [v_add(p, v_mul(n, off)) for p in outer_pts]
    o2 = [v_add(p, v_mul(n, off - deep)) for p in outer_pts]
    if inner_pts is None:
        # сплошной диск: крышка веером от центра
        c = (sum(p[0] for p in o1) / len(o1), sum(p[1] for p in o1) / len(o1),
             sum(p[2] for p in o1) / len(o1))
        rings = [o1, o2]
        idx = [mesh.add_ring(r) for r in rings]
        for k in range(len(o1)):
            k2 = (k + 1) % len(o1)
            mesh.add_face([idx[0][k], idx[0][k2], idx[1][k2], idx[1][k]], tag)
        ci = mesh.add_vert(v_add(c, v_mul(n, off)))
        for k in range(len(o1)):
            k2 = (k + 1) % len(o1)
            mesh.add_face([idx[0][k], ci, idx[0][k2]], tag)
        c2 = mesh.add_vert(v_add(c, v_mul(n, off - deep)))
        for k in range(len(o1)):
            k2 = (k + 1) % len(o1)
            mesh.add_face([idx[1][k2], c2, idx[1][k]], tag)
        return
    i1 = [v_add(p, v_mul(n, off)) for p in inner_pts]
    i2 = [v_add(p, v_mul(n, off - deep)) for p in inner_pts]
    rings = [o1, i1, i2, o2]
    mesh.torus_loft(rings, tag)


def build_wing_roundel(mesh, side=1, lower=False):
    y0 = 3.55 if not lower else 3.35
    x0 = 0.5 * (wing_le_x(y0) + wing_te_x(y0)) + (0.06 if not lower else 0.0)
    y0 *= side
    nrm = wing_normal(x0, y0, lower)
    n_ang = 30

    def circ(r):
        pts = []
        for k in range(n_ang):
            a = TAU * k / n_ang
            px = x0 + r * math.cos(a)
            py = y0 + r * math.sin(a)
            pts.append((px, py, wing_surface_z(px, py, lower) + 0.004))
        return pts

    if lower:
        spec = [(0.560, 0.300, "roundel_blue"), (0.300, 0.150, "roundel_white")]
        red_r = 0.150
    else:
        # верхние — тип B (без белого): синее кольцо + красный центр 1:3
        spec = [(0.560, 0.187, "roundel_blue")]
        red_r = 0.187
    for r_out, r_in, name in spec:
        ring_on_surface(mesh, circ(r_out), circ(r_in), nrm, name)
    ring_on_surface(mesh, circ(red_r), None, nrm, "roundel_red")


def build_fuselage_roundel(mesh, side=1):
    x0 = -1.12
    phi0 = side * math.pi * 0.5
    r_loc = fus_ry(x0)
    n_ang = 30

    def circ(r):
        pts = []
        for k in range(n_ang):
            a = TAU * k / n_ang
            px = x0 + r * math.cos(a)
            pphi = phi0 + (r * math.sin(a)) / r_loc
            pts.append(fus_point(px, pphi))
        return pts

    nrm = (0.0, math.sin(phi0), math.cos(phi0))
    spec = [(0.400, 0.240, "roundel_yellow"), (0.240, 0.130, "roundel_blue"),
            (0.130, 0.068, "roundel_white")]
    for r_out, r_in, name in spec:
        ring_on_surface(mesh, circ(r_out), circ(r_in), nrm, name)
    ring_on_surface(mesh, circ(0.068), None, nrm, "roundel_red")


def build_fin_flash(mesh, side=1):
    """Флажок RAF на киле: синий — белый — красный (от носка к задней кромке)."""
    stripes = [(0.45, 0.60, "roundel_blue"), (0.60, 0.75, "roundel_white"),
               (0.75, 0.90, "roundel_red")]
    zs = [0.70, 0.88, 1.06, 1.24, 1.38]
    for t0, t1, name in stripes:
        rings = []
        for z in zs:
            x_le, x_te = fin_chord(z)
            chord = x_le - x_te
            xa = x_le - t1 * chord
            xb = x_le - t0 * chord
            ht = fin_half_thick(z) + 0.008
            if z > 1.30:
                ht *= lerp(1.0, 0.55, clamp((z - 1.30) / 0.26, 0.0, 1.0))
            rings.append([(xa, ht, z), (xb, ht, z), (xb, -ht, z), (xa, -ht, z)])
        mesh.loft(rings, name, closed=True, cap_start=True, cap_end=True)


# ==========================================================================
# 6. КАМУФЛЯЖ (схема «A», цвета 1943-44: Dark Green / Ocean Grey)
# ==========================================================================
CAMO_BND = Curve1D([
    (0.0, 1.95), (0.6, 1.30), (1.2, 0.50), (1.8, 0.82), (2.4, 0.00),
    (3.0, 0.55), (3.6, -0.20), (4.2, 0.28), (4.8, -0.45), (5.4, -0.62),
    (5.62, -0.65),
])


def camo_green_wing(x, y):
    s = abs(y)
    return x > CAMO_BND(s if y >= 0 else s + 1.35)


def camo_green_fuse(x, y):
    if x < -3.0:
        return x > (-3.45 + 0.55 * math.sin(2.4 * y + 1.0))
    b = -0.10 + 0.95 * math.sin(1.75 * y + 1.35) + 0.55 * math.sin(0.85 * y + 2.7)
    return x > b


# ==========================================================================
# 7. СБОРКА + РАСКРАСКА
# ==========================================================================
def paint(face):
    t = face.tag
    c = face.cen
    if t in ("fus_top", "fus_bot"):
        # полоса Sky (18") вокруг задней части фюзеляжа — как на ML296/NH341
        if -2.95 <= c[0] <= -2.50:
            return M_SPIN
        if t == "fus_bot":
            return M_SEA
        return M_GREEN if camo_green_fuse(c[0], c[1]) else M_OCEAN
    if t in ("wing_top", "tail_top", "elev_top", "ail_top",
             "cannon_fairing", "cannon_bulge"):
        if t == "wing_top":
            # жёлтые опознавательные полосы на передней кромке (см. фото MH434)
            ch = wing_chord(c[1])
            tt = (wing_le_x(c[1]) - c[0]) / ch
            if tt < 0.10 and 0.20 < abs(c[1]) < 2.45:
                return M_YELLOW
        return M_GREEN if camo_green_wing(c[0], c[1]) else M_OCEAN
    if t in ("wing_bot", "tail_bot", "elev_bot", "ail_bot"):
        return M_SEA
    if t in ("fin_left", "fin_right", "rud_left", "rud_right"):
        return M_GREEN if camo_green_fuse(c[0], c[1]) else M_OCEAN
    if t in ("spinner",):
        return M_SPIN
    if t == "prop_blade":
        r = math.sqrt(c[1] ** 2 + c[2] ** 2)
        return M_TIP if r > 0.86 * SPEC["prop_dia"] * 0.5 else M_BLADE
    if t == "canopy_glass":
        return M_GLASS
    if t in ("canopy_frame",):
        return M_FRAME
    if t == "mirror":
        return M_METAL
    if t == "exhaust":
        return M_SOOT if abs(c[1]) > 0.43 else M_EXH
    if t == "gun":
        return M_GUN
    if t == "antiglare":
        return M_BLACK
    if t in ("oleo_sleeve", "gear_leg", "gear_link"):
        return M_METAL
    if t == "tyre":
        return M_TYRE
    if t == "wheel_hub":
        return M_HUB
    if t == "gear_door":
        return M_SEA
    if t in ("radiator", "chin"):
        return M_SEA
    if t == "radiator_face":
        return M_MATRIX
    if t in ("roundel_red", "roundel_white", "roundel_blue", "roundel_yellow"):
        return t
    raise ValueError("нет материала для тега %r" % t)


def build():
    mesh = Mesh()
    build_fuselage(mesh)
    build_spinner(mesh)
    build_propeller(mesh)
    build_canopy(mesh)
    build_fin(mesh)
    build_rudder(mesh)
    build_details(mesh)
    build_antiglare(mesh)

    m = mesh.mark()
    build_wing(mesh, +1)
    build_aileron(mesh, +1)
    build_tailplane(mesh, +1)
    build_elevator(mesh, +1)
    build_main_gear(mesh, +1)
    build_wing_roundel(mesh, +1, lower=False)
    build_wing_roundel(mesh, +1, lower=True)
    mesh.mirror_y(*m)

    build_fuselage_roundel(mesh, +1)
    build_fuselage_roundel(mesh, -1)
    build_fin_flash(mesh, +1)
    build_fin_flash(mesh, -1)
    build_radiators(mesh)
    build_chin_intake(mesh)
    build_tailwheel(mesh)
    return mesh


def report(mesh):
    lo, hi = mesh.bounds()
    print("=" * 62)
    print("Supermarine Spitfire Mk IX — процедурная модель")
    print("=" * 62)
    print("вершин: %d, полигонов: %d, треугольников: %d"
          % (mesh.stats()[0], mesh.stats()[1], mesh.stats()[2]))
    print("габариты модели (м):")
    print("  длина  X: %6.2f  (эталон %.2f)"
          % (hi[0] - lo[0], SPEC["length"]))
    print("  размах Y: %6.2f  (эталон %.2f)"
          % (hi[1] - lo[1], SPEC["span"]))
    print("  высота Z: %6.2f" % (hi[2] - lo[2]))
    # площадь крыла (трапеции по сечениям)
    ys = [0.0, 0.5, 1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.0]
    area = 0.0
    for i in range(len(ys) - 1):
        area += 0.5 * (wing_chord(ys[i]) + wing_chord(ys[i + 1])) * (ys[i + 1] - ys[i])
    for i in range(24):
        y0 = 5.0 + (Y_TIP - 5.0) * i / 24.0
        y1 = 5.0 + (Y_TIP - 5.0) * (i + 1) / 24.0
        area += 0.5 * (wing_chord(y0) + wing_chord(y1)) * (y1 - y0)
    area *= 2.0
    print("  площадь крыла: %.2f м²  (эталон %.2f)" % (area, SPEC["wing_area"]))
    print("  диаметр винта: %.2f м" % SPEC["prop_dia"])


def main():
    mesh = build()
    mesh.finalize()
    flipped = mesh.fix_orientation()
    print("оболочек: %d, развёрнуто наружу: %d" % (len(mesh.part_faces), flipped))
    bad = mesh.open_shells()
    if bad:
        print("ВНИМАНИЕ: незакрытые оболочки:")
        for pid, n, cnt, tag in bad[:20]:
            print("   #%d (%s): граней %d, открытых рёбер %d" % (pid, tag, cnt, n))
    mesh.paint(paint)
    mesh.finalize()
    report(mesh)

    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    comment = ("Supermarine Spitfire Mk IX — procedural model, "
               "github.com/Klischa/3D/spitfire-mk-ix")
    p = export_obj(mesh, os.path.join(OUT, "spitfire.obj"), MATERIALS, comment)
    print("Записано: %s" % p)
    p = export_glb(mesh, os.path.join(OUT, "spitfire.glb"), MATERIALS,
                   "Supermarine Spitfire Mk IX")
    print("Записано: %s (%.2f МБ)" % (p, os.path.getsize(p) / 1048576.0))
    p = export_stl(mesh, os.path.join(OUT, "spitfire.stl"), "Spitfire Mk IX")
    print("Записано: %s (%.2f МБ)" % (p, os.path.getsize(p) / 1048576.0))
    p = export_materials_json(MATERIALS, os.path.join(OUT, "materials.json"))
    print("Записано: %s" % p)


if __name__ == "__main__":
    main()
