# -*- coding: utf-8 -*-
"""ДИАГНОСТИЧЕСКАЯ ДЕМОНСТРАЦИЯ (не вариант модели для печати).

Отвечает на вопрос «а что если развернуть ТОЛЬКО корпус?».

Строит ту же модель, что и generate_spitfire.py, а затем поворачивает на 180°
вокруг вертикальной оси один только фюзеляж — обшивку (fus_top/fus_bot) вместе
с фонарём, антенной мачтой, противопылевой накладкой, зеркалом, кабиной и
бортовыми опознавательными знаками. Винт, двигатель (выхлоп, воздухозаборник),
крылья, шасси и хвостовое оперение при этом остаются ровно там же.

Результат — картинка-доказательство, как выглядел бы такой «мутант». В
настоящем файле spitfire.obj/glb корпус стоит правильно, и на этой картинке
видно, насколько она не похожа на нормальный самолёт.

Запуск из папки spitfire-mk-ix:
    python3 tools/demo_flip_body.py [папка_вывода]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generate_spitfire as g            # noqa: E402
from render_preview import load_obj, load_mtl, render  # noqa: E402


def body_face(f):
    """Нужно ли относить грань к «корпусу» (то, что поворачиваем)."""
    t, c = f.tag, f.cen
    if t in ("fus_top", "fus_bot",            # обшивка фюзеляжа
             "canopy_frame", "canopy_glass",  # фонарь, переплёт, мачта
             "antiglare", "mirror"):          # накладка перед фонарём, зеркало
        return True
    # кабина изнутри: ручка, прицел, приборная доска (тег gun, но |y| < 0.5)
    if t == "gun" and abs(c[1]) < 0.50 and -1.2 < c[0] < 1.0:
        return True
    # опознавательные знаки на бортах фюзеляжа (не на крыле и не на киле)
    if (t in ("roundel_blue", "roundel_red", "roundel_white")
            and abs(c[1]) < 0.60 and -2.0 < c[0] < -0.4):
        return True
    return False


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flipdemo"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    mesh = g.build()
    mesh.finalize()

    sel = set()
    for f in mesh.faces:
        if body_face(f):
            sel.update(f.v)
    xs = [mesh.verts[i][0] for i in sel]
    x0, x1 = min(xs), max(xs)
    s = x0 + x1
    print("корпус: x[%.3f, %.3f], длина %.3f м" % (x0, x1, x1 - x0))
    print("поворот на 180° вокруг вертикальной оси x=%.3f" % (s * 0.5))
    print("вершин в корпусе: %d из %d" % (len(sel), len(mesh.verts)))

    for i in sel:
        x, y, z = mesh.verts[i]
        mesh.verts[i] = (s - x, -y, z)

    mesh.finalize()
    print("развёрнуто оболочек наружу обратно: %d" % mesh.fix_orientation())
    mesh.paint(g.paint)
    mesh.finalize()

    obj = os.path.join(outdir, "body_flip.obj")
    g.export_obj(mesh, obj, g.MATERIALS,
                 "WHAT-IF: повёрнут только корпус. НЕ модель для печати.")
    print("записано: %s" % obj)

    mats = load_mtl(os.path.splitext(obj)[0] + ".mtl")
    verts, norms, faces = load_obj(obj)
    jobs = (
        ("flip_side_r.png", "side_r", (1500, 820)),
        ("flip_top.png", "top", (1400, 860)),
        ("flip_persp.png", "persp", (1400, 900)),
    )
    for name, view, size in jobs:
        out = os.path.join(outdir, name)
        render(verts, norms, faces, mats, out, size, view, 32.0, None)
        print("  -> %s (%.0f КБ)" % (name, os.path.getsize(out) / 1024.0))


if __name__ == "__main__":
    main()
