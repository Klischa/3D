"""CLI gt_collector: collect / selftest / report / pairs / stats /
save-calibration / check-board / print-board / devices.

Запуск: ``python -m gt_collector <команда> ...``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import BoardSpec, SessionConfig


def _print_board_choices() -> None:
    print("Пресеты дошек: " + ", ".join(
        f"{k} ({v.tag_count} тэгов, {v.width_m*1000:.0f}×{v.height_m*1000:.0f} мм, "
        f"тег {v.tag_mm:.0f} мм)" for k, v in BoardSpec.presets().items()))


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def _make_backend(args):
    if args.backend == "fake":
        from .fake import FakeBackend, FakeConfig
        fcfg = FakeConfig(
            duration_s=args.duration,
            fps=args.fps,
            trajectory=args.trajectory,
            orbit_radius_m=args.radius,
            seed=args.seed,
            board_preset=args.board,
        )
        return FakeBackend(fcfg, BoardSpec.from_preset(args.board))
    from .backends import OrbbecBackend
    return OrbbecBackend(name_hint=args.device_name, serial=args.serial, fps=args.fps)


def cmd_collect(args) -> int:
    backend = _make_backend(args)
    cfg = SessionConfig(
        board=BoardSpec.from_preset(args.board),
        duration_s=args.duration,
        warmup_frames=args.warmup,
        min_tags=args.min_tags,
        max_residual_mm=args.max_residual_mm,
        min_points=args.min_points,
        save_frames_every=args.save_every,
        fps=args.fps,
    )
    if args.scene:
        cfg.scene = args.scene
    if args.operator:
        cfg.operator = args.operator
    from .session import Session
    sess = Session(args.out, cfg, backend)
    manifest = sess.run()
    from .pairs import save_pairs
    from .report import save_report
    save_pairs(args.out)
    save_report(args.out)
    print(json.dumps({
        "frames": manifest["frames_written"],
        "flags": manifest["flags_summary"],
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def cmd_selftest(args) -> int:
    from .calibrate import verify_board
    from .fake import FakeBackend, FakeConfig
    from .pairs import extract_pairs
    from .report import build_report
    from .session import Session

    board = BoardSpec.from_preset(args.board)
    print(f"[1/5] Проверка раскладки дошки (декодирование битмапа) ...", flush=True)
    n = verify_board(board)
    print(f"      {n}/{board.tag_count} тэгов декодированы по сетке.")

    results = {}
    quick = args.quick

    def run_session(name: str, trajectory: str, duration: float, fps: int):
        out = Path(args.out) / name
        fcfg = FakeConfig(duration_s=duration, fps=fps, trajectory=trajectory,
                          orbit_radius_m=args.radius, seed=args.seed,
                          board_preset=args.board)
        # облака — не ежесекундно: round-trip ACGD проверяется по первому
        # файлу, а полный selftest на eжесекундных облаках жрёт ~9 ГБ
        cfg = SessionConfig(board=board, duration_s=duration, warmup_frames=5,
                            fps=fps, save_frames_every=10)
        be = FakeBackend(fcfg, board)
        sess = Session(out, cfg, be)
        manifest = sess.run()
        # GT для сравнения: тот же seed → та же траектория
        gt = FakeBackend(FakeConfig(duration_s=duration, fps=fps, trajectory=trajectory,
                                    orbit_radius_m=args.radius, seed=args.seed,
                                    board_preset=args.board), board)
        return out, manifest, gt

    # --- статичная сессия: noise floor ------------------------------------
    print("[2/5] Статичная сессия (noise floor GT) ...", flush=True)
    dur_s = 6 if quick else 12
    out_st, man_st, gt_st = run_session("selftest_static", "static", dur_s, 30)
    rep_st = build_report(out_st)
    nf = rep_st.get("noise_floor")
    results["static_valid_ratio"] = rep_st["valid_ratio"]
    results["noise_floor"] = nf
    assert nf is not None, "Не оценён noise floor (нет статичных кадров)"
    assert nf["translation_p95_mm"] < 5.0, f"Noise floor по сдвигу: {nf}"
    assert nf["rotation_p95_deg"] < 1.0, f"Noise floor по повороту: {nf}"
    print(f"      noise floor: {nf['translation_p95_mm']:.2f} мм / "
          f"{nf['rotation_p95_deg']:.3f}° (p95)")

    # --- орбита: точность GT и пары ----------------------------------------
    print("[3/5] Орбита (точность GT, пары) ...", flush=True)
    dur_o = 8 if quick else 30
    out_or, man_or, gt_or = run_session("selftest_orbit", "orbit", dur_o, 30)
    from .pairs import load_poses
    from scipy.spatial.transform import Rotation
    rows = load_poses(out_or)
    # frame_idx в poses.csv = idx из цикла сессии (warmup не пишется)
    errs_rot, errs_tr = [], []
    for r in rows:
        if "no_board" in r["flags"]:
            continue
        idx = int(r["frame_idx"])
        R_gt, t_gt = gt_or._pose(idx)
        R_hat = Rotation.from_quat([float(r["qx"]), float(r["qy"]),
                                    float(r["qz"]), float(r["qw"])]).as_matrix()
        errs_rot.append(np.degrees(Rotation.from_matrix(R_hat @ R_gt.T).magnitude()))
        errs_tr.append(np.linalg.norm([float(r["tx"]), float(r["ty"]), float(r["tz"])]
                                      - t_gt) * 1000.0)
    errs_rot = np.array(errs_rot)
    errs_tr = np.array(errs_tr)
    results["orbit_valid_ratio"] = man_or["flags_summary"].get("ok", 0) / max(1, man_or["frames_written"])
    results["pose_err_rot_deg"] = {"mean": float(errs_rot.mean()), "p95": float(np.percentile(errs_rot, 95))}
    results["pose_err_trans_mm"] = {"mean": float(errs_tr.mean()), "p95": float(np.percentile(errs_tr, 95))}
    assert results["orbit_valid_ratio"] > 0.95, f"Доля годных кадров: {results}"
    assert errs_rot.mean() < 0.3, f"Средняя ошибка вращения: {errs_rot.mean():.3f}°"
    assert errs_tr.mean() < 5.0, f"Средняя ошибка сдвига: {errs_tr.mean():.2f} мм"
    print(f"      valid={results['orbit_valid_ratio']:.3f}, "
          f"pose err: {errs_rot.mean():.3f}° / {errs_tr.mean():.2f} мм (mean)")

    # --- пары ---------------------------------------------------------------
    print("[4/5] Пары: T_rel по GT против пары из poses.csv ...", flush=True)
    pairs = extract_pairs(out_or)
    assert len(pairs) > (30 if not quick else 5), f"Слишком мало пар: {len(pairs)}"
    perr_rot, perr_tr = [], []
    for p in pairs:
        Rg_s, tg_s = gt_or._pose(int(p["source_frame"]))
        Rg_t, tg_t = gt_or._pose(int(p["target_frame"]))
        M_t = np.eye(4); M_t[:3, :3] = Rg_t; M_t[:3, 3] = tg_t
        M_s = np.eye(4); M_s[:3, :3] = Rg_s; M_s[:3, 3] = tg_s
        T_rel = np.linalg.inv(M_t) @ M_s
        dT = np.linalg.inv(p["T"]) @ T_rel
        perr_rot.append(np.degrees(Rotation.from_matrix(dT[:3, :3]).magnitude()))
        perr_tr.append(np.linalg.norm(dT[:3, 3]) * 1000.0)
    perr_rot = np.array(perr_rot)
    perr_tr = np.array(perr_tr)
    results["pairs_n"] = len(pairs)
    results["pairs_err"] = {
        "rot_deg": {"mean": float(perr_rot.mean()),
                    "p95": float(np.percentile(perr_rot, 95)),
                    "max": float(perr_rot.max())},
        "trans_mm": {"mean": float(perr_tr.mean()),
                     "p95": float(np.percentile(perr_tr, 95)),
                     "max": float(perr_tr.max())},
    }
    # Ошибка пары = сумма ошибок двух кадров (каждый ~0.5°/2.5 мм p95);
    # на хвосте орбиты (крайние углы) одиночный кадр достигает ~1.3°,
    # поэтому приём по p95 + ограниченный max, а не по чистому max.
    assert float(np.percentile(perr_rot, 95)) < 1.0, \
        f"Ошибка T_rel поворот p95: {np.percentile(perr_rot, 95):.3f}°"
    assert perr_rot.max() < 2.5, f"Ошибка T_rel поворот max: {perr_rot.max():.3f}°"
    assert float(np.percentile(perr_tr, 95)) < 10.0, \
        f"Ошибка T_rel сдвиг p95: {np.percentile(perr_tr, 95):.2f} мм"
    assert perr_tr.max() < 30.0, f"Ошибка T_rel сдвиг max: {perr_tr.max():.2f} мм"
    print(f"      пар: {len(pairs)}, rot p95/max: "
          f"{np.percentile(perr_rot, 95):.3f}° / {perr_rot.max():.3f}°, "
          f"trans p95/max: {np.percentile(perr_tr, 95):.2f} / {perr_tr.max():.2f} мм")

    # --- облако: round-trip --------------------------------------------------
    print("[5/5] Облака: round-trip ACGD ...", flush=True)
    from .cloud import load_cloud
    clouds = sorted((Path(out_or) / "frames").glob("*.cloud"))
    assert clouds, "Не сохранено ни одного .cloud"
    c = load_cloud(clouds[0])
    assert c.n > 1000, f"Облако слишком маленькое: {c.n}"
    assert np.all(np.isfinite(c.xyz)), "Ненечные координаты в облаке"
    zmax = float(c.xyz[:, 2].max())
    assert 0.2 < zmax < 5.0, f"Задняя плоскость глубины вне разумного: {zmax:.2f} м"
    results["cloud_points"] = c.n
    print(f"      точек: {c.n}, дальность до {zmax:.2f} м")

    (Path(args.out) / "selftest_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSELFTEST PASS")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# остальные команды
# ---------------------------------------------------------------------------

def cmd_report(args) -> int:
    from .report import save_report
    rep = save_report(args.session)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


def cmd_pairs(args) -> int:
    from .pairs import save_pairs
    print(json.dumps(save_pairs(args.session), indent=2, ensure_ascii=False))
    return 0


def cmd_stats(args) -> int:
    from .report import build_report
    from .pairs import load_poses, extract_pairs
    rep = build_report(args.session)
    rep["pairs"] = len(extract_pairs(args.session))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


def cmd_save_calibration(args) -> int:
    from .calibrate import save_device_calibration
    data = save_device_calibration(args.out, name_hint=args.device_name, serial=args.serial)
    print(json.dumps({"saved": args.out, "K_color": data["rgb_intrinsic"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_check_board(args) -> int:
    from .calibrate import check_board_image
    board = BoardSpec.from_preset(args.board)
    res = check_board_image(args.image, board, calibration_path=args.calibration)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["ok"] else 2


def cmd_print_board(args) -> int:
    from .calibrate import print_board
    board = BoardSpec.from_preset(args.board)
    png, svg = print_board(board, args.png, args.svg)
    print(f"PNG: {png}\nSVG: {svg}\nПечать 1:1 (масштаб!). Проверьте на экране: "
          f"{board.width_m*1000:.0f}×{board.height_m*1000:.0f} мм.")
    return 0


def cmd_devices(args) -> int:
    try:
        from .backends import list_devices
    except ImportError:
        print("pyorbbecsdk2 не установлен (pip install --no-deps pyorbbecsdk2)")
        return 1
    devs = list_devices()
    if not devs:
        print("Устройства Orbbec не найдены.")
    for d in devs:
        print(f"[{d['index']}] {d['name']}  serial={d['serial']}  fw={d['firmware']}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gt_collector",
                                description="Сбор записей Astra с GT-позами по AprilTag-дошке")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("collect", help="записать сессию")
    sp.add_argument("--backend", choices=["orbbec", "fake"], default="orbbec")
    sp.add_argument("--board", default="a4")
    sp.add_argument("--out", required=True)
    sp.add_argument("--duration", type=float, default=60.0)
    sp.add_argument("--fps", type=int, default=30)
    sp.add_argument("--warmup", type=int, default=20)
    sp.add_argument("--min-tags", type=int, default=5)
    sp.add_argument("--max-residual-mm", type=float, default=20.0)
    sp.add_argument("--min-points", type=int, default=20000)
    sp.add_argument("--save-every", type=int, default=1)
    sp.add_argument("--scene", default="")
    sp.add_argument("--operator", default="")
    sp.add_argument("--device-name", default="astra")
    sp.add_argument("--serial", default=None)
    # параметры фейка
    sp.add_argument("--trajectory", choices=["orbit", "static", "sweep"], default="orbit")
    sp.add_argument("--radius", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=11)
    sp.set_defaults(fn=cmd_collect)

    sp = sub.add_parser("selftest", help="самотест на синтетическом бэкенде")
    sp.add_argument("--out", default="selftest_out")
    sp.add_argument("--board", default="a4")
    sp.add_argument("--radius", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=11)
    sp.add_argument("--quick", action="store_true", help="быстрый (для CI)")
    sp.set_defaults(fn=cmd_selftest)

    sp = sub.add_parser("report", help="отчёт по сессии")
    sp.add_argument("session")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("pairs", help="извлечь пары из сессии")
    sp.add_argument("session")
    sp.set_defaults(fn=cmd_pairs)

    sp = sub.add_parser("stats", help="краткая статистика сессии")
    sp.add_argument("session")
    sp.set_defaults(fn=cmd_stats)

    sp = sub.add_parser("save-calibration", help="заводская калибровка устройства")
    sp.add_argument("--out", default="calibration.json")
    sp.add_argument("--device-name", default="astra")
    sp.add_argument("--serial", default=None)
    sp.set_defaults(fn=cmd_save_calibration)

    sp = sub.add_parser("check-board", help="диагностика фото дошки")
    sp.add_argument("image")
    sp.add_argument("--board", default="a4")
    sp.add_argument("--calibration", default=None)
    sp.set_defaults(fn=cmd_check_board)

    sp = sub.add_parser("print-board", help="печатные PNG/SVG дошки")
    sp.add_argument("--board", default="a4")
    sp.add_argument("--png", default="board.png")
    sp.add_argument("--svg", default="board.svg")
    sp.set_defaults(fn=cmd_print_board)

    sp = sub.add_parser("devices", help="список устройств Orbbec")
    sp.set_defaults(fn=cmd_devices)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
