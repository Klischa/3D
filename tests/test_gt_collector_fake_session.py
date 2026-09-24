"""Интеграционный тест: короткая фейковая сессия через полный пайплайн."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gt_collector.config import BoardSpec, SessionConfig
from gt_collector.fake import FakeBackend, FakeConfig
from gt_collector.session import Session
from gt_collector.pairs import extract_pairs, save_pairs
from gt_collector.report import save_report
from gt_collector.cloud import load_cloud


@pytest.fixture(scope="module")
def session_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("gt_session")
    board = BoardSpec.from_preset("a4")
    fcfg = FakeConfig(duration_s=3.0, fps=30, trajectory="orbit", seed=42,
                      board_preset="a4")
    cfg = SessionConfig(board=board, duration_s=3.0, warmup_frames=5, fps=30,
                        save_frames_every=3)
    be = FakeBackend(fcfg, board)
    sess = Session(out, cfg, be)
    manifest = sess.run()
    save_pairs(out)
    save_report(out)
    return out, manifest


def test_session_files(session_out):
    out, manifest = session_out
    assert (out / "manifest.json").exists()
    assert (out / "poses.csv").exists()
    assert (out / "pairs.csv").exists()
    assert (out / "report.json").exists()
    # 90 кадров − 5 warmup = 85 в poses.csv
    assert manifest["frames_written"] == 3 * 30 - 5
    # позы пишутся каждый кадр, артефакты — каждый 3-й (idx 6..87, idx%3==0)
    jpgs = list((out / "frames").glob("*.jpg"))
    clouds = list((out / "frames").glob("*.cloud"))
    assert len(jpgs) == 28
    assert len(clouds) == 28


def test_session_poses_valid(session_out):
    out, manifest = session_out
    rows = list(csv.DictReader(open(out / "poses.csv")))
    valid = [r for r in rows if not (set(r["flags"].split("|"))
                                     & {"no_board", "poor_fit", "fast", "stale", "thin"})]
    assert len(valid) / len(rows) > 0.9
    # координаты камеры в разумной области (орбита 1 м)
    ts = np.array([np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]) for r in valid])
    assert np.all(np.linalg.norm(ts, axis=1) > 0.5)
    assert np.all(np.linalg.norm(ts, axis=1) < 2.0)
    # резидуалы маленькие (мм)
    resid = np.array([float(r["mean_residual_mm"]) for r in valid])
    assert resid.max() < 5.0


def test_session_pairs(session_out):
    out, _ = session_out
    pairs = extract_pairs(out)
    assert len(pairs) > 50
    for p in pairs[:5]:
        T = p["T"]
        # ортогональность вращения
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)
        assert p["step_mm"] < 200
        assert p["source_frame"] > p["target_frame"]


def test_session_report(session_out):
    out, _ = session_out
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["frames"] == 85
    assert rep["valid_ratio"] > 0.9
    assert rep["residual_mm"]["p95"] < 5.0


def test_session_clouds(session_out):
    out, _ = session_out
    clouds = sorted((out / "frames").glob("*.cloud"))
    c = load_cloud(clouds[0])
    assert c.n > 20000
    assert np.all(np.isfinite(c.xyz))
    assert 0.3 < c.xyz[:, 2].max() <= 4.0
