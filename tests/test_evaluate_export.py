import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
torch = pytest.importorskip("torch")  # тяжёлая зависимость: пропуск без torch
from pose_training.data import DataConfig, SyntheticPairs
from pose_training.geometry import pose_errors
from pose_training.evaluate import centroid_pose, icp, quality_gates
from pose_training.export import export
from pose_training.train import TrainConfig, read_config, train


def test_centroid_translation_baseline_and_icp():
    pair = SyntheticPairs(DataConfig(points=128, max_rotation_deg=2)).pair('validation', 31, mode='clean', identity=False)
    initial = centroid_pose(pair.source[None], pair.target[None])[0]
    refined, iterations = icp(pair.source, pair.target, initial)
    angle, shift = pose_errors(refined[None], pair.pose[None])
    assert angle[0] < .01 and shift[0] < 1e-5
    assert 1 <= iterations <= 25


def test_gates_never_approve_real_world_deployment():
    good = {'rotation_mean_deg': 1., 'translation_mean_mm': 2., 'success_at_2deg_10mm': 1.}
    base = {'rotation_mean_deg': 3., 'translation_mean_mm': 10., 'success_at_2deg_10mm': 1.}
    identity = {'rotation_mean_deg': 0., 'translation_mean_mm': 0.}
    all_groups = {'all': {'neural': good, 'centroid': base, 'neural_icp': good, 'centroid_icp': base}}
    result = quality_gates(all_groups, identity)
    assert result['synthetic_gates_passed']
    assert not result['deployment_approved']
    all_groups['all']['neural'] = base
    assert not quality_gates(all_groups, identity)['synthetic_gates_passed']


def test_export_roundtrip_dynamic_shapes_and_metadata(checkpoint, tmp_path):
    path = tmp_path/'experimental.onnx'
    result = export(checkpoint, path)
    assert result['all_passed'] and not result['deployment_approved']
    assert len(result['checks']) == 4
    import onnx
    graph = onnx.load(path)
    meta = {x.key: x.value for x in graph.metadata_props}
    assert meta['training.real_world_validated'] == 'false'
    assert meta['architecture.symmetric_head'] == 'true'
    assert json.loads(meta['pose.contract'])['external_centering_allowed'] is False
    assert path.with_suffix('.parity.json').exists()


def test_export_refuses_unrelated_evaluation(checkpoint, tmp_path):
    report = tmp_path/'report.json'
    report.write_text(json.dumps({'checkpoint_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='different checkpoint'):
        export(checkpoint, tmp_path/'model.onnx', report)


def test_invalid_training_counts():
    with pytest.raises(ValueError):
        TrainConfig(steps=0)


def test_training_and_exact_resume_do_not_touch_test_split(tmp_path, monkeypatch):
    from pathlib import Path
    from pose_training import train as training
    from pose_training.model import load_checkpoint
    import torch
    base = Path('weights/original_pose_regressor.onnx')
    if not base.exists():
        pytest.skip('Pinned base needed for training smoke/resume test')
    cfg = TrainConfig(data=DataConfig(points=32, train_scenes=2, validation_scenes=2),
                      steps=2, batch_size=2, evaluate_every=1, validation_pairs=2, symmetric_head=True)
    splits = []
    original_pair = SyntheticPairs.pair
    def pair(self, split, *args, **kwargs):
        splits.append(split)
        return original_pair(self, split, *args, **kwargs)
    monkeypatch.setattr(SyntheticPairs, 'pair', pair)
    train(cfg, base, tmp_path/'full')
    full, _ = load_checkpoint(tmp_path/'full'/'last.pt')
    original_save = training.atomic_checkpoint
    class SimulatedInterruption(Exception):
        pass
    def interrupt(path, value):
        original_save(path, value)
        if path.name == 'last.pt' and value['step'] == 1:
            raise SimulatedInterruption()
    monkeypatch.setattr(training, 'atomic_checkpoint', interrupt)
    with pytest.raises(SimulatedInterruption):
        train(cfg, base, tmp_path/'resumed')
    monkeypatch.setattr(training, 'atomic_checkpoint', original_save)
    train(cfg, None, tmp_path/'resumed', resume=tmp_path/'resumed'/'last.pt')
    resumed, _ = load_checkpoint(tmp_path/'resumed'/'last.pt')
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    assert set(splits) == {'train', 'validation'}
    with pytest.raises(ValueError, match='already contains'):
        train(cfg, base, tmp_path/'full')


def test_failed_export_does_not_replace_existing_artifact(checkpoint, tmp_path, monkeypatch):
    from pose_training import export as exporter
    class BrokenRuntime:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            return [np.full((1, 7), np.nan, dtype=np.float32)]
    path = tmp_path/'existing.onnx'
    path.write_bytes(b'previous verified artifact')
    monkeypatch.setattr(exporter.ort, 'InferenceSession', BrokenRuntime)
    with pytest.raises(RuntimeError, match='parity'):
        exporter.export(checkpoint, path)
    assert path.read_bytes() == b'previous verified artifact'
    assert not list(tmp_path.glob('.onnx-export-*'))
