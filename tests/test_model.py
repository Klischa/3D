from pathlib import Path
import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")  # тяжёлая зависимость: пропуск без onnxruntime
torch = pytest.importorskip("torch")  # тяжёлая зависимость: пропуск без torch
from scipy.spatial.transform import Rotation
from pose_training.geometry import quaternion_matrix
from pose_training.model import RawCloudPoseRegressor, PoseRegressor, load_original_core, load_checkpoint
from pose_training.train import TrainConfig, loss_for
from pose_training.data import DataConfig, SyntheticPairs


class KnownCenteredTransform(torch.nn.Module):
    def __init__(self, q):
        super().__init__()
        self.register_buffer('q', torch.as_tensor(q, dtype=torch.float32))

    def forward(self, a, b):
        return torch.cat([torch.zeros((len(a), 3), dtype=a.dtype), self.q.expand(len(a), -1)], dim=1)


def test_centroids_and_scale_are_restored():
    torch.manual_seed(11)
    source = torch.randn(2, 3, 128)*.07
    source[:, 2] += .8
    q = Rotation.from_euler('xyz', [10, -4, 7], degrees=True).as_quat()
    r = quaternion_matrix(torch.tensor(q, dtype=torch.float32)).expand(2, 3, 3)
    t = torch.tensor([[.03, -.02, .01], [-.01, .05, .02]])
    target = torch.bmm(r, source) + t.unsqueeze(2)
    model = RawCloudPoseRegressor(KnownCenteredTransform(q)).eval()
    pose = model(source, target)
    assert torch.allclose(pose[:, :3], t, atol=3e-7)
    assert torch.allclose(quaternion_matrix(pose[:, 3:]), r, atol=2e-7)


def test_translation_information_not_discarded():
    source = torch.randn(1, 3, 64)*.1 + .5
    shift = torch.tensor([.04, -.02, .01]).reshape(1, 3, 1)
    model = RawCloudPoseRegressor(KnownCenteredTransform([0, 0, 0, 1])).eval()
    assert torch.allclose(model(source, source+shift)[:, :3], shift.squeeze(2), atol=3e-7)


def test_parameter_count_matches_original():
    assert sum(p.numel() for p in PoseRegressor().parameters()) == 1325319


def test_symmetric_head_identity_inverse_and_permutation():
    torch.manual_seed(7)
    # Arbitrary nonzero head, not the trivially zero-initialized training head.
    model = RawCloudPoseRegressor(symmetric_head=True).eval()
    a = torch.randn(2, 3, 64)*.1; a[:, 2] += .7
    b = torch.randn(2, 3, 48)*.1; b[:, 2] += .6
    with torch.inference_mode():
        same = model(a, a)
        f, back = model(a, b), model(b, a)
        shuffled = model(a[:, :, torch.randperm(64)], b[:, :, torch.randperm(48)])
        duplicated = model(a.repeat_interleave(2, dim=2), b.repeat_interleave(2, dim=2))
    assert torch.allclose(same[:, :3], torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(quaternion_matrix(same[:, 3:]), torch.eye(3), atol=1e-6)
    rf, rb = quaternion_matrix(f[:, 3:]), quaternion_matrix(back[:, 3:])
    assert torch.allclose(rb @ rf, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.bmm(rb, f[:, :3, None]).squeeze(2)+back[:, :3], torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(f, shuffled, atol=2e-5)
    assert torch.allclose(f, duplicated, atol=2e-5)


def test_training_loss_and_gradients_finite():
    cfg = TrainConfig(data=DataConfig(points=32), symmetric_head=True)
    factory = SyntheticPairs(cfg.data)
    a, b, truth = [torch.from_numpy(v) for v in factory.batch('train', range(4), mode='clean', identity=False)]
    model = RawCloudPoseRegressor(symmetric_head=True)
    model.core.reset_pose_head()
    loss, _ = loss_for(model, a, b, truth, cfg)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.core.fc3.weight.grad.abs().sum() > 0


def test_original_recovery_parity():
    path = Path('weights/original_pose_regressor.onnx')
    if not path.exists():
        pytest.skip('Run python -m pose_training.fetch_base for original-model parity test')
    core = load_original_core(path).eval()
    rng = np.random.default_rng(23)
    a, b = [rng.normal(size=(2, 3, 64)).astype('float32') for _ in range(2)]
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    expected = session.run(None, {'source': a, 'target': b})[0]
    with torch.inference_mode():
        actual = core(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_checksum_rejection(tmp_path):
    path = tmp_path/'bad.onnx'; path.write_bytes(b'not the pinned model')
    with pytest.raises(ValueError, match='SHA256'):
        load_original_core(path)


def test_safe_checkpoint_restores_architecture(checkpoint):
    model, state = load_checkpoint(checkpoint)
    assert model.symmetric_head and state['step'] == 0
