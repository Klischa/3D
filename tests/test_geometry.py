import numpy as np
import pytest
torch = pytest.importorskip("torch")  # тяжёлая зависимость: пропуск без torch
from scipy.spatial.transform import Rotation
from pose_training.geometry import normalize_quaternion, quaternion_matrix, quaternion_loss, pose_errors, rigid_fit, matrix_pose


def test_xyzw_matches_scipy():
    q = torch.tensor([[.1, .2, -.3, .9], [0., 0., 0., 1.]])
    r = quaternion_matrix(q).numpy()
    np.testing.assert_allclose(r, Rotation.from_quat(q.numpy()).as_matrix(), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-6)


def test_zero_quaternion_has_finite_identity_fallback():
    q = normalize_quaternion(torch.zeros(2, 4))
    assert torch.equal(q, torch.tensor([[0., 0., 0., 1.]]).expand(2, 4))


def test_quaternion_loss_sign_invariant_and_differentiable():
    p = torch.tensor([[.02, 0, 0, 1.]], requires_grad=True)
    target = torch.tensor([[0., .1, 0, 1.]])
    assert torch.allclose(quaternion_loss(p, target), quaternion_loss(p, -target))
    loss = quaternion_loss(p, target).sum()
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert quaternion_loss(target, -target).item() == 0


def test_pose_errors_sign_and_units():
    a = np.array([[0, 0, 0, 0, 0, 0, 1.]])
    b = np.array([[.01, 0, 0, 0, 0, 0, -1.]])
    angle, shift = pose_errors(a, b)
    np.testing.assert_allclose(angle, 0)
    np.testing.assert_allclose(shift, .01)


@pytest.mark.parametrize("bad", [np.zeros((1, 7)), np.full((1, 7), np.nan)])
def test_bad_poses_are_not_silently_scored(bad):
    with pytest.raises(ValueError):
        pose_errors(bad, np.array([[0, 0, 0, 0, 0, 0, 1.]]))


def test_kabsch_direction_and_reflection():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(100, 3))
    r = Rotation.from_euler('xyz', [15, -6, 8], degrees=True).as_matrix()
    t = np.array([.04, -.02, .01])
    actual_r, actual_t = rigid_fit(a, a @ r.T+t)
    np.testing.assert_allclose(actual_r, r, atol=1e-10)
    np.testing.assert_allclose(actual_t, t, atol=1e-10)
    reflected = a.copy(); reflected[:, 0] *= -1
    rr, _ = rigid_fit(a, reflected)
    assert np.linalg.det(rr) > .99999
    assert matrix_pose(actual_r, actual_t).shape == (7,)
