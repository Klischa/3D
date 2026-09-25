import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pose_training.data import DataConfig, SyntheticPairs, MODES


def test_reproducible_pairs_and_scene_disjointness():
    a = SyntheticPairs(DataConfig(points=64))
    b = SyntheticPairs(DataConfig(points=64))
    pa, pb = a.pair('train', 91), b.pair('train', 91)
    assert pa.scene == pb.scene
    np.testing.assert_array_equal(pa.source, pb.source)
    np.testing.assert_array_equal(pa.target, pb.target)
    assert a.pair('train', 91).scene != a.pair('validation', 91).scene
    assert not np.array_equal(a.scene('train', 0), a.scene('validation', 0))
    assert not np.array_equal(a.scene('validation', 0), a.scene('test', 0))


def test_clean_ground_truth_is_exact_despite_point_permutation():
    pair = SyntheticPairs(DataConfig(points=128)).pair('validation', 6, mode='clean', identity=False)
    r = Rotation.from_quat(pair.pose[3:]).as_matrix()
    moved = pair.source.T @ r.T + pair.pose[:3]
    distance, _ = cKDTree(pair.target.T).query(moved)
    assert distance.max() < 5e-7


@pytest.mark.parametrize('mode', MODES)
def test_pair_shapes_finiteness_and_labels(mode):
    pair = SyntheticPairs(DataConfig(points=64)).pair('train', 12, mode=mode)
    assert pair.source.shape == pair.target.shape == (3, 64)
    assert pair.pose.shape == (7,)
    assert pair.source.dtype == np.float32
    assert np.isfinite(pair.source).all() and np.isfinite(pair.target).all()
    assert abs(np.linalg.norm(pair.pose[3:])-1) < 1e-6


def test_cached_scene_cannot_be_mutated_by_pair_generation():
    factory = SyntheticPairs(DataConfig(points=64))
    before = factory.scene('train', 0).copy()
    for i in range(20):
        factory.pair('train', i)
    np.testing.assert_array_equal(before, factory.scene('train', 0))
    assert not factory.scene('train', 0).flags.writeable


@pytest.mark.parametrize('kwargs', [{'points': 0}, {'train_scenes': 0}, {'partial_outlier_fraction': .5}, {'noise_std_m': -.1}])
def test_invalid_data_config(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)
