"""On-the-fly procedural pairs. These are NOT recorded Astra scans.

Splits use different scene seeds, not different views of the same object.
Partial visibility is approximated with half-space crops, not a depth renderer.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation

SPLITS = {"train": 101, "validation": 202, "test": 303}
MODES = ("clean", "resampled", "partial")


@dataclass(frozen=True)
class DataConfig:
    seed: int = 20260922
    points: int = 256
    train_scenes: int = 128
    validation_scenes: int = 24
    test_scenes: int = 32
    max_rotation_deg: float = 12.0
    max_translation_m: float = .025
    noise_std_m: float = .0007
    partial_outlier_fraction: float = .01
    identity_probability: float = .15

    def __post_init__(self):
        if self.points < 16 or min(self.train_scenes, self.validation_scenes, self.test_scenes) < 1:
            raise ValueError("Need at least 16 points and nonempty scene splits")
        if not 0 <= self.identity_probability <= 1 or not 0 <= self.partial_outlier_fraction < .2:
            raise ValueError("Invalid probability/outlier rate")
        if not 0 < self.max_rotation_deg < 180 or self.max_translation_m < 0 or self.noise_std_m < 0:
            raise ValueError("Invalid motion/noise range")


@dataclass
class Pair:
    source: np.ndarray
    target: np.ndarray
    pose: np.ndarray
    scene: str
    mode: str


def _surface(rng, count, kind):
    if kind == 0:
        x = rng.uniform(-1, 1, (count, 3))
        axis = rng.integers(0, 3, count)
        x[np.arange(count), axis] = rng.choice([-1., 1.], count)
    else:
        x = rng.normal(size=(count, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


class SyntheticPairs:
    def __init__(self, config: DataConfig):
        self.config = config

    @lru_cache(maxsize=256)
    def scene(self, split: str, index: int):
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, SPLITS[split], index, 971]))
        count = 2048
        components = []
        parts = int(rng.integers(2, 5))
        for part in range(parts):
            n = count//parts + (part < count % parts)
            p = _surface(rng, n, int(rng.integers(0, 2)))
            p *= rng.uniform(.025, .10, 3)
            p = p @ Rotation.random(random_state=rng).as_matrix().T
            p += rng.uniform(-.065, .065, 3)
            components.append(p)
        cloud = np.concatenate(components)
        cloud -= cloud.mean(0)
        cloud = cloud @ Rotation.random(random_state=rng).as_matrix().T
        cloud = cloud.astype(np.float32)
        cloud.setflags(write=False)
        return cloud

    def pair(self, split: str, index: int, mode: str | None = None, identity: bool | None = None) -> Pair:
        cfg = self.config
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, SPLITS[split], index, 3181]))
        scene_count = getattr(cfg, f"{split}_scenes")
        scene_index = int(rng.integers(0, scene_count))
        pool = self.scene(split, scene_index)
        if mode is None:
            mode = str(rng.choice(MODES, p=[.55, .25, .20]))
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if identity is None:
            identity = split == "train" and rng.random() < cfg.identity_probability
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = 0 if identity else np.deg2rad(rng.uniform(0, cfg.max_rotation_deg))
        rotation = Rotation.from_rotvec(axis*angle)
        r = rotation.as_matrix().astype(np.float32)
        t = np.zeros(3, dtype=np.float32) if identity else rng.uniform(-cfg.max_translation_m, cfg.max_translation_m, 3).astype(np.float32)
        center = np.array([rng.uniform(-.12, .12), rng.uniform(-.12, .12), rng.uniform(.45, .9)], dtype=np.float32)
        source_pool, target_pool = pool, pool
        if mode == "partial":
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            other = normal + rng.normal(0, .15, 3)
            other /= np.linalg.norm(other)
            # Similar, but not identical, retained regions: approximate adjacent views.
            cutoff = rng.uniform(.10, .35)
            projection_s, projection_t = pool @ normal, pool @ other
            source_pool = pool[projection_s >= np.quantile(projection_s, cutoff)]
            target_pool = pool[projection_t >= np.quantile(projection_t, np.clip(cutoff + rng.uniform(-.07, .07), 0, .45))]
        a = source_pool[rng.choice(len(source_pool), cfg.points, replace=cfg.points > len(source_pool))].copy()
        if mode == "clean":
            b = a.copy()
        else:
            b = target_pool[rng.choice(len(target_pool), cfg.points, replace=cfg.points > len(target_pool))].copy()
        a += center
        b = (b + center) @ r.T + t
        if mode != "clean":
            a += rng.normal(0, cfg.noise_std_m, a.shape).astype(np.float32)
            b += rng.normal(0, cfg.noise_std_m, b.shape).astype(np.float32)
        if mode == "partial":
            for points in (a, b):
                n = int(cfg.points * cfg.partial_outlier_fraction)
                if n:
                    indices = rng.choice(cfg.points, n, replace=False)
                    points[indices] = rng.uniform(points.min(0), points.max(0), (n, 3))
        # Never expose point correspondences, even for clean pairs.
        a, b = a[rng.permutation(cfg.points)], b[rng.permutation(cfg.points)]
        pose = np.concatenate([t, rotation.as_quat()]).astype(np.float32)
        return Pair(a.T.copy(), b.T.copy(), pose, f"{split}:{scene_index}", mode)

    def batch(self, split, indices, **kwargs):
        pairs = [self.pair(split, int(i), **kwargs) for i in indices]
        return tuple(np.stack([getattr(p, field) for p in pairs]) for field in ("source", "target", "pose"))

    def description(self):
        return {"kind": "procedural synthetic only", "config": asdict(self.config),
                "split_namespace": SPLITS, "generator_version": 1,
                "real_scans_used": False, "occlusion": "approximate half-space crops"}
