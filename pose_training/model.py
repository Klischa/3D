"""Original PointNet architecture, with an explicit raw-coordinate pose contract."""
from pathlib import Path
import hashlib
import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn
from .geometry import normalize_quaternion, quaternion_matrix

BASE_SHA256 = "78bdab58ef9ddad97930150b5b1409868fd81c66cd6baaab5cc3b2d42e9d845a"
BASE_URL = "https://raw.githubusercontent.com/Klischa/astra2/fc628b3c6f4d8f2e02554ae91833a732804ac2dd/models/pose_regressor.onnx"
CONTRACT = {
    "version": 1,
    "inputs": {"source": "float32[B,3,Ns]", "target": "float32[B,3,Nt]"},
    "coordinates": "raw, uncentered XYZ in metres; same batch size; finite, nonempty clouds",
    "output": "float32[B,7] = [tx,ty,tz,qx,qy,qz,qw]",
    "direction": "target = R(qxyzw) @ source + t",
    "quaternion": "unit xyzw; q and -q represent the same rotation",
    "preprocessing": "centering and shared RMS scaling INSIDE the ONNX graph",
    "translation": "centroids and scale restored INSIDE the ONNX graph",
    "external_centering_allowed": False,
    "confidence_output": False,
}


class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = nn.Conv1d(3, 64, 1), nn.Conv1d(64, 128, 1), nn.Conv1d(128, 1024, 1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)

    def forward(self, points):
        x = torch.relu(self.bn1(self.conv1(points)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.bn3(self.conv3(x)).max(dim=2).values


class PoseRegressor(nn.Module):
    """Numerically equivalent to the core network in the original ONNX."""
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.fc1, self.fc2, self.fc3 = nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 7)

    def regress_features(self, source_features, target_features):
        combined = torch.cat([source_features, target_features], dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def forward(self, source, target):
        return self.regress_features(self.encoder(source), self.encoder(target))

    def forward_symmetric(self, source, target):
        # Both head directions reuse the same two encoder evaluations.
        a, b = self.encoder(source), self.encoder(target)
        forward = self.regress_features(a, b)
        backward = self.regress_features(b, a)
        q = normalize_quaternion(torch.cat([
            forward[:, 3:6] - backward[:, 3:6],
            forward[:, 6:7] + backward[:, 6:7],
        ], dim=1))
        r = quaternion_matrix(q)
        residual = .5*(forward[:, :3] - torch.bmm(r, backward[:, :3].unsqueeze(2)).squeeze(2))
        # Swapping inputs gives (R^-1, -R^-1 t); identical inputs give I.
        return torch.cat([residual, q], dim=1)

    def reset_pose_head(self):
        # The old seven channels have no verified training-time semantics.
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        with torch.no_grad():
            self.fc3.bias[6] = 1.0


def load_original_core(path: str | Path, expected_sha: str = BASE_SHA256) -> PoseRegressor:
    path = Path(path)
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"Unexpected base model SHA256: {actual_sha}; expected {expected_sha}")
    model = onnx.load(path, load_external_data=False)
    if any(i.external_data for i in model.graph.initializer):
        raise ValueError("External weights are not supported for the pinned base")
    onnx.checker.check_model(model, full_check=True)
    weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    core = PoseRegressor()
    state = core.state_dict()
    for key, value in state.items():
        if key.endswith("num_batches_tracked"):
            state[key] = torch.zeros_like(value)
            continue
        if key not in weights or tuple(weights[key].shape) != tuple(value.shape):
            raise ValueError(f"Missing or incompatible tensor: {key}")
        if not np.isfinite(weights[key]).all():
            raise ValueError(f"Non-finite weight: {key}")
        state[key] = torch.from_numpy(weights[key].copy())
    core.load_state_dict(state, strict=True)
    return core


class RawCloudPoseRegressor(nn.Module):
    """Public inputs are always RAW metres, not independently centered clouds."""
    def __init__(self, core: PoseRegressor | None = None, symmetric_head: bool = False):
        super().__init__()
        self.core = core if core is not None else PoseRegressor()
        self.symmetric_head = symmetric_head

    @staticmethod
    def preprocess(source, target):
        cs, ct = source.mean(dim=2, keepdim=True), target.mean(dim=2, keepdim=True)
        xs, xt = source-cs, target-ct
        variance = .5 * (xs.square().sum(dim=1).mean(dim=1) + xt.square().sum(dim=1).mean(dim=1))
        scale = variance.clamp_min(1e-8).sqrt().reshape(-1, 1, 1)
        return xs/scale, xt/scale, cs, ct, scale

    def forward_details(self, source, target):
        xs, xt, cs, ct, scale = self.preprocess(source, target)
        raw = self.core.forward_symmetric(xs, xt) if self.symmetric_head else self.core(xs, xt)
        q = normalize_quaternion(raw[:, 3:])
        rotation = quaternion_matrix(q)
        # Required inverse of centering and shared normalization.
        t = ct.squeeze(2) - torch.bmm(rotation, cs).squeeze(2) + scale.squeeze(2)*raw[:, :3]
        pose = torch.cat([t, q], dim=1)
        return pose, raw[:, :3], cs, ct, scale

    def forward(self, source, target):
        return self.forward_details(source, target)[0]


def load_checkpoint(path: str | Path) -> tuple[RawCloudPoseRegressor, dict]:
    # Never allow arbitrary Python object unpickling from downloaded checkpoints.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("contract") != CONTRACT:
        raise ValueError("Incompatible checkpoint/coordinate contract")
    model = RawCloudPoseRegressor(symmetric_head=checkpoint.get("symmetric_head", False))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), checkpoint
