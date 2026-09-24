from dataclasses import asdict
import pytest

try:
    import torch
    from pose_training.model import CONTRACT, RawCloudPoseRegressor
    from pose_training.train import TrainConfig
    from pose_training.data import DataConfig

    torch.set_num_threads(1)
    _TORCH = True
except ModuleNotFoundError:  # torch тяжёлый (906 МБ); gt_collector не требует
    _TORCH = False


@pytest.fixture
def checkpoint(tmp_path):
    if not _TORCH:
        pytest.skip("torch не установлен в этом окружении")
    torch.manual_seed(9)
    model = RawCloudPoseRegressor(symmetric_head=True)
    model.core.reset_pose_head()
    config = TrainConfig(data=DataConfig(points=32), steps=2, batch_size=2, validation_pairs=4, symmetric_head=True)
    path = tmp_path / "model.pt"
    torch.save({"format_version": 1, "contract": CONTRACT, "symmetric_head": True,
                "model_state": model.state_dict(), "config": asdict(config), "step": 0,
                "manifest": {"initialization": "test fixture"}}, path)
    return path
