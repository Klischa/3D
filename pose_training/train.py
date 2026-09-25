"""Bounded, resumable CPU/GPU training; the test split is never used here."""
import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import platform
import time
import numpy as np

# Required by deterministic CUDA GEMM; no effect on the tested CPU path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from .data import DataConfig, SyntheticPairs
from .geometry import quaternion_matrix, quaternion_loss, summarize_errors
from .model import BASE_SHA256, BASE_URL, CONTRACT, RawCloudPoseRegressor, load_original_core, load_checkpoint


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    steps: int = 1200
    batch_size: int = 8
    learning_rate: float = .001
    encoder_lr_multiplier: float = .25
    weight_decay: float = .0001
    evaluate_every: int = 100
    validation_pairs: int = 96
    rotation_loss_scale_deg: float = 5.0
    translation_loss_scale_m: float = .01
    threads: int = 1
    device: str = "cpu"
    symmetric_head: bool = False

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.evaluate_every, self.validation_pairs, self.threads) < 1:
            raise ValueError("Training counts must be positive")
        if min(self.learning_rate, self.encoder_lr_multiplier, self.rotation_loss_scale_deg, self.translation_loss_scale_m) <= 0:
            raise ValueError("Learning rates and loss scales must be positive")
        if not isinstance(self.symmetric_head, bool):
            raise ValueError("symmetric_head must be a boolean")
        if self.weight_decay < 0 or self.device not in ("cpu", "cuda"):
            raise ValueError("Invalid weight decay or device")


def read_config(path):
    values = json.loads(Path(path).read_text())
    values["data"] = DataConfig(**values.get("data", {}))
    return TrainConfig(**values)


def loss_for(model, source, target, truth, config):
    pose, residual, cs, ct, scale = model.forward_details(source, target)
    true_rotation = quaternion_matrix(truth[:, 3:])
    # Supervise the CENTERED residual, not the original translation after discarding centroids.
    true_residual = (torch.bmm(true_rotation, cs).squeeze(2) + truth[:, :3] - ct.squeeze(2))/scale.squeeze(2)
    residual_error_m = (residual - true_residual)*scale.squeeze(2)
    translation_loss = (residual_error_m/config.translation_loss_scale_m).square().mean()
    rotation_loss = quaternion_loss(pose[:, 3:], truth[:, 3:]).mean()/math.radians(config.rotation_loss_scale_deg)**2
    return rotation_loss + translation_loss, pose


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(config, base_onnx, out_dir, resume=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not resume and any((out_dir/name).exists() for name in ("best.pt", "last.pt", "history.jsonl")):
        raise ValueError("Run directory already contains training state; use --resume or a new directory")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.data.seed)
    torch.use_deterministic_algorithms(True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; explicitly choose CPU instead")
    device = torch.device(config.device)
    if resume:
        model, previous = load_checkpoint(resume)
        previous_config = dict(previous["config"])
        previous_config.setdefault("symmetric_head", False)
        if previous_config != asdict(config):
            raise ValueError("Resume configuration must match exactly (including total steps and data seed)")
    else:
        if base_onnx is None:
            raise ValueError("--base-onnx is required for a new run")
        core = load_original_core(base_onnx)
        core.reset_pose_head()
        model = RawCloudPoseRegressor(core, symmetric_head=config.symmetric_head)
        previous = None
    model.to(device)
    head = [p for name, p in model.core.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": model.core.encoder.parameters(), "lr": config.learning_rate*config.encoder_lr_multiplier},
        {"params": head, "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .1 + .9*(1 + math.cos(math.pi*min(step, config.steps)/config.steps))/2)
    start_step, best_score = 0, float("inf")
    if previous:
        optimizer.load_state_dict(previous["optimizer_state"])
        schedule.load_state_dict(previous["scheduler_state"])
        torch.set_rng_state(previous["torch_rng_state"])
        if config.device == "cuda" and "cuda_rng_state" in previous:
            torch.cuda.set_rng_state_all(previous["cuda_rng_state"])
        start_step, best_score = previous["step"], previous["best_validation_loss"]
    factory = SyntheticPairs(config.data)
    validation = factory.batch("validation", range(config.validation_pairs))
    manifest = {
        "status": "experimental_synthetic_only",
        "real_world_validated": False,
        "base_url": BASE_URL, "base_sha256": BASE_SHA256,
        "initialization": "original encoder weights; reinitialized 2048->512->256->7 head",
        "selection": "minimum validation loss; independent test split untouched during training",
        "config": asdict(config), "data": factory.description(), "contract": CONTRACT,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                        "device": str(device), "threads": config.threads},
    }
    (out_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()

    def validate():
        model.eval()
        losses, poses = [], []
        with torch.inference_mode():
            for i in range(0, config.validation_pairs, config.batch_size):
                arrays = [torch.from_numpy(v[i:i+config.batch_size]).to(device) for v in validation]
                loss, pose = loss_for(model, *arrays, config)
                losses.extend([float(loss)] * len(pose))
                poses.append(pose.cpu().numpy())
        return float(np.mean(losses)), summarize_errors(np.concatenate(poses), validation[2])

    def save(path, step, score):
        value = {
            "format_version": 1, "contract": CONTRACT, "config": asdict(config),
            "symmetric_head": config.symmetric_head,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "scheduler_state": schedule.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "step": step, "validation_loss": score, "best_validation_loss": best_score,
            "manifest": manifest,
        }
        if config.device == "cuda":
            value["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        atomic_checkpoint(path, value)

    with (out_dir/"history.jsonl").open("a", encoding="utf-8") as history:
        def record(step, score, metrics, train_loss=None):
            row = {"step": step, "validation_loss": score, "validation": metrics,
                   "elapsed_seconds_this_process": time.perf_counter()-started}
            if train_loss is not None:
                row["training_loss_last_batch"] = train_loss
            history.write(json.dumps(row, allow_nan=False) + "\n")
            history.flush()
            print(f"step {step:4d}/{config.steps}: val_loss={score:.4f}, rot={metrics['rotation_mean_deg']:.3f} deg, "
                  f"translation={metrics['translation_mean_mm']:.2f} mm, elapsed={row['elapsed_seconds_this_process']:.1f}s", flush=True)
        if not previous:
            best_score, metrics = validate()
            save(out_dir/"best.pt", 0, best_score)
            record(0, best_score, metrics)
        for step in range(start_step, config.steps):
            model.train()
            arrays = factory.batch("train", range(step*config.batch_size, (step+1)*config.batch_size))
            arrays = [torch.from_numpy(a).to(device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(model, *arrays, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            if (step+1) % config.evaluate_every == 0 or step+1 == config.steps:
                score, metrics = validate()
                if score < best_score:
                    best_score = score
                    save(out_dir/"best.pt", step+1, score)
                save(out_dir/"last.pt", step+1, score)
                record(step+1, score, metrics, float(loss.detach()))
    elapsed = time.perf_counter()-started
    manifest["elapsed_seconds_this_process"] = elapsed
    manifest["completed_steps"] = config.steps
    manifest["best_validation_loss"] = best_score
    (out_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Finished. Best checkpoint: {out_dir/'best.pt'}. NOT validated on real scans.", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-onnx")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(read_config(args.config), args.base_onnx, args.out, args.resume)


if __name__ == "__main__":
    main()
