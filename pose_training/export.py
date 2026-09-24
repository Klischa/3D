"""Export an explicitly experimental ONNX with a documented coordinate contract."""
import argparse
import hashlib
import json
import os
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from .model import CONTRACT, BASE_SHA256, load_checkpoint


def export(checkpoint, output, evaluation=None):
    torch.set_num_threads(1)
    model, state = load_checkpoint(checkpoint)
    checkpoint_sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    quality = {"synthetic_gates_passed": False, "deployment_approved": False, "reason": "Not evaluated"}
    if evaluation:
        report = json.loads(Path(evaluation).read_text())
        if report["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Evaluation belongs to a different checkpoint")
        quality = report["quality_gates"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = output
    # Never leave an unverified export at the requested final filename.
    with TemporaryDirectory(prefix=".onnx-export-", dir=output.parent) as temporary:
        output = Path(temporary)/destination.name
        torch.manual_seed(1729)
        source = torch.randn(1, 3, 64)*.1
        source[:, 2] += .6
        target = source + .01
        with torch.inference_mode():
            torch.onnx.export(
                model, (source, target), str(output), input_names=["source", "target"], output_names=["pose"],
                dynamic_axes={"source": {0: "batch_size", 2: "num_source_points"},
                              "target": {0: "batch_size", 2: "num_target_points"}, "pose": {0: "batch_size"}},
                opset_version=18, do_constant_folding=True, dynamo=False,
            )
        graph = onnx.load(output)
        graph.model_version = 1
        graph.doc_string = "EXPERIMENTAL: procedural synthetic pretraining only. Not validated on real Astra scans. Raw XYZ metres in; source-to-target [txyz,qxyzw] out."
        metadata = {
            "pose.contract": json.dumps(CONTRACT),
            "training.status": "experimental_synthetic_only",
            "training.real_world_validated": "false",
            "training.checkpoint_sha256": checkpoint_sha,
            "training.selected_step": str(state["step"]),
            "training.config": json.dumps(state["config"]),
            "training.initialization": state["manifest"]["initialization"],
            "architecture.symmetric_head": json.dumps(state.get("symmetric_head", False)),
            "training.quality_gates": json.dumps(quality),
            "provenance.original_sha256": BASE_SHA256,
            "license": "CC-BY-NC-SA-4.0; derived from Klischa/astra2; see THIRD_PARTY.md",
        }
        onnx.helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, output)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(1729)
        checks = []
        for batch, ns, nt in [(1, 32, 32), (2, 64, 48), (1, 256, 256), (1, 2048, 1024)]:
            a = rng.normal(0, .1, (batch, 3, ns)).astype(np.float32)
            b = rng.normal(0, .1, (batch, 3, nt)).astype(np.float32)
            a[:, 2] += .7
            b[:, 2] += .72
            with torch.inference_mode():
                expected = model(torch.from_numpy(a), torch.from_numpy(b)).numpy()
            actual = session.run(["pose"], {"source": a, "target": b})[0]
            if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=2e-5, rtol=2e-4):
                raise RuntimeError("PyTorch/ONNX parity check failed")
            if not np.allclose(np.linalg.norm(actual[:, 3:], axis=1), 1, atol=1e-6):
                raise RuntimeError("ONNX quaternion is not normalized")
            checks.append({"batch": batch, "source_points": ns, "target_points": nt,
                           "max_abs_difference": float(np.abs(actual-expected).max())})
        result = {"onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "checkpoint_sha256": checkpoint_sha, "checks": checks,
                  "opset": 18, "all_passed": True, "deployment_approved": False}
        os.replace(output, destination)
        destination.with_suffix(".parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evaluation")
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.evaluation)


if __name__ == "__main__":
    main()
