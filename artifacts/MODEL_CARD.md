# pose_regressor.experimental.onnx

**Research artifact only. Failed the synthetic improvement gates. Do not deploy
as an Astra scanner tracker. Not a drop-in replacement for the upstream model.**

## Intended task and provenance

Relative rigid registration of adjacent point-cloud frames. A PointNet encoder
with a shared two-direction pose head. 1,325,319 trainable parameters before
inference-only BatchNorm folding. Initial encoder weights originate from
Klischa/astra2; the regression head was reinitialized and all parameters trained.
See [attribution/license](../THIRD_PARTY.md).

- Date: 2026-09-22.
- License of these derivative weights: **CC BY-NC-SA 4.0**.
- No real scans or third-party dataset used.
- Export: ONNX opset 18, float32, self-contained.
- SHA256: `d424d62cf80824fb669daf53bb97f3981976dca19ddee5bc8740598c8ff29369`.
- Selected checkpoint SHA256: `615c132f25d3ea7f256cca94a8d5334173b385975828060b531132c67e0609be`.

## Input/output contract

Inputs: `source` float32 `[B,3,Ns]`, `target` float32 `[B,3,Nt]`.
Use **raw uncentered XYZ in metres**, finite nonempty clouds, same batch size.
Reject unusable/degenerate geometry upstream. There is no confidence output.

Output: `pose` float32 `[B,7]`, ordered **`tx,ty,tz,qx,qy,qz,qw`**.
The quaternion is normalized. Direction: `p_target = R(q) @ p_source + t`.
Centering, shared RMS scale, and centroid/scale compensation are inside the graph.
**Do not center or independently normalize inputs externally.**

Identical clouds yield identity; swapping inputs yields the inverse pose, within
floating-point error. These are architecture constraints, not evidence of learned accuracy.

## Training and evaluation

- 1200 CPU steps, batch 8, 256 points/cloud; selected step **600**, using validation loss.
- Independent scene namespaces: 128 training scenes, 24 validation scenes,
  32 possible test scenes. Test contains 192 pairs spanning 26 unique test scenes,
  plus 64 identity pairs.
- Procedural composites of boxes/ellipsoids; motion up to 12 degrees and ±25 mm
  per axis; clean pairs, resampling, approximate partial visibility, 0.7 mm noise,
  and approximately 1% outliers in the partial mode.
- No calibrated depth noise, actual occlusion rendering, real calibration,
  temporal sequences, trajectory-drift testing, or real-world validation.

| Method | Mean rotation error | Mean translation error |
|---|---:|---:|
| Centroid baseline | 5.487 deg | 52.11 mm |
| Network | 5.468 deg | 50.82 mm |
| Centroid + ICP | 1.193 deg | 10.58 mm |
| Network + ICP | 1.135 deg | 10.34 mm |

**`synthetic_gates_passed=false`; `deployment_approved=false`.**
The network did not achieve the required 10% improvement over the elementary
centroid baseline in either rotation or translation. Better results against
upstream ONNX under the unverified C++ interpretation do not establish model quality.

Numerical PyTorch/ONNX parity was checked for B=1/2 and Ns/Nt=32..2048,
including unequal Ns/Nt. Max absolute discrepancy approximately 1.05e-7.
Accuracy was measured at **256 points only**; shape flexibility does not establish
accuracy at other resolutions, scales, motions, sensors, or scenes.

Full [Russian report](../reports/RESULTS.md), [machine-readable metrics](../reports/cpu_synthetic_symmetric_v1.test.json),
[configuration](../configs/cpu_synthetic_symmetric.json), and [integration contract](../docs/ASTRA2_INTEGRATION.md).
