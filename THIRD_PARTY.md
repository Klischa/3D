# Provenance and licensing of the model weights

The initial encoder weights and recovered network architecture originate from:

- Author/repository: **Klischa**, <https://github.com/Klischa/astra2>
- Model: `models/pose_regressor.onnx`
- Immutable source: <https://github.com/Klischa/astra2/blob/fc628b3c6f4d8f2e02554ae91833a732804ac2dd/models/pose_regressor.onnx>
- Git blob: `dadf614b35eb26b3a61db8961ca3a151605fb55f`
- SHA256: `78bdab58ef9ddad97930150b5b1409868fd81c66cd6baaab5cc3b2d42e9d845a`
- Upstream repository license: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**, copyright Klischa, 2026.
- Upstream notice: <https://github.com/Klischa/astra2/blob/klischa-patch/LICENSE.MD>
- License text: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>

The exported experimental weights in `artifacts/` are derived from those encoder
weights and are distributed under **CC BY-NC-SA 4.0**, with this attribution.
This is not a claim that the upstream weights are available under a permissive
commercial software license. Keep the attribution and applicable license when
redistributing derivative weights. See the linked full legal text for terms.

Changes made in this project: recovery of the inference network in PyTorch;
reset of the regression head; synthetic CPU training of encoder and head;
internal centering and shared scale normalization; explicit unit-quaternion
source-to-target pose output; centroid compensation; shared bidirectional head
with inverse consistency; new export metadata and independent evaluation.

No upstream training dataset or training recipe was available. The synthetic
point clouds are generated procedurally by this repository; no third-party
scan dataset was downloaded. The original ONNX is fetched separately and is
not duplicated in Git.

This notice covers the upstream/derived model material. It does not assign a
new blanket license to unrelated repository contents or third-party Python packages.
