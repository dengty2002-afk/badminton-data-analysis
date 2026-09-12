# MonoTrack HitNet isolated experiment

Frozen before training or scoring new predictions, 2026-09-13.

Scope: detection experiment only. No frontend, production source, existing
weights, Gold, historical predictions, or historical evaluation is modified.

Official source: https://github.com/jhwang7628/monotrack
commit: 39e9227bd44fa2a579731b1cc97c7f529f391cd6.
The official HDF5 checkpoint `hitnet_conv_model_predict_direction-12-6-0.h5`
was not found in the repository, README, or official issue replies. Issues
3 (weights) and 4 (training labels) have no author solution. This experiment
is a locally trained architecture reproduction, NOT evaluation of the
official pretrained weights or reproduction of the published performance.

Network: 12 x 78 input, two bidirectional 64-unit GRUs, global temporal max,
three logits. Features: shuttle xy, bottom COCO17 xy, top COCO17 xy, four
court corners. Zero is missing; nonzero x and y separately scale to [1,2]
across a clip. Use the official inference corner order. Use cached existing
RTMPose/YOLO shuttle features for both methods; no new perception model.

Original TensorFlow architecture is implemented in PyTorch using the existing
environment, without dependency installation. This does not load Keras weights.
Training differs from upstream: Adam 0.001, class-weighted cross entropy,
gradient norm 2, batch 256, 80 epochs, seeds 17/29/43. Select checkpoints by
validation window macro F1. Temperature stays 1, no test calibration.
Labels follow upstream: any hit in offsets 6..11 labels a 12-frame window.
Conflicting two-side windows are omitted during training. Native approximately 30 FPS is
retained and recorded; original detector defaults to 25 FPS.

Training: vid_2e2258f261aa (58 hits), vid_a598228fbc11 (64 hits).
Validation: vid_576b0c1ab370 (41 hits).
Evaluation: six clips in play-state-prospective-001 (174 hits).
All are existing human hit labels, not machine pseudo-labels. These clips
share a camera setup; this is a retrospective same-camera comparison, not a
new blind test or cross-match generalization claim. Historical locked data
are read-only. No test-based threshold/window/epoch/seed selection.

Primary decoding: original MLHitDetector.dp_postprocessing, extracted from
the pinned source unchanged. Record its constraints: minimum half-second
spacing, alternating sides, roughly 1.2 hits/sec maximum count, +8 frame
offset. If no windows satisfy its confidence condition, flag the undefined
tau case and return no detections rather than invent scores.
Secondary diagnostic: original naive_postprocessing (default threshold).
Report separately, do not select based on test results.

Metrics: existing one-to-one matcher; primary tolerance +/-5 frames plus
declared Gold uncertainty; secondary historical relaxed metric additionally
allows 6 frames. Report event precision/recall/F1, hitter accuracy, signed
and absolute timing error, per-clip counts, all seeds. No Gold time shifts.

BST: export a separate window manifest using upstream boundaries with FPS
limits; do not patch production BST. No claim of improved ball-type accuracy
without semantic Gold. All inputs, upstream code and checkpoints are hashed.

License: upstream/LICENSE (Adobe Research License, noncommercial research).
