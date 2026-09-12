# SwingNet pretrained isolated test — frozen design

2026-09-13. No frontend, annotation interface, production model or source changes.

## Provenance and compatibility

- Badminton repo commit adac0c8f1abd999ade9997d9334f2585788ab7f9.
- Its current `rt_conversion_datasets.py` is a placeholder; restored actual
  author code from commit 989ff9cf2bf572d27920facd609f5aef3c0a9701.
- Zenodo record 14677727, assets.zip, member
  weights/SwingNet/models/fold5_swingnet_3000.pth.tar, selected by the author's
  inference notebook, NOT selected using local evaluation scores.
- Checkpoint SHA256 d896c3e4f1a9a8ad629feadc9d5c29944d9cbb00ddcb86b341bb68e6b8e8901d.
- Only that 63,336,909-byte member was fetched with HTTP ranges, ZIP CRC checked.
- Original GolfDB network from a63b4ce8c0900d09abbe92519223efe16810f5f4.
  Missing model_custom.py reconstructed as its identical MobileNetV2 + one
  bidirectional LSTM(256), with a two-logit head dictated by checkpoint shapes.
  Require strict matching of ALL checkpoint tensors, no randomly initialized
  unfilled layers and no additional ImageNet weights.

## Exact upstream preprocessing / inference arm

Sobel gradient magnitude, 0.5 absolute difference across t and t-2 (t=1 uses
t-1), 3x3 box filter, per-frame minmax to uint8, BGR=(0,signal,signal), initial
black frame. Encode full-size mp4v then read as the author did. Guard only the
constant-image divide-by-zero case. No source videos are copied.

Replicate notebook crop x=280:1000 for 1280x720 inputs, INTER_AREA resize
180x180, then its computed padding of 35 top/bottom, yielding 250x180.
This discrepancy with the paper's stated 180x180 is explicitly preserved.
Normalize RGB using ImageNet means/std. Independent 64-frame recurrent chunks.
Author decoder: values above the 80th percentile (method=higher), consecutive
runs >=3, highest probability frame. Retain its last-candidate omission in
the primary upstream baseline, document it. Class 0=hit, 1=background.

## Data and comparison

No local neural network training. Development: vid_2e2258f261aa (58 hits) and
vid_a598228fbc11 (64). Validation: vid_576b0c1ab370 (41).
Six old test clips / 174 hits are regression data, not a new blind test;
they have already been analyzed in earlier experiments. Do not tune on them.
Original pretraining membership cannot be certified from model metadata.
No cross-match/generalization or new independent-test claim.

A: stored rules v0.6. B: frozen author SwingNet above.
C: independent union of rule and SwingNet candidates, then confidence/evidence
fusion. Sweep only 18 predeclared settings on development:
rule keep SwingNet-rank threshold in [0,0.4,0.6]; new SwingNet candidate minimum
local raw-rule evidence in [0,0.2,0.35]; frame refinement in [off,on].
Rule scores from existing rules evaluated at threshold 0 and min separation 1,
so no new classifier is fitted. SwingNet percentile ranks are per clip.
Rule candidates require SwingNet rank at +/-4 frames above keep threshold.
SwingNet candidates require maximum raw rule score within +/-4 above add threshold.
Union merged at 6 frames; prioritize agreement, then stronger rank/physical score.
Refinement (when on): choose raw-rule peak within +/-4 maximizing rule score
times (0.5+0.5*local SwingNet rank), and move only when rule evidence >=0.35.
No forced alternation, no global half-second exclusion, no Gold time offset.

Choose best development strict F1 (tie: precision, fewer events), record top 3
configurations. On validation select among those 3 and fallback A; C deployable
only if validation strict F1 exceeds A and recall does not decrease by >0.02.
Always report best validation fusion diagnostic even if fallback A selected.
Freeze choice/configuration hashes BEFORE accessing regression labels/scores.

Same existing monotonic one-to-one matcher: primary +/-5 + Gold uncertainty,
secondary +6 reaction allowance. Report P/R/F1 and median timing error. B has
no hitter head; do not fabricate hitter accuracy for it. C may use existing
rule side at associated physical peak; unknown otherwise. Retain FP/FN lists.

BST: output window manifests with upstream previous-hit/next-hit+extension
boundaries, separate from production. No ball-type accuracy claim.
No assumption that raw softmax equals calibrated correctness probability.
