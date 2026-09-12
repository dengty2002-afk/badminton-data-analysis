# HitNet isolated experiment

Read [REPORT.md](REPORT.md) for measured results and [PROTOCOL.md](PROTOCOL.md)
for the frozen design. This is a **local PyTorch architecture reproduction**,
not the unavailable official TensorFlow pretrained checkpoint.

No production imports point into this folder. Existing features and evaluation
code are read-only inputs. No frontend changes or annotation UI are involved.

## Files

- `upstream/`: pinned official source and Adobe Research License.
- `hitnet.py`: network, feature adapter, original postprocessing, BST windows.
- `run_experiment.py`: training, checkpoint selection, frozen evaluation.
- `infer.py`: standalone inference on one contiguous cached feature stream.
- `test_contract.py`: seven contract and edge-case checks.
- `results/summary.json`: all measured aggregate scores.
- `results/manifest-before.json`: hashes of protected inputs and source.
- `results/frozen-checkpoints.json`: checkpoint hashes before test scoring.
- `results/hitnet_seed_*/`: predictions, timing-window manifests and matches.
- `smoke-inference/`: CLI smoke result on the development video, without Gold.

## Run from the project root

Use the existing `.venv-model` Python. No installation required. The measured
environment is Python 3.12, PyTorch 2.13.0+cu130, NVIDIA RTX 5060.

```powershell
.\.venv-model\Scripts\python.exe -B -m unittest discover -s experiments/monotrack-hitnet -p test_contract.py -v
```

Retraining/evaluation requires a new output directory (never overwrite a result):

```powershell
.\.venv-model\Scripts\python.exe -B experiments/monotrack-hitnet/run_experiment.py --output experiments/monotrack-hitnet/results-new
```

Inference requires no human labels:

```powershell
.\.venv-model\Scripts\python.exe -B experiments/monotrack-hitnet/infer.py `
  --frames data/runs/full-local-v0.3-play-gate-dev/frames.enriched.jsonl `
  --calibration data/calibrations/vid_2e2258f261aa/v1.json `
  --checkpoint experiments/monotrack-hitnet/results/hitnet-local-seed-43.pt `
  --output experiments/monotrack-hitnet/predict-new
```

Seed 43 has the highest **validation** window macro-F1 among these three runs;
it is not selected for best test performance. These tiny-data checkpoints are
experimental, not recommended production replacements. CLI currently accepts
only contiguous, approximately 30 FPS streams. Full broadcasts with cuts need
independent continuous segments and automatic rally/camera handling.

`bst-windows.jsonl` contains frame ranges; no MP4 clips are generated. The
production BST adapter is not changed and no semantic accuracy claim is made.
