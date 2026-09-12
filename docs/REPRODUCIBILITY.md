# Reproducibility and publication boundary

This source snapshot separates the working application from local videos, pretrained binaries, cached features and machine-specific manifests. The original working directory was preserved.

## Three levels of use

1. **Source and synthetic tests:** install the base package, run `examples/evaluate_hits.py` and Python tests. No videos or pretrained models are required. One optional weight-integrity integration test is skipped until its weights are present.
2. **Inference on your videos:** use Windows, install the GPU runtime, fetch the manifest-pinned models, register your video and create/accept court calibration. The repository includes the minimal licensed vendor runtime. A compatible NVIDIA driver is required for CUDA execution. Batch locking currently uses Windows `msvcrt`; Linux batch execution is not supported.
3. **Historical result reproduction:** restore the exact original recordings and per-video registrations/features/Gold expected by the experiment scripts. Those input files are not included. Aggregate result JSON files are evidence snapshots, not sufficient input to replay all experiments.

## SwingNet

Run `python experiments/swingnet/fetch_upstream.py` to retrieve the four source files listed by fixed commit and SHA256. Consult their authors' terms before use. Then run:

```text
python experiments/swingnet/fetch_assets.py --extract weights/SwingNet/models/fold5_swingnet_3000.pth.tar
```

The downloader uses HTTP byte ranges to extract only the selected checkpoint, not the complete 14.5 GB archive. Standalone inference takes a user-supplied 1280×720 video. Historical evaluation and smoke scripts also require the original private local inputs.

## Research

`research/shi_player_profile_v2.py` reads `data/batch_runs/broadcast-uncensored-v1/videos/`, `research/identity_mapping.csv` and the preserved protocol in `research_outputs/shi_tactical_profile_v1/`. The full batch is not included. Selected tables and plots are snapshots; the 24-match research metrics are not newly validated by publishing this repository.

The source-only package adds a portable project root to the two manuscript construction scripts. Old prose in `docs/archive/` is historical, can contain outdated state descriptions, and is not the release status. Absolute working-directory examples have been replaced by `<PROJECT_ROOT>` in publication copies of prose.

## Optional UI

The existing `demo/` source is retained. Its publication copy uses generic local D1/R2 binding names in `hosting.local.json`, and contains no original deployment configuration or videos. No frontend feature development or website deployment is included in this release task. Full frontend build/runtime validation is not part of the source publication checks; existing UI examples may need local input data.

## Checks and limits

See `docs/PUBLICATION_CHECKS.md` for the checks actually executed. Passing code tests is not evidence of hit-detection or BST accuracy. No project-wide license was selected automatically.
