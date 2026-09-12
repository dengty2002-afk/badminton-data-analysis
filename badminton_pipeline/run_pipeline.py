"""Run model inference, Homography tracking, and hit-candidate generation end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from badminton_pipeline.exports.dataset import export_dataset
from badminton_pipeline.models.infer_video import run as run_inference
from badminton_pipeline.models.infer_video import video_id_for_path
from badminton_pipeline.stroke_classification.bst_adapter import annotate_event_rows
from badminton_pipeline.stroke_classification.destination import annotate_destinations
from badminton_pipeline.stroke_classification.features import export_bst_samples
from badminton_pipeline.tracking.postprocess import _video_ids_match, process_files


def _load_reusable_inference(
    raw_path: Path,
    input_video: Path,
    segments_manifest: Path | None,
) -> dict[str, object] | None:
    metadata_path = raw_path.with_suffix(".metadata.json")
    if not raw_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if Path(str(metadata.get("input"))).resolve() != input_video.resolve():
            return None
        recorded_manifest = metadata.get("segments_manifest")
        expected_manifest = None if segments_manifest is None else str(segments_manifest.resolve())
        if recorded_manifest != expected_manifest:
            return None
        expected_rows = int(metadata.get("processed_frames") or 0)
        if expected_rows <= 0:
            return None
        with raw_path.open("rb") as stream:
            actual_rows = sum(1 for line in stream if line.strip())
        if actual_rows != expected_rows:
            return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return {**metadata, "reused_existing_raw": True}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=root / "models" / "good-badminton")
    parser.add_argument("--manifest", type=Path, default=root / "configs" / "models.good-badminton.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--segments-manifest", type=Path, help="Optional no-copy main-view source-frame manifest")
    parser.add_argument(
        "--reuse-existing-raw", action="store_true",
        help="Reuse a complete matching frames.raw.jsonl cache, normally after calibration changes",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pose-confidence", type=float, default=0.2)
    parser.add_argument("--shuttle-confidence", type=float, default=0.18)
    parser.add_argument("--shuttle-top-k", type=int, default=5)
    parser.add_argument("--event-threshold", type=float, default=0.38)
    parser.add_argument("--min-hit-separation-sec", type=float, default=0.4)
    parser.add_argument("--hitter-window-sec", type=float, default=2 / 30)
    parser.add_argument("--play-state-window-sec", type=float, default=1.0)
    parser.add_argument("--play-state-support-sec", type=float, default=3.0)
    parser.add_argument("--play-state-gate-mode", choices=("off", "mechanics"), default="mechanics")
    parser.add_argument("--play-state-gate-min-link-sec", type=float, default=0.25)
    parser.add_argument("--play-state-gate-max-link-sec", type=float, default=4.0)
    parser.add_argument("--play-state-gate-pre-roll-sec", type=float, default=0.15)
    parser.add_argument("--play-state-gate-post-roll-sec", type=float, default=0.35)
    parser.add_argument("--play-state-gate-min-tracked-ratio", type=float, default=0.55)
    parser.add_argument("--play-state-gate-min-moving-ratio", type=float, default=0.25)
    parser.add_argument("--play-state-gate-min-sequence-support", type=float, default=0.75)
    parser.add_argument("--skip-shuttle", action="store_true")
    parser.add_argument("--video-record", type=Path, help="Optional registered video metadata JSON")
    parser.add_argument("--skip-dataset-export", action="store_true", help="Do not write ShuttleSet-like Parquet/CSV outputs")
    parser.add_argument("--export-bst-inputs", action="store_true", help="Also package candidate windows as BST .npz inputs")
    parser.add_argument("--bst-window-size", type=int, default=30)
    parser.add_argument(
        "--stroke-model", type=Path,
        default=root / "models" / "bst" / "bst_0_seq100_fine_serial2.pt",
        help="BST weight used for automatic stroke-type prelabels",
    )
    parser.add_argument(
        "--skip-semantic-prediction", action="store_true",
        help="Do not add stroke-type and destination prelabels",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("quality_status") != "accepted":
        raise ValueError("calibration is not accepted")
    input_video_id = video_id_for_path(args.input)
    calibration_video_id = str(calibration.get("video_id", ""))
    if not _video_ids_match(input_video_id, calibration_video_id):
        raise ValueError(
            f"calibration video_id {calibration_video_id!r} does not match input video_id {input_video_id!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "frames.raw.jsonl"
    enriched_path = args.output_dir / "frames.enriched.jsonl"
    events_path = args.output_dir / "events.jsonl"
    suppressed_events_path = args.output_dir / "events.suppressed.jsonl"
    provisional_events_path = args.output_dir / "events.provisional.jsonl"
    started = perf_counter()
    segments_manifest = (
        None if getattr(args, "segments_manifest", None) is None
        else Path(getattr(args, "segments_manifest"))
    )
    inference_args = argparse.Namespace(
        input=args.input,
        output=raw_path,
        models_dir=args.models_dir,
        manifest=args.manifest,
        device=args.device,
        stride=args.stride,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        segments_manifest=segments_manifest,
        pose_confidence=args.pose_confidence,
        shuttle_confidence=args.shuttle_confidence,
        shuttle_top_k=int(getattr(args, "shuttle_top_k", 5)),
        skip_shuttle=args.skip_shuttle,
        resume_existing_raw=bool(getattr(args, "reuse_existing_raw", False)),
    )
    inference = None
    if getattr(args, "reuse_existing_raw", False):
        inference = _load_reusable_inference(raw_path, args.input, segments_manifest)
    if inference is None:
        inference = run_inference(inference_args)
    postprocess = process_files(
        raw_path,
        args.calibration,
        enriched_path,
        events_path,
        fps=float(inference["fps"]),
        event_threshold=args.event_threshold,
        min_hit_separation_sec=float(getattr(args, "min_hit_separation_sec", 0.4)),
        hitter_window_sec=float(getattr(args, "hitter_window_sec", 2 / 30)),
        play_state_window_sec=float(getattr(args, "play_state_window_sec", 1.0)),
        play_state_support_sec=float(getattr(args, "play_state_support_sec", 3.0)),
        play_state_gate_mode=str(getattr(args, "play_state_gate_mode", "mechanics")),
        play_state_gate_min_link_sec=float(getattr(args, "play_state_gate_min_link_sec", 0.25)),
        play_state_gate_max_link_sec=float(getattr(args, "play_state_gate_max_link_sec", 4.0)),
        play_state_gate_pre_roll_sec=float(getattr(args, "play_state_gate_pre_roll_sec", 0.15)),
        play_state_gate_post_roll_sec=float(getattr(args, "play_state_gate_post_roll_sec", 0.35)),
        play_state_gate_min_tracked_ratio=float(getattr(args, "play_state_gate_min_tracked_ratio", 0.55)),
        play_state_gate_min_moving_ratio=float(getattr(args, "play_state_gate_min_moving_ratio", 0.25)),
        play_state_gate_min_sequence_support=float(getattr(args, "play_state_gate_min_sequence_support", 0.75)),
        output_suppressed_events_path=suppressed_events_path,
        output_provisional_events_path=provisional_events_path,
    )
    semantic = None
    if not getattr(args, "skip_semantic_prediction", False):
        stroke_model = Path(getattr(args, "stroke_model"))
        if not stroke_model.is_file():
            raise ValueError(
                f"stroke model is unavailable: {stroke_model}; use --skip-semantic-prediction "
                "for a geometry-only run"
            )
        semantic_temp_path = args.output_dir / "events.stroke.jsonl"
        stroke_summary = annotate_event_rows(
            enriched_path, events_path, semantic_temp_path, stroke_model,
            device=args.device, keypoint_threshold=float(args.pose_confidence),
        )
        destination_summary = annotate_destinations(
            enriched_path, semantic_temp_path, args.calibration, events_path,
        )
        semantic_temp_path.unlink(missing_ok=True)
        semantic = {"stroke_type": stroke_summary, "destination": destination_summary}
    dataset = None
    if not getattr(args, "skip_dataset_export", False):
        video_record = getattr(args, "video_record", None)
        if video_record is None:
            candidate = Path(__file__).resolve().parents[1] / "data" / "videos" / f"{input_video_id}.json"
            video_record = candidate if candidate.is_file() else None
        dataset = export_dataset(
            enriched_path,
            events_path,
            args.calibration,
            args.output_dir / "dataset",
            video_record_path=video_record,
            input_video=args.input,
        )
    bst_input = None
    if getattr(args, "export_bst_inputs", False):
        bst_input = export_bst_samples(
            enriched_path,
            events_path,
            args.output_dir / "bst-inputs.npz",
            window_size=int(getattr(args, "bst_window_size", 30)),
            keypoint_threshold=float(args.pose_confidence),
        )
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "video_id": input_video_id,
        "elapsed_sec": round(perf_counter() - started, 3),
        "inference": inference,
        "postprocess": postprocess,
        "semantic": semantic,
        "dataset": dataset,
        "bst_input": bst_input,
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_pipeline(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
