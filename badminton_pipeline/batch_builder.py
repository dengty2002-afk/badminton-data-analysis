"""Resumable batch orchestration for ShuttleSet-like visual datasets.

The builder deliberately orchestrates the existing production stages instead
of introducing another inference path.  State is written atomically after
every stage, so a later invocation skips completed work and retries failures.
Court proposals always require explicit human acceptance before inference.
"""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from badminton_pipeline.calibration.auto_court import propose_court
from badminton_pipeline.exports.batch_dataset import (
    build_batch_dataset,
    validate_per_video_dataset,
)
from badminton_pipeline.ingest.register_video import register_video, sha256_file
from badminton_pipeline.run_pipeline import build_parser as build_pipeline_parser
from badminton_pipeline.run_pipeline import run_pipeline
from badminton_pipeline.video.main_view import run as run_main_view


SCHEMA_VERSION = "1.0"
BUILDER_VERSION = "batch-builder-v0.4"
VIDEO_STAGES = ("register", "main_view", "court", "infer", "validate")
TERMINAL_VIDEO_STATUSES = {"completed", "completed_with_warning"}
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


@contextmanager
def _batch_lock(path: Path):
    """Hold a non-blocking Windows file lock for the complete batch run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(
                f"batch is already running (lock unavailable): {path}"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps({"pid": os.getpid(), "acquired_at": utc_now()}).encode("utf-8")
        )
        stream.flush()
        yield
    finally:
        try:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        stream.close()


def _stage(status: str = "pending") -> dict[str, Any]:
    return {"status": status, "attempts": 0}


def _new_video(path: Path) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "file_name": path.name,
        "status": "pending",
        "video_id": None,
        "quality_status": None,
        "quality_warnings": [],
        "stages": {name: _stage() for name in VIDEO_STAGES},
    }


def _new_state(
    batch_id: str, videos: Iterable[Path], configuration: dict[str, Any]
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "batch_id": batch_id,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "configuration": configuration,
        "videos": [_new_video(path) for path in videos],
        "merge": _stage(),
        "summary": {},
    }


def _mark_derived_source_variants(state: dict[str, Any]) -> int:
    """Exclude legacy main-view encodes when the original broadcast is present."""
    rows = state.get("videos", [])
    originals = {
        str(row.get("file_name") or "").casefold(): row
        for row in rows
        if Path(str(row.get("source_path") or "")).parent.name.casefold()
        != "main_view_outputs"
    }
    excluded = 0
    for row in rows:
        path = Path(str(row.get("source_path") or ""))
        if path.parent.name.casefold() != "main_view_outputs":
            continue
        suffix = "_主视角"
        if not path.stem.endswith(suffix):
            continue
        canonical_name = f"{path.stem[:-len(suffix)]}{path.suffix}".casefold()
        canonical = originals.get(canonical_name)
        if canonical is None:
            continue
        row["status"] = "excluded_source_variant"
        row["excluded_reason"] = (
            "legacy main-view derivative duplicates an original broadcast in this batch"
        )
        row["canonical_source_path"] = canonical.get("source_path")
        row["canonical_file_name"] = canonical.get("file_name")
        row["quality_status"] = "excluded_duplicate"
        row["quality_warnings"] = []
        excluded += 1
    return excluded


def discover_videos(inputs: Iterable[Path], recursive: bool = False) -> list[Path]:
    """Resolve files/directories to a stable, duplicate-free video list."""
    found: dict[str, Path] = {}
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"unsupported video extension: {path}")
            found[str(path).casefold()] = path
            continue
        if not path.is_dir():
            raise ValueError(f"input does not exist: {path}")
        iterator = path.rglob("*") if recursive else path.iterdir()
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                resolved = candidate.resolve()
                found[str(resolved).casefold()] = resolved
    if not found:
        raise ValueError("no supported videos were found")
    return sorted(found.values(), key=lambda item: str(item).casefold())


def _latest_accepted_calibration(data_dir: Path, video_id: str) -> Path | None:
    directory = data_dir / "calibrations" / video_id
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("v*.json") if directory.is_dir() else ():
        try:
            value = _read_json(path)
            if value.get("quality_status") == "accepted" and value.get("video_id") == video_id:
                candidates.append((int(value.get("version") or 0), path.resolve()))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _accepted_proposal_calibration(proposal_path: Path, video_id: str) -> Path | None:
    if not proposal_path.is_file():
        return None
    proposal = _read_json(proposal_path)
    if proposal.get("video_id") != video_id or not proposal.get("accepted"):
        return None
    raw = proposal.get("calibration")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (proposal_path.parent / path).resolve()
    return path if path.is_file() else None


def _inference_calibration_path(video: dict[str, Any]) -> Path | None:
    raw = (
        video.get("stages", {})
        .get("infer", {})
        .get("result", {})
        .get("postprocess", {})
        .get("calibration")
    )
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve()


def _court_sample_times(
    main_view_result: dict[str, Any] | None,
    fallback_ms: int,
    *,
    max_segment_samples: int = 6,
) -> list[int]:
    """Prefer midpoints of long main-camera segments over broadcast intros."""
    values: list[int] = []
    result = main_view_result or {}
    fps = float(result.get("fps") or 0.0)
    segments = result.get("segments") if isinstance(result.get("segments"), list) else []
    if fps > 0:
        ranked = sorted(
            (row for row in segments if isinstance(row, dict)),
            key=lambda row: int(row.get("frame_count") or 0),
            reverse=True,
        )
        for segment in ranked[:max_segment_samples]:
            start = int(segment.get("start_frame") or 0)
            end = int(segment.get("end_frame") or start)
            midpoint_ms = round(((start + end) / 2) / fps * 1000)
            if midpoint_ms >= 0 and midpoint_ms not in values:
                values.append(midpoint_ms)
    fallback_ms = max(0, int(fallback_ms))
    if fallback_ms not in values:
        values.append(fallback_ms)
    return values


class BatchBuilder:
    """Run and persist one batch while keeping stage functions injectable for tests."""

    def __init__(
        self,
        *,
        batch_id: str,
        inputs: Iterable[Path],
        data_dir: Path,
        work_dir: Path,
        output_dir: Path,
        device: str = "cuda",
        frame_time_ms: int = 10000,
        match_id: str = "unknown",
        source: str = "local",
        domain: str = "fixed-camera-singles",
        redistribution: str = "private",
        pipeline_options: dict[str, Any] | None = None,
        rerun_warnings: bool = False,
        workspace_warn_mb: int = 2048,
        main_view_template: Path | None = None,
        main_view_threshold: float = 0.8,
        main_view_analysis_width: int = 160,
        register_fn: Callable[..., dict[str, Any]] = register_video,
        court_fn: Callable[..., dict[str, Any]] = propose_court,
        pipeline_fn: Callable[[argparse.Namespace], dict[str, Any]] = run_pipeline,
        validate_fn: Callable[[Path], dict[str, Any]] = validate_per_video_dataset,
        merge_fn: Callable[[Iterable[Path], Path], dict[str, Any]] = build_batch_dataset,
    ) -> None:
        self.batch_id = batch_id
        self.inputs = [Path(path).resolve() for path in inputs]
        self.data_dir = data_dir.resolve()
        self.work_dir = work_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.state_path = self.work_dir / "batch_state.json"
        self.summary_path = self.work_dir / "batch_summary.json"
        self.device = device
        self.frame_time_ms = frame_time_ms
        self.match_id = match_id
        self.source = source
        self.domain = domain
        self.redistribution = redistribution
        self.pipeline_options = pipeline_options or {}
        self.rerun_warnings = rerun_warnings
        self.workspace_warn_mb = max(1, int(workspace_warn_mb))
        self.main_view_template = None if main_view_template is None else main_view_template.resolve()
        self.main_view_threshold = float(main_view_threshold)
        self.main_view_analysis_width = int(main_view_analysis_width)
        self.register_fn = register_fn
        self.court_fn = court_fn
        self.pipeline_fn = pipeline_fn
        self.validate_fn = validate_fn
        self.merge_fn = merge_fn

    def _configuration(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "frame_time_ms": self.frame_time_ms,
            "match_id": self.match_id,
            "source": self.source,
            "domain": self.domain,
            "redistribution": self.redistribution,
            "pipeline_options": self.pipeline_options,
            "workspace_warn_mb": self.workspace_warn_mb,
            "main_view_template": None if self.main_view_template is None else str(self.main_view_template),
            "main_view_threshold": self.main_view_threshold,
            "main_view_analysis_width": self.main_view_analysis_width,
        }

    def _load_or_create(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            state = _new_state(self.batch_id, self.inputs, self._configuration())
            _mark_derived_source_variants(state)
            self._save(state)
            return state
        state = _read_json(self.state_path)
        for row in state.get("videos", []):
            stages = row.setdefault("stages", {})
            for name in VIDEO_STAGES:
                stages.setdefault(name, _stage())
        if state.get("batch_id") != self.batch_id:
            raise ValueError(f"batch_id differs from existing state: {self.state_path}")
        existing = [Path(str(row["source_path"])).resolve() for row in state.get("videos", [])]
        if existing != self.inputs:
            raise ValueError(
                "input video list differs from existing batch state; use the original list "
                "or create a new --batch-id"
            )
        if state.get("configuration") != self._configuration():
            raise ValueError(
                "batch configuration differs from existing state; resume with the original "
                "options or create a new --batch-id"
            )
        _mark_derived_source_variants(state)
        return state

    def _save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        _write_json(self.state_path, state)

    def _mark_started(self, state: dict[str, Any], video: dict[str, Any], name: str) -> None:
        stage = video["stages"][name]
        stage["status"] = "running"
        stage["attempts"] = int(stage.get("attempts") or 0) + 1
        stage["started_at"] = utc_now()
        stage.pop("error", None)
        video["status"] = "running"
        self._save(state)

    def _mark_completed(
        self, state: dict[str, Any], video: dict[str, Any], name: str, result: dict[str, Any]
    ) -> None:
        stage = video["stages"][name]
        stage.update({"status": "completed", "completed_at": utc_now(), "result": result})
        self._save(state)

    def _mark_failed(
        self, state: dict[str, Any], video: dict[str, Any], name: str, exc: BaseException
    ) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if name not in video["stages"]:
            name = next(
                (
                    stage_name
                    for stage_name in VIDEO_STAGES
                    if video["stages"][stage_name].get("status") != "completed"
                ),
                VIDEO_STAGES[-1],
            )
        video["stages"][name].update(
            {"status": "failed", "failed_at": utc_now(), "error": message}
        )
        video["status"] = "failed"
        video["failed_stage"] = name
        video["error"] = message
        self._save(state)

    def _proposal_path(self, video_id: str) -> Path:
        return self.data_dir / "court_proposals" / f"{video_id}.court_proposal.json"

    def _latest_calibration(self, video_id: str) -> Path | None:
        proposal = self._proposal_path(video_id)
        return _accepted_proposal_calibration(proposal, video_id) or _latest_accepted_calibration(
            self.data_dir, video_id
        )

    def _calibration_changed_after_inference(self, video: dict[str, Any]) -> bool:
        video_id = str(video.get("video_id") or "")
        if not video_id:
            return False
        latest = self._latest_calibration(video_id)
        used = _inference_calibration_path(video)
        return latest is not None and used is not None and latest != used

    def _run_video(self, state: dict[str, Any], video: dict[str, Any]) -> None:
        source_path = Path(str(video["source_path"])).resolve()
        if not source_path.is_file():
            raise ValueError(f"source video is unavailable: {source_path}")

        # Register. A completed record is rechecked against source content so
        # silently replacing a video cannot reuse old downstream artifacts.
        register_stage = video["stages"]["register"]
        if register_stage.get("status") == "completed":
            current_checksum = sha256_file(source_path)
            if current_checksum != register_stage.get("result", {}).get("checksum"):
                video["failed_stage"] = "register"
                raise ValueError(f"source video changed after registration: {source_path}")
        else:
            self._mark_started(state, video, "register")
            result = self.register_fn(
                source_path,
                data_dir=self.data_dir,
                match_id=self.match_id,
                source=self.source,
                domain=self.domain,
                redistribution=self.redistribution,
            )
            video["video_id"] = result["video_id"]
            self._mark_completed(state, video, "register", result)
        video_id = str(video.get("video_id") or register_stage.get("result", {}).get("video_id"))
        video["video_id"] = video_id

        # Optional no-copy main-camera indexing for broadcast videos. The
        # source MP4 remains untouched; only source-frame ranges are persisted.
        main_view_stage = video["stages"]["main_view"]
        main_view_manifest: Path | None = None
        if self.main_view_template is None:
            if main_view_stage.get("status") != "completed":
                main_view_stage.update(
                    {"status": "completed", "completed_at": utc_now(), "result": {"status": "not_requested"}}
                )
                self._save(state)
        else:
            main_view_manifest = self.work_dir / "videos" / video_id / "main-view.json"
            if main_view_stage.get("status") != "completed" or not main_view_manifest.is_file():
                self._mark_started(state, video, "main_view")
                result = run_main_view(
                    argparse.Namespace(
                        input=source_path,
                        template=self.main_view_template,
                        output_video=None,
                        output_manifest=main_view_manifest,
                        manifest_only=True,
                        threshold=self.main_view_threshold,
                        adaptive_margin=0.03,
                        analysis_width=self.main_view_analysis_width,
                        court_frames_threshold=5,
                        non_court_frames_threshold=5,
                    )
                )
                if not result.get("segment_count"):
                    raise ValueError(f"no main-camera segments found: {source_path}")
                self._mark_completed(state, video, "main_view", result)

        # Court stage. An accepted calibration may be created between batch
        # invocations by reviewing the proposal with auto_court --accept.
        court_stage = video["stages"]["court"]
        proposal_path = self._proposal_path(video_id)
        calibration = self._latest_calibration(video_id)
        if calibration is not None:
            court_stage.update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "result": {
                        "calibration": str(calibration),
                        "proposal": str(proposal_path) if proposal_path.is_file() else None,
                    },
                }
            )
            self._save(state)
        else:
            existing_proposal = _read_json(proposal_path) if proposal_path.is_file() else None
            prior_sampling = (existing_proposal or {}).get("batch_sampling", {})
            prior_invalid = (existing_proposal or {}).get("validation", {}).get("valid") is False
            needs_proposal = existing_proposal is None or (
                prior_invalid and prior_sampling.get("strategy") != "main_view_longest_segment_midpoints"
            )
            if needs_proposal:
                self._mark_started(state, video, "court")
                sample_times = _court_sample_times(
                    main_view_stage.get("result"), self.frame_time_ms
                )
                attempts: list[dict[str, Any]] = []
                proposal: dict[str, Any] | None = None
                for sample_time_ms in sample_times:
                    proposal = self.court_fn(
                        source_path,
                        video_id,
                        self.data_dir / "court_proposals",
                        frame_time_ms=sample_time_ms,
                        accept=False,
                        data_dir=self.data_dir,
                    )
                    validation = proposal.get("validation")
                    attempts.append(
                        {
                            "frame_time_ms": sample_time_ms,
                            "valid": None if not isinstance(validation, dict) else validation.get("valid"),
                            "message": None if not isinstance(validation, dict) else validation.get("message"),
                        }
                    )
                    if not isinstance(validation, dict) or validation.get("valid") is not False:
                        break
                assert proposal is not None
                proposal["batch_sampling"] = {
                    "strategy": "main_view_longest_segment_midpoints",
                    "attempts": attempts,
                }
                if proposal_path.is_file():
                    _write_json(proposal_path, proposal)
                court_stage["result"] = proposal
            court_stage["status"] = "waiting_review"
            review_frame_time_ms = int(
                (court_stage.get("result") or existing_proposal or {}).get("frame_time_ms")
                or self.frame_time_ms
            )
            court_stage["review_command"] = (
                f"python -m badminton_pipeline.calibration.auto_court --input \"{source_path}\" "
                f"--video-id {video_id} --frame-time-ms {review_frame_time_ms} --accept"
            )
            video["status"] = "waiting_court_review"
            video.pop("failed_stage", None)
            video.pop("error", None)
            self._save(state)
            return

        # Full inference and per-video export.
        infer_stage = video["stages"]["infer"]
        if infer_stage.get("status") == "completed":
            used_calibration = _inference_calibration_path(video)
            if used_calibration is not None and used_calibration != calibration:
                video["stages"]["infer"] = infer_stage = _stage()
                video["stages"]["validate"] = _stage()
                video["quality_status"] = None
                video["quality_warnings"] = []
                video["status"] = "running"
                state["merge"] = _stage()
                self._save(state)
        run_dir = self.work_dir / "videos" / video_id
        run_manifest = run_dir / "run.json"
        dataset_manifest = run_dir / "dataset" / "manifest.json"
        if infer_stage.get("status") != "completed" or not run_manifest.is_file() or not dataset_manifest.is_file():
            self._mark_started(state, video, "infer")
            pipeline_args = build_pipeline_parser().parse_args(
                [
                    "--input", str(source_path),
                    "--calibration", str(calibration),
                    "--output-dir", str(run_dir),
                    "--device", self.device,
                    "--video-record", str(self.data_dir / "videos" / f"{video_id}.json"),
                ]
            )
            for key, value in self.pipeline_options.items():
                if not hasattr(pipeline_args, key):
                    raise ValueError(f"unknown pipeline option: {key}")
                setattr(pipeline_args, key, value)
            if main_view_manifest is not None:
                pipeline_args.segments_manifest = main_view_manifest
            # Complete caches are reused after calibration-only changes; an
            # interrupted raw JSONL is also validated and resumed on every
            # ordinary retry.
            pipeline_args.reuse_existing_raw = True
            result = self.pipeline_fn(pipeline_args)
            self._mark_completed(state, video, "infer", result)

        # Validate artifact hashes/schema and retain machine-quality warnings.
        validate_stage = video["stages"]["validate"]
        should_validate = validate_stage.get("status") != "completed"
        should_validate = should_validate or (
            self.rerun_warnings and video.get("quality_status") == "warning"
        )
        if should_validate:
            self._mark_started(state, video, "validate")
            result = self.validate_fn(run_dir / "dataset")
            video["quality_status"] = result["quality_status"]
            video["quality_warnings"] = result["warnings"]
            self._mark_completed(state, video, "validate", result)
        quality = str(video.get("quality_status") or "unknown")
        video["status"] = "completed" if quality == "ok" else "completed_with_warning"
        video.pop("failed_stage", None)
        video.pop("error", None)
        self._save(state)

    @staticmethod
    def _counts(state: dict[str, Any]) -> dict[str, int]:
        counts = {
            "total": len(state.get("videos", [])),
            "processable_total": 0,
            "completed": 0,
            "warnings": 0,
            "waiting_review": 0,
            "failed": 0,
            "pending": 0,
            "excluded": 0,
        }
        for row in state.get("videos", []):
            status = row.get("status")
            if status == "excluded_source_variant":
                counts["excluded"] += 1
                continue
            counts["processable_total"] += 1
            if status == "completed":
                counts["completed"] += 1
            elif status == "completed_with_warning":
                counts["completed"] += 1
                counts["warnings"] += 1
            elif status == "waiting_court_review":
                counts["waiting_review"] += 1
            elif status == "failed":
                counts["failed"] += 1
            else:
                counts["pending"] += 1
        return counts

    def _write_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        counts = self._counts(state)
        source_bytes = sum(Path(str(row["source_path"])).stat().st_size for row in state["videos"])
        canonical_source_bytes = sum(
            Path(str(row["source_path"])).stat().st_size
            for row in state["videos"]
            if row.get("status") != "excluded_source_variant"
        )
        work_bytes = _tree_bytes(self.work_dir)
        output_inside_work = self.output_dir == self.work_dir or self.work_dir in self.output_dir.parents
        output_bytes = 0 if output_inside_work else _tree_bytes(self.output_dir)
        generated_bytes = work_bytes + output_bytes
        size_warnings: list[str] = []
        if generated_bytes >= self.workspace_warn_mb * 1024 * 1024:
            size_warnings.append(
                f"batch-generated artifacts use {generated_bytes / 1024 / 1024:.1f} MiB, "
                f"above the configured {self.workspace_warn_mb} MiB warning threshold"
            )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "batch_id": self.batch_id,
            "status": state["status"],
            "updated_at": utc_now(),
            "counts": counts,
            "state": str(self.state_path),
            "dataset": state.get("merge", {}).get("result"),
            "storage": {
                "source_video_bytes": source_bytes,
                "canonical_source_video_bytes": canonical_source_bytes,
                "source_videos_are_not_copied": True,
                "work_bytes": work_bytes,
                "output_bytes_outside_work": output_bytes,
                "generated_bytes": generated_bytes,
                "warning_threshold_mb": self.workspace_warn_mb,
                "warnings": size_warnings,
            },
            "videos": [
                {
                    "video_id": row.get("video_id"),
                    "file_name": row.get("file_name"),
                    "source_path": row.get("source_path"),
                    "status": row.get("status"),
                    "quality_status": row.get("quality_status"),
                    "quality_warnings": row.get("quality_warnings", []),
                    "failed_stage": row.get("failed_stage"),
                    "error": row.get("error"),
                    "excluded_reason": row.get("excluded_reason"),
                    "canonical_source_path": row.get("canonical_source_path"),
                    "court_review_command": row.get("stages", {}).get("court", {}).get("review_command"),
                }
                for row in state.get("videos", [])
            ],
        }
        _write_json(self.summary_path, summary)
        state["summary"] = {"path": str(self.summary_path), "counts": counts}
        self._save(state)
        return summary

    def _run_locked(self) -> dict[str, Any]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_or_create()
        state["status"] = "running"
        self._save(state)
        for video in state["videos"]:
            if video.get("status") == "excluded_source_variant":
                continue
            if video.get("status") == "completed" and not self._calibration_changed_after_inference(video):
                continue
            if (
                video.get("status") == "completed_with_warning"
                and not self.rerun_warnings
                and not self._calibration_changed_after_inference(video)
            ):
                continue
            try:
                self._run_video(state, video)
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                running = next(
                    (name for name in VIDEO_STAGES if video["stages"][name].get("status") == "running"),
                    video.get("failed_stage") or "unknown",
                )
                self._mark_failed(state, video, str(running), exc)
                video["traceback"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()

        counts = self._counts(state)
        all_ready = (
            counts["completed"] == counts["processable_total"]
            and counts["processable_total"] > 0
        )
        merge = state["merge"]
        if all_ready:
            dataset_dirs = [
                self.work_dir / "videos" / str(row["video_id"]) / "dataset"
                for row in state["videos"]
                if row.get("status") in TERMINAL_VIDEO_STATUSES
            ]
            try:
                needs_merge = merge.get("status") != "completed" or not (
                    self.output_dir / "manifest.json"
                ).is_file()
                if needs_merge:
                    merge["status"] = "running"
                    merge["attempts"] = int(merge.get("attempts") or 0) + 1
                    merge["started_at"] = utc_now()
                    merge.pop("error", None)
                    self._save(state)
                    result = self.merge_fn(dataset_dirs, self.output_dir)
                    merge.update({"status": "completed", "completed_at": utc_now(), "result": result})
                state["status"] = "completed_with_warning" if counts["warnings"] else "completed"
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                merge.update(
                    {
                        "status": "failed",
                        "failed_at": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                state["status"] = "failed"
        elif counts["waiting_review"] and not counts["failed"]:
            state["status"] = "waiting_court_review"
        elif counts["failed"]:
            state["status"] = "partial_failure"
        else:
            state["status"] = "incomplete"
        self._save(state)
        return self._write_summary(state)

    def run(self) -> dict[str, Any]:
        with _batch_lock(self.work_dir / "batch.lock"):
            return self._run_locked()


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True, help="Stable identifier used for resume")
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="Video or directory; repeatable. Directories are non-recursive unless --recursive.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--work-dir", type=Path, help="Default: data/batch_runs/<batch-id>")
    parser.add_argument("--output-dir", type=Path, help="Default: data/datasets/<batch-id>")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--frame-time-ms", type=int, default=10000)
    parser.add_argument("--match-id", default="unknown")
    parser.add_argument("--source", default="local")
    parser.add_argument("--domain", default="fixed-camera-singles")
    parser.add_argument("--redistribution", choices=("allowed", "private", "unknown"), default="private")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug/smoke limit passed to every video")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip-shuttle", action="store_true")
    parser.add_argument("--skip-semantic-prediction", action="store_true")
    parser.add_argument("--stroke-model", type=Path)
    parser.add_argument("--segments-manifest", type=Path, help="Only valid for a one-video batch")
    parser.add_argument("--rerun-warnings", action="store_true", help="Revalidate completed warning videos")
    parser.add_argument("--workspace-warn-mb", type=int, default=2048)
    parser.add_argument("--main-view-template", type=Path, help="Enable no-copy broadcast main-camera indexing")
    parser.add_argument("--main-view-threshold", type=float, default=0.8)
    parser.add_argument("--main-view-analysis-width", type=int, default=160)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        videos = discover_videos(args.input, recursive=args.recursive)
        data_dir = args.data_dir.resolve()
        work_dir = (args.work_dir or data_dir / "batch_runs" / args.batch_id).resolve()
        output_dir = (args.output_dir or data_dir / "datasets" / args.batch_id).resolve()
        if args.segments_manifest and len(videos) != 1:
            raise ValueError("--segments-manifest currently requires exactly one input video")
        result = BatchBuilder(
            batch_id=args.batch_id,
            inputs=videos,
            data_dir=data_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            device=args.device,
            frame_time_ms=args.frame_time_ms,
            match_id=args.match_id,
            source=args.source,
            domain=args.domain,
            redistribution=args.redistribution,
            pipeline_options={
                "max_frames": args.max_frames,
                "stride": args.stride,
                "skip_shuttle": args.skip_shuttle,
                "skip_semantic_prediction": args.skip_semantic_prediction,
                **({"stroke_model": str(args.stroke_model.resolve())} if args.stroke_model else {}),
                **({"segments_manifest": str(args.segments_manifest.resolve())} if args.segments_manifest else {}),
            },
            rerun_warnings=args.rerun_warnings,
            workspace_warn_mb=args.workspace_warn_mb,
            main_view_template=args.main_view_template,
            main_view_threshold=args.main_view_threshold,
            main_view_analysis_width=args.main_view_analysis_width,
        ).run()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"completed", "completed_with_warning", "waiting_court_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
