"""Build one validated ShuttleSet-like dataset from per-video exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
REQUIRED_ARTIFACTS = {
    "videos": "videos.parquet",
    "calibrations": "calibrations.parquet",
    "frames": "frames.parquet",
    "dataset_rows": "dataset_rows.parquet",
    "dictionary": "data_dictionary.json",
    "quality": "quality.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def discover_dataset_dirs(roots: Iterable[Path]) -> list[Path]:
    """Find per-video dataset directories below roots without recursive ambiguity."""
    found: set[Path] = set()
    for raw_root in roots:
        root = raw_root.resolve()
        if (root / "manifest.json").is_file():
            found.add(root)
            continue
        for manifest in root.glob("*/dataset/manifest.json"):
            found.add(manifest.parent.resolve())
        for manifest in root.glob("*/manifest.json"):
            found.add(manifest.parent.resolve())
    return sorted(found)


def _load_and_validate(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema in {manifest_path}: {manifest.get('schema_version')!r}"
        )
    artifacts: dict[str, Path] = {}
    declared = manifest.get("artifacts", {})
    for key, filename in REQUIRED_ARTIFACTS.items():
        path = dataset_dir / filename
        if not path.is_file():
            raise ValueError(f"missing required artifact: {path}")
        expected = declared.get(key, {}).get("sha256")
        actual = _sha256(path)
        if expected and actual != expected:
            raise ValueError(f"checksum mismatch: {path}")
        artifacts[key] = path
    manifest["_manifest_path"] = str(manifest_path.resolve())
    manifest["_manifest_sha256"] = _sha256(manifest_path)
    return manifest, artifacts


def validate_per_video_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validate one export and expose its machine-quality status to orchestration."""
    dataset_dir = dataset_dir.resolve()
    manifest, artifacts = _load_and_validate(dataset_dir)
    quality = json.loads(artifacts["quality"].read_text(encoding="utf-8"))
    video_id = str(manifest.get("video_id") or "")
    if not video_id:
        raise ValueError(f"manifest has no video_id: {dataset_dir / 'manifest.json'}")
    return {
        "status": "validated",
        "video_id": video_id,
        "quality_status": str(quality.get("status") or "unknown"),
        "warnings": list(quality.get("warnings") or []),
        "rows": dict(manifest.get("rows") or {}),
        "manifest": str((dataset_dir / "manifest.json").resolve()),
        "manifest_sha256": manifest["_manifest_sha256"],
    }


def _concat_tables(paths: list[Path], sort_by: list[str]):
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("polars is required for batch dataset export") from exc
    tables = [pl.read_parquet(path) for path in paths]
    base_schema = tables[0].schema
    for path, table in zip(paths[1:], tables[1:], strict=True):
        if table.schema != base_schema:
            raise ValueError(f"Parquet schema mismatch: {path}")
    return pl.concat(tables, how="vertical", rechunk=True).sort(sort_by)


def build_batch_dataset(dataset_dirs: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    """Validate and merge per-video exports into one deterministic dataset."""
    inputs = [Path(path).resolve() for path in dataset_dirs]
    if not inputs:
        raise ValueError("no per-video datasets were found")
    resolved_output = output_dir.resolve()
    if resolved_output in inputs:
        raise ValueError("output directory must not overwrite an input per-video dataset")
    loaded = [_load_and_validate(path) for path in inputs]
    manifests = [item[0] for item in loaded]
    artifact_sets = [item[1] for item in loaded]
    video_ids = [str(manifest.get("video_id")) for manifest in manifests]
    duplicates = sorted({value for value in video_ids if video_ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate video_id values: {', '.join(duplicates)}")

    dictionaries = [json.loads(items["dictionary"].read_text(encoding="utf-8")) for items in artifact_sets]
    if any(value != dictionaries[0] for value in dictionaries[1:]):
        raise ValueError("data dictionaries differ; migrate per-video exports to one schema first")

    tables = {
        "videos": _concat_tables([items["videos"] for items in artifact_sets], ["video_id"]),
        "calibrations": _concat_tables(
            [items["calibrations"] for items in artifact_sets], ["video_id", "version"]
        ),
        "frames": _concat_tables(
            [items["frames"] for items in artifact_sets], ["video_id", "frame_idx"]
        ),
        "dataset_rows": _concat_tables(
            [items["dataset_rows"] for items in artifact_sets], ["video_id", "hit_frame", "event_id"]
        ),
    }
    for table_name, table in tables.items():
        unknown_ids = sorted(set(table.get_column("video_id").to_list()) - set(video_ids))
        if unknown_ids:
            raise ValueError(f"{table_name} contains undeclared video IDs: {unknown_ids}")
        expected_rows = sum(int(manifest.get("rows", {}).get(table_name, -1)) for manifest in manifests)
        if expected_rows != table.height:
            raise ValueError(
                f"{table_name} row count differs from input manifests: {table.height} != {expected_rows}"
            )
    if tables["videos"].height != len(video_ids):
        raise ValueError("videos.parquet must contain exactly one row per input video")
    if tables["frames"].select(["video_id", "frame_idx"]).is_duplicated().any():
        raise ValueError("duplicate (video_id, frame_idx) keys in frames.parquet")
    if tables["dataset_rows"].select(["video_id", "event_id"]).is_duplicated().any():
        raise ValueError("duplicate (video_id, event_id) keys in dataset_rows.parquet")

    output_dir = resolved_output
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = {
        "videos": output_dir / "videos.parquet",
        "calibrations": output_dir / "calibrations.parquet",
        "frames": output_dir / "frames.parquet",
        "dataset_rows": output_dir / "dataset_rows.parquet",
    }
    for key, path in parquet_paths.items():
        tables[key].write_parquet(path, compression="zstd")
    csv_path = output_dir / "dataset_rows.csv"
    jsonl_path = output_dir / "dataset_rows.jsonl"
    tables["dataset_rows"].write_csv(csv_path)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in tables["dataset_rows"].iter_rows(named=True):
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    dictionary_path = output_dir / "data_dictionary.json"
    _write_json(dictionary_path, dictionaries[0])

    per_video_quality = {
        video_id: json.loads(items["quality"].read_text(encoding="utf-8"))
        for video_id, items in zip(video_ids, artifact_sets, strict=True)
    }
    warning_videos = [video_id for video_id, value in per_video_quality.items() if value.get("status") != "ok"]
    quality = {
        "schema_version": SCHEMA_VERSION,
        "status": "warning" if warning_videos else "ok",
        "scope": "batch integrity and per-video automatic coverage diagnostics; not Gold accuracy",
        "video_count": len(video_ids),
        "warning_video_ids": warning_videos,
        "rows": {key: table.height for key, table in tables.items()},
        "per_video": per_video_quality,
    }
    quality_path = output_dir / "quality.json"
    _write_json(quality_path, quality)

    output_artifacts = {
        **parquet_paths,
        "dataset_csv": csv_path,
        "dataset_jsonl": jsonl_path,
        "dictionary": dictionary_path,
        "quality": quality_path,
    }
    artifact_info = {
        key: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for key, path in output_artifacts.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": "local-visual-batch-v0.1",
        "batch_id": f"batch_{uuid.uuid4()}",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_ids": sorted(video_ids),
        "rows": {key: table.height for key, table in tables.items()},
        "quality": {"status": quality["status"], "warning_video_ids": warning_videos},
        "inputs": [
            {
                "video_id": video_id,
                "manifest": manifest.pop("_manifest_path"),
                "manifest_sha256": manifest.pop("_manifest_sha256"),
            }
            for video_id, manifest in zip(video_ids, manifests, strict=True)
        ],
        "artifacts": artifact_info,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    manifest["total_size_bytes"] = sum(value["size_bytes"] for value in artifact_info.values()) + manifest_path.stat().st_size
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, action="append", required=True,
        help="Per-video dataset dir, or a root containing <video>/dataset directories; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = discover_dataset_dirs(args.input_root)
        result = build_batch_dataset(inputs, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
