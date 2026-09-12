"""Verify a shared ShuttleLab checkout without modifying project data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = ROOT / "configs" / "models.good-badminton.json"
BST_MANIFEST = ROOT / "configs" / "models.bst.json"
DATASET_DIR = ROOT / "data" / "datasets" / "play-state-prospective-001"
DATASET_FILES = {
    "videos": "videos.parquet",
    "calibrations": "calibrations.parquet",
    "frames": "frames.parquet",
    "dataset_rows": "dataset_rows.parquet",
    "dataset_csv": "dataset_rows.csv",
    "dataset_jsonl": "dataset_rows.jsonl",
    "dictionary": "data_dictionary.json",
    "quality": "quality.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(full: bool = False) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        ROOT / "badminton_pipeline" / "run_pipeline.py",
        ROOT / "badminton_pipeline" / "exports" / "batch_dataset.py",
        ROOT / "badminton_pipeline" / "calibration" / "auto_court.py",
        ROOT / "configs" / "pipeline.local.json",
        MODEL_MANIFEST,
        BST_MANIFEST,
        ROOT / "vendor" / "BST-Badminton-Stroke-type-Transformer" / "LICENSE",
        ROOT / "vendor" / "Good-Badminton" / "LICENSE",
        ROOT / "vendor" / "Good-Badminton" / "badminton_analysis" / "court" / "detector.py",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    model_results: dict[str, str] = {}
    if MODEL_MANIFEST.is_file():
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
        for name, spec in manifest["models"].items():
            path = ROOT / "models" / "good-badminton" / spec["filename"]
            if not path.is_file():
                errors.append(f"missing model: {path.relative_to(ROOT)}")
                continue
            if path.stat().st_size != int(spec["size_bytes"]) or sha256(path) != spec["sha256"]:
                errors.append(f"model verification failed: {name}")
            else:
                model_results[name] = "verified"
    if BST_MANIFEST.is_file():
        manifest = json.loads(BST_MANIFEST.read_text(encoding="utf-8"))
        for spec in manifest["models"]:
            path = ROOT / "models" / "bst" / spec["file"]
            if not path.is_file():
                errors.append(f"missing model: {path.relative_to(ROOT)}")
                continue
            if path.stat().st_size != int(spec["bytes"]) or sha256(path) != spec["sha256"]:
                errors.append(f"model verification failed: {spec['file']}")
            else:
                model_results[spec["file"]] = "verified"

    dataset_status = "not_in_bundle"
    manifest_path = DATASET_DIR / "manifest.json"
    if manifest_path.is_file():
        dataset_status = "verified"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, filename in DATASET_FILES.items():
            path = DATASET_DIR / filename
            expected = manifest.get("artifacts", {}).get(key, {}).get("sha256")
            if not path.is_file() or (expected and sha256(path) != expected):
                dataset_status = "failed"
                errors.append(f"dataset artifact verification failed: {filename}")

    dependencies = {}
    for module in ("numpy", "cv2", "polars", "onnxruntime", "rtmlib", "torch", "ultralytics", "positional_encodings", "torchinfo", "gdown"):
        available = importlib.util.find_spec(module) is not None
        dependencies[module] = available
        if full and not available:
            errors.append(f"runtime dependency unavailable: {module}")
        elif not available:
            warnings.append(f"runtime dependency unavailable before setup: {module}")

    tests = "not_run"
    if full and not errors:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=ROOT,
            check=False,
        )
        tests = "passed" if completed.returncode == 0 else "failed"
        if completed.returncode:
            errors.append("unit tests failed")
    return {
        "status": "ok" if not errors else "failed",
        "python": sys.version.split()[0],
        "models": model_results,
        "dataset": dataset_status,
        "dependencies": dependencies,
        "tests": tests,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Require runtime dependencies and run all unit tests")
    args = parser.parse_args()
    result = verify(args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
