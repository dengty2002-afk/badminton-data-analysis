"""Download missing model weights and verify the pinned manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(manifest_path: Path, output_dir: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for name, spec in manifest["models"].items():
        destination = output_dir / spec["filename"]
        expected_size = int(spec["size_bytes"])
        expected_hash = str(spec["sha256"])
        if destination.is_file() and destination.stat().st_size == expected_size and sha256(destination) == expected_hash:
            results.append({"model": name, "status": "already_verified", "path": str(destination)})
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        if temporary.exists():
            temporary.unlink()
        print(f"Downloading {name}: {expected_size / 1024 / 1024:.1f} MiB")
        urllib.request.urlretrieve(spec["url"], temporary)
        if temporary.stat().st_size != expected_size or sha256(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"download verification failed for {name}")
        os.replace(temporary, destination)
        results.append({"model": name, "status": "downloaded", "path": str(destination)})
    return results


def download_bst_models(manifest_path: Path, output_dir: Path) -> list[dict[str, object]]:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required to restore BST weights") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for spec in manifest["models"]:
        destination = output_dir / spec["file"]
        expected_size = int(spec["bytes"])
        expected_hash = str(spec["sha256"])
        if destination.is_file() and destination.stat().st_size == expected_size and sha256(destination) == expected_hash:
            results.append({"model": spec["file"], "status": "already_verified", "path": str(destination)})
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        temporary.unlink(missing_ok=True)
        result = gdown.download(id=spec["google_drive_id"], output=str(temporary), quiet=False)
        if result is None or temporary.stat().st_size != expected_size or sha256(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"download verification failed for {spec['file']}")
        os.replace(temporary, destination)
        results.append({"model": spec["file"], "status": "downloaded", "path": str(destination)})
    return results


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "configs" / "models.good-badminton.json")
    parser.add_argument("--output-dir", type=Path, default=root / "models" / "good-badminton")
    parser.add_argument("--include-bst", action="store_true")
    args = parser.parse_args()
    try:
        result = download_models(args.manifest, args.output_dir)
        if args.include_bst:
            result.extend(download_bst_models(root / "configs" / "models.bst.json", root / "models" / "bst"))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
