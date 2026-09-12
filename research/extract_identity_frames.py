from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data" / "batch_runs" / "broadcast-uncensored-v1" / "videos"
OUT = ROOT / "research_outputs" / "shi_tactical_profile_v1" / "identity_frames"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: extract_identity_frames.py VID [frame ...]")
    video_id = sys.argv[1]
    video_dir = BATCH / video_id
    run = json.loads((video_dir / "run.json").read_text(encoding="utf-8"))
    input_path = run.get("input") or run.get("inference", {}).get("input")
    default_frame = 0
    with (video_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                default_frame = int(json.loads(line)["candidate_frame"])
                break
    frames = [int(x) for x in sys.argv[2:]] or [default_frame, default_frame + 2500, default_frame + 5000]
    cap = cv2.VideoCapture(input_path)
    for frame_number in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            continue
        output = OUT / f"{video_id}_frame_{frame_number:06d}.jpg"
        ok_encode, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok_encode:
            continue
        encoded.tofile(str(output))
        print(output)
    cap.release()


if __name__ == "__main__":
    main()
