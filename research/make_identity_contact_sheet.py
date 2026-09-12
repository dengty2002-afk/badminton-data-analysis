from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data" / "batch_runs" / "broadcast-uncensored-v1" / "videos"
OUT = ROOT / "research_outputs" / "shi_tactical_profile_v1"
OUT.mkdir(parents=True, exist_ok=True)


def load_run(video_dir: Path) -> dict:
    return json.loads((video_dir / "run.json").read_text(encoding="utf-8"))


def first_event(video_dir: Path) -> dict | None:
    path = video_dir / "events.jsonl"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def read_frame(path: str, frame_number: int) -> Image.Image:
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return Image.new("RGB", (480, 270), "#dddddd")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image.thumbnail((480, 270))
    canvas = Image.new("RGB", (480, 270), "#eeeeee")
    canvas.paste(image, ((480 - image.width) // 2, (270 - image.height) // 2))
    return canvas


def main() -> None:
    use_start = "--start" in sys.argv
    videos = sorted(BATCH.glob("vid_*/"))
    cards: list[tuple[str, Image.Image]] = []
    for video_dir in videos:
        try:
            run = load_run(video_dir)
            event = first_event(video_dir)
            input_path = run.get("input") or run.get("inference", {}).get("input")
            frame_number = 0 if use_start else int((event or {}).get("candidate_frame", 0))
            image = read_frame(input_path, frame_number)
            draw = ImageDraw.Draw(image)
            label = f"{video_dir.name} | frame {frame_number}"
            draw.rectangle((0, 0, 480, 26), fill="#111111")
            draw.text((6, 6), label, fill="#ffffff")
            cards.append((video_dir.name, image))
        except Exception as exc:  # keep the sheet useful if one source is unavailable
            image = Image.new("RGB", (480, 270), "#ffeeee")
            ImageDraw.Draw(image).text((8, 8), f"{video_dir.name}: {exc}", fill="#990000")
            cards.append((video_dir.name, image))

    columns = 3
    rows = (len(cards) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 480, rows * 300), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(cards):
        x = (index % columns) * 480
        y = (index // columns) * 300
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + 274), name, fill="#000000")

    output = OUT / ("identity_start_contact_sheet.jpg" if use_start else "identity_contact_sheet.jpg")
    sheet.save(output, quality=92)
    print(output)


if __name__ == "__main__":
    main()
