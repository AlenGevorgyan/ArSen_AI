import cv2
import os
from pathlib import Path

def extract_frames(video_path, out_dir, fps=2):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(video_fps / fps)))
    idx, saved = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = out_dir / f"{video_path.stem}_frame{saved:05d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"{saved} frames saved from {video_path} → {out_dir}")

# Paths
root = Path("custo_dataset")
out_root = Path("custo_dataset_frames")

splits = ["train_aug"]
for split in splits:
    split_dir = root / split
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            for video_file in class_dir.glob("*.mp4"):
                out_dir = out_root / split.replace("train_aug", "train") / class_dir.name
                extract_frames(video_file, out_dir, fps=2)  # adjust fps
