import os
import cv2
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image

# ---- Video augmentation class ----
class VideoAugment:
    def __init__(self):
        self.spatial = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def temporal_jitter(self, frames):
        new_frames = []
        for f in frames:
            if random.random() < 0.05:  # drop frame
                continue
            new_frames.append(f)
            if random.random() < 0.05:  # duplicate
                new_frames.append(f)
        return new_frames if new_frames else frames

    def speed_change(self, frames, factor=None):
        if factor is None:
            factor = random.choice([0.8, 1.0, 1.2])
        idxs = np.linspace(0, len(frames)-1, int(len(frames)*factor)).astype(int)
        return [frames[i] for i in idxs]

    def __call__(self, frames):
        frames = self.temporal_jitter(frames)
        frames = self.speed_change(frames)
        frames = [self.spatial(Image.fromarray(f)) for f in frames]
        return [np.array(f) for f in frames]


# ---- Helper functions ----
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # fallback
        fps = 25
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, fps


def save_video(frames, path, fps=25):
    if not frames:
        print(f"⚠️ No frames to save for {path}")
        return
    h, w, _ = frames[0].shape

    # Ensure extension is mp4 for compatibility
    if not path.endswith(".mp4"):
        path = os.path.splitext(path)[0] + ".mp4"

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


# ---- Main augmentation loop ----
def augment_dataset(input_dir="custo_dataset/train", output_dir="custo_dataset/train_aug", n_aug=3):
    augmenter = VideoAugment()

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not (file.endswith(".mp4") or file.endswith(".webm") or file.endswith(".mov")):
                continue
            in_path = os.path.join(root, file)

            # Mirror folder structure in output
            rel_path = os.path.relpath(root, input_dir)
            out_folder = os.path.join(output_dir, rel_path)
            os.makedirs(out_folder, exist_ok=True)

            # Load video
            frames, fps = load_video(in_path)
            if not frames:
                print(f"⚠️ Skipping {in_path}, no frames read")
                continue

            # Save original (converted to mp4 for consistency)
            save_video(frames, os.path.join(out_folder, os.path.splitext(file)[0] + ".mp4"), fps)

            # Augmented versions
            for i in range(n_aug):
                aug_frames = augmenter(frames)
                out_path = os.path.join(out_folder, f"{os.path.splitext(file)[0]}_aug{i+1}.mp4")
                save_video(aug_frames, out_path, fps)
                print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    augment_dataset("custom_dataset/train", "custo_dataset/train_aug", n_aug=3)
