import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --------------------
# Dataset
# --------------------
class SignVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames=16):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames
        self.samples = []

        # Collect videos and labels
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for file in os.listdir(cls_dir):
                if file.endswith((".mp4", ".mov", ".webm")):
                    self.samples.append((os.path.join(cls_dir, file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self.load_video(path, self.max_frames)
        return frames, label

    def load_video(self, path, max_frames=16):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()

        # Sample fixed number of frames
        if len(frames) > max_frames:
            idxs = np.linspace(0, len(frames)-1, max_frames).astype(int)
            frames = [frames[i] for i in idxs]
        elif len(frames) < max_frames:
            # pad by repeating last frame
            frames += [frames[-1]] * (max_frames - len(frames))

        if self.transform:
            frames = [self.transform(f) for f in frames]

        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        return frames.float()


# --------------------
# Model (ResNet3D)
# --------------------
def get_model(num_classes):
    model = models.video.r3d_18(pretrained=True)  # small 3D CNN
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --------------------
# Training loop
# --------------------
def train_model(train_loader, val_loader, model, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_loss, correct, total = 0, 0, 0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {total_loss:.3f} Acc: {train_acc:.3f}")

        # ---- Validate ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} Val Acc: {val_acc:.3f}")

    return model


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    # Paths
    train_dir = "custo_dataset/train_aug"
    val_dir = "custo_dataset/val"  # create val split manually

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    # Datasets
    train_dataset = SignVideoDataset(train_dir, transform=transform)
    val_dataset = SignVideoDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Model
    device = torch.device("cpu")  # force CPU
    model = get_model(num_classes=len(train_dataset.class_to_idx))
    model = model.to(device)

    # Train
    trained_model = train_model(train_loader, val_loader, model, device, epochs=5, lr=1e-4)

    # Save
    torch.save(trained_model.state_dict(), "sign_model.pth")
    print("✅ Training complete, model saved to sign_model.pth")
