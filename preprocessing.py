import cv2
import numpy as np
import mediapipe as mp
import os
import random

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "videos"
OUTPUT_PATH = "processed_data"
SEQUENCE_LENGTH = 30
AUGMENTATION_FACTOR = 5

# ============================================================================
# INITIALIZE MEDIAPIPE
# ============================================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================
def augment_landmarks(landmarks, aug_type):
    aug_landmarks = landmarks.copy()

    if aug_type == 0:
        return aug_landmarks

    elif aug_type == 1:
        angle = random.uniform(-0.15, 0.15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        for i in range(len(aug_landmarks)):
            frame = aug_landmarks[i].reshape(21, 3)
            centroid = frame.mean(axis=0)
            frame_centered = frame - centroid

            x_rot = frame_centered[:, 0] * cos_a - frame_centered[:, 1] * sin_a
            y_rot = frame_centered[:, 0] * sin_a + frame_centered[:, 1] * cos_a

            frame[:, 0] = x_rot + centroid[0]
            frame[:, 1] = y_rot + centroid[1]
            aug_landmarks[i] = frame.flatten()

    elif aug_type == 2:
        shift_x = random.uniform(-0.05, 0.05)
        shift_y = random.uniform(-0.05, 0.05)

        for i in range(len(aug_landmarks)):
            frame = aug_landmarks[i].reshape(21, 3)
            frame[:, 0] += shift_x
            frame[:, 1] += shift_y
            aug_landmarks[i] = frame.flatten()

    elif aug_type == 3:
        scale = random.uniform(0.9, 1.1)

        for i in range(len(aug_landmarks)):
            frame = aug_landmarks[i].reshape(21, 3)
            centroid = frame.mean(axis=0)
            frame_centered = frame - centroid
            frame = frame_centered * scale + centroid
            aug_landmarks[i] = frame.flatten()

    elif aug_type == 4:
        noise = np.random.normal(0, 0.01, aug_landmarks.shape)
        aug_landmarks += noise

    return aug_landmarks


# ============================================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================================
def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks_sequence.append(np.array(landmarks))

    cap.release()
    return landmarks_sequence


def pad_or_truncate_sequence(sequence, target_length=SEQUENCE_LENGTH):
    if len(sequence) == 0:
        return np.zeros((target_length, 63))

    sequence = np.array(sequence)

    if len(sequence) > target_length:
        indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
        return sequence[indices]

    elif len(sequence) < target_length:
        padding = np.repeat([sequence[-1]], target_length - len(sequence), axis=0)
        return np.vstack([sequence, padding])

    return sequence


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================
def process_dataset():
    print("=" * 70)
    print("SIGN LANGUAGE TRANSLATOR - FEATURE EXTRACTION")
    print("=" * 70)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    class_folders = sorted([f for f in os.listdir(DATASET_PATH)
                            if os.path.isdir(os.path.join(DATASET_PATH, f))])

    if len(class_folders) != 26:
        print(f"⚠️  Warning: Expected 26 classes, found {len(class_folders)}")

    print(f"\n📁 Found {len(class_folders)} classes: {class_folders}")
    print(f"🔄 Augmentation factor: {AUGMENTATION_FACTOR}x")
    print(f"📏 Sequence length: {SEQUENCE_LENGTH} frames\n")

    all_features = []
    all_labels = []

    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_path = os.path.join(DATASET_PATH, class_name)
        video_files = [f for f in os.listdir(class_path)
                       if f.endswith(('.mp4', '.avi', '.mov'))]

        print(f"📹 Processing class '{class_name}' ({len(video_files)} videos)...")

        class_label = label_map[class_name]

        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)

            landmarks_seq = extract_landmarks_from_video(video_path)

            if len(landmarks_seq) == 0:
                print(f"  ⚠️  No landmarks detected in {video_file}, skipping...")
                continue

            normalized_seq = pad_or_truncate_sequence(landmarks_seq)

            for aug_idx in range(AUGMENTATION_FACTOR):
                augmented_seq = augment_landmarks(normalized_seq, aug_idx)
                all_features.append(augmented_seq)
                all_labels.append(class_label)

        print(f"  ✅ Completed '{class_name}': {len(video_files) * AUGMENTATION_FACTOR} samples generated")

    X = np.array(all_features)
    y = np.array(all_labels)

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"📊 Total samples: {len(X)}")
    print(f"📐 Feature shape: {X.shape}")
    print(f"🏷️  Labels shape: {y.shape}")
    print(f"🎯 Number of classes: {len(np.unique(y))}")
    print(f"🔍 Unique labels found: {sorted(np.unique(y).tolist())}")

    print("\n📋 Samples per class:")
    for class_name, class_idx in sorted(label_map.items(), key=lambda x: x[1]):
        count = np.sum(y == class_idx)
        if count > 0:
            print(f"   {class_name} (label {class_idx}): {count} samples")
        else:
            print(f"   {class_name} (label {class_idx}): ⚠️  NO SAMPLES - This class was skipped!")

    np.save(os.path.join(OUTPUT_PATH, 'X_features.npy'), X)
    np.save(os.path.join(OUTPUT_PATH, 'y_labels.npy'), y)

    np.save(os.path.join(OUTPUT_PATH, 'label_map.npy'), label_map)

    print(f"\n💾 Saved features to: {OUTPUT_PATH}/X_features.npy")
    print(f"💾 Saved labels to: {OUTPUT_PATH}/y_labels.npy")
    print(f"💾 Saved label mapping to: {OUTPUT_PATH}/label_map.npy")
    print("\n✨ Feature extraction complete!\n")


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    process_dataset()
    hands.close()