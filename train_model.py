import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "/kaggle/working/processed_data2"
MODEL_SAVE_PATH = "action.h5"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("SIGN LANGUAGE TRANSLATOR - MODEL TRAINING")
print("=" * 70)

print("\n📂 Loading preprocessed data...")

X = np.load(os.path.join(DATA_PATH, 'X_features.npy'))
y = np.load(os.path.join(DATA_PATH, 'y_labels.npy'))
label_map = np.load(os.path.join(DATA_PATH, 'label_map.npy'), allow_pickle=True).item()

print(f"✅ Loaded features: {X.shape}")
print(f"✅ Loaded labels: {y.shape}")
print(f"✅ Label map classes: {len(label_map)}")
print(f"✅ Classes in label map: {sorted(label_map.keys())}")

# Check for any missing classes
expected_classes = set(label_map.keys())
print(f"\n🔍 Analyzing dataset...")
print(f"   Expected classes from label map: {sorted(expected_classes)}")

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n🔄 Preparing data...")
num_classes = len(np.unique(y))
print(f"🔍 Detected {num_classes} unique classes in the data")
print(f"📝 Label indices: {sorted(np.unique(y).tolist())}")

if len(label_map) != num_classes:
    print(f"⚠️  WARNING: Label map has {len(label_map)} classes but data has {num_classes} classes")
    print("   This is okay - using the actual number of classes from data")

y_categorical = to_categorical(y, num_classes=num_classes)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.argmax(axis=1)
)

print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Validation samples: {len(X_val)}")
print(f"📊 Test samples: {len(X_test)}")

# ============================================================================
# BUILD MODEL
# ============================================================================
print("\n🏗️  Building LSTM model...")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(32, activation='relu'),

    Dense(num_classes, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# ============================================================================
# CALLBACKS
# ============================================================================
print("\n⚙️  Setting up callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n🚀 Starting training...\n")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n📈 Test Loss: {test_loss:.4f}")
print(f"📈 Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================
print("\n📊 Generating training plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✅ Saved training plot to: training_history.png")

# ============================================================================
# CLASS-WISE ACCURACY
# ============================================================================
print("\n📊 Computing class-wise accuracy...")

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

inv_label_map = {v: k for k, v in label_map.items()}

print("\n🎯 Per-Class Accuracy:")
print("-" * 50)

for class_idx in range(num_classes):
    mask = y_test_classes == class_idx
    if mask.sum() > 0:
        class_acc = (y_pred_classes[mask] == class_idx).mean()
        class_name = inv_label_map[class_idx]
        print(f"  {class_name}: {class_acc * 100:.2f}% ({mask.sum()} samples)")

# ============================================================================
# SAVE MODEL INFO
# ============================================================================
print("\n💾 Saving model information...")

model_info = {
    'label_map': label_map,
    'num_classes': num_classes,
    'sequence_length': X.shape[1],
    'feature_dim': X.shape[2],
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss)
}

np.save('model_info.npy', model_info)
print("✅ Saved model info to: model_info.npy")

print("\n" + "=" * 70)
print("✨ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎉 Model saved to: {MODEL_SAVE_PATH}")
print(f"🎯 Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("\n✅ You can now run Part 3 for real-time inference!\n")