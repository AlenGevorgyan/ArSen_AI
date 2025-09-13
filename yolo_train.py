from ultralytics import YOLO

# Load a pretrained YOLO11s classification model
model = YOLO("yolo11s-cls.pt")

# Train on your dataset
model.train(
    data="custo_dataset_frames",  # path to dataset root (must contain train/ and val/)
    epochs=50,                    # number of epochs
    imgsz=224,                    # image size (224 is good for classification)
    batch=32                      # batch size (adjust based on GPU)
)

# Evaluate model
metrics = model.val()
print(metrics)

# Save final model
model.export(format="torchscript")  # optional: export for deployment
