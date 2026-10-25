"""
YOLO11n Custom Training Script
Dataset: Indian/South Asian Traffic Dataset (21 vehicle classes)
Device: CPU (no CUDA required)

Run this script in VS Code terminal:
    cd C:\\Users\\msafw\\vehicle-ai
    .venv\\Scripts\\python.exe train_custom_model.py

Training will take 2-5 hours on CPU.
Progress is saved automatically — if interrupted, training resumes from last checkpoint.
"""

import os
from ultralytics import YOLO

# =============================================
# CONFIGURATION — modify if needed
# =============================================
PRETRAINED_MODEL = "models/yolo11n.pt"      # Base model to fine-tune from
DATASET_YAML     = "models/indian_traffic.yaml"  # Dataset configuration
EPOCHS           = 30                        # Number of training epochs (increase for better accuracy)
IMAGE_SIZE       = 416                       # Input image size (smaller = faster on CPU)
BATCH_SIZE       = 8                         # Batch size (reduce to 4 if RAM is low)
OUTPUT_NAME      = "indian_traffic_yolo11n"  # Name for saved trained model run
DEVICE           = "cpu"                     # Force CPU training
# =============================================

def train():
    print("=" * 60)
    print("  AI Vehicle Detection - Custom YOLO Training")
    print("  Dataset: Indian Traffic (21 vehicle classes)")
    print(f"  Epochs: {EPOCHS} | Image Size: {IMAGE_SIZE} | Batch: {BATCH_SIZE}")
    print("=" * 60)

    # Load pretrained YOLO11n as the base model
    model = YOLO(PRETRAINED_MODEL)

    print("\n[INFO] Starting training... This will take 2-5 hours on CPU.")
    print("[INFO] Training progress is shown below. Do NOT close this terminal.\n")

    # Begin fine-tuning on the Indian traffic dataset
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name=OUTPUT_NAME,
        project="models/runs",          # Save trained models here
        pretrained=True,                 # Use pretrained weights as starting point
        optimizer="SGD",                 # SGD optimizer works well on CPU
        lr0=0.01,                        # Initial learning rate
        patience=10,                     # Stop early if no improvement for 10 epochs
        cache=False,                     # Don't cache images (saves RAM)
        workers=2,                       # Number of data loading workers
        exist_ok=True,                   # Overwrite if run name exists
        verbose=True,
    )

    # Save best trained model to models/ folder for project use
    best_model_path = f"models/runs/{OUTPUT_NAME}/weights/best.pt"
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, "models/indian_traffic_best.pt")
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE!")
        print(f"  Best model saved: models/indian_traffic_best.pt")
        print("  To use this model in the project:")
        print("  Change model_path in video_routes.py to:")
        print("  'models/indian_traffic_best.pt'")
        print("=" * 60)
    else:
        print("[WARNING] best.pt not found. Check runs/train folder.")

if __name__ == "__main__":
    train()
