import os
from ultralytics import YOLO
from dotenv import load_dotenv

def train():
    # Load .env variables
    load_dotenv()
    DATA_YAML = os.getenv("DATA_PROCESSED_YAML")

    print(f"Data Path: {DATA_YAML}")
    print("Exists?", os.path.exists(DATA_YAML))

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

    # Load YOLOv8 small model
    model = YOLO("yolov8n.pt")

    # Train on your dataset
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=8,       # Safe for 4GB VRAM
        device=0,      # Your RTX 3050
        workers=4      # Number of CPU cores for data loading
    )

# THIS IS THE CRITICAL PART FOR WINDOWS
if __name__ == '__main__':
    train()