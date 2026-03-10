import os
from ultralytics import YOLO
from dotenv import load_dotenv

def evaluate():
    # load env variables
    load_dotenv()

    DATA_YAML = os.getenv("YML_PATH")

    # load trained model
    model = YOLO("runs/detect/train6/weights/best.pt")

    # evaluate using test dataset
    metrics = model.val(
        data=DATA_YAML,
        split="test"
    )

    print(metrics)

if __name__ == '__main__':
    evaluate()