import os
import sys

from ultralytics import YOLO
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from postprocessing import annotate_detections


# load trained model
model = YOLO("runs/detect/train6/weights/best.pt")

# test image
image_path = "test.jpg"

results = model(image_path)

# show detection
for r in results:
    frame = annotate_detections(r, model.names)
    cv2.imshow("Mask Detection", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()