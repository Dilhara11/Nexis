from ultralytics import YOLO
import cv2

# load trained model
model = YOLO("runs/detect/train6/weights/best.pt")

# test image
image_path = "test.jpg"

results = model(image_path)

# show detection
for r in results:
    frame = r.plot()
    cv2.imshow("Mask Detection", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()