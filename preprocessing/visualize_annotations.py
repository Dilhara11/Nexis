import os
import cv2
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")

split = "train"   # change to valid/test if needed

image_dir = os.path.join(DATASET_PATH, split, "images")
label_dir = os.path.join(DATASET_PATH, split, "labels")

for img_name in os.listdir(image_dir):

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".png", ".txt"))

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        class_id, x, y, bw, bh = map(float, line.split())

        # convert YOLO format → pixel coordinates
        x_center = int(x * w)
        y_center = int(y * h)
        box_w = int(bw * w)
        box_h = int(bh * h)

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, str(int(class_id)), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Annotation Check", image)

    if cv2.waitKey(0) == 27:   # press ESC to exit
        break

cv2.destroyAllWindows()