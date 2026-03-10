import os
import cv2
import shutil
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_PATH = os.getenv("PROCESSED_DATASET_PATH")

splits = ["train", "valid", "test"]

def preprocess_image(img):

    # noise reduction
    img = cv2.GaussianBlur(img, (5,5), 0)

    # convert to LAB for CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


for split in splits:

    img_src = os.path.join(DATASET_PATH, split, "images")
    lbl_src = os.path.join(DATASET_PATH, split, "labels")

    img_dst = os.path.join(OUTPUT_PATH, split, "images")
    lbl_dst = os.path.join(OUTPUT_PATH, split, "labels")

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    for file in os.listdir(img_src):

        img_path = os.path.join(img_src, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = preprocess_image(img)

        save_path = os.path.join(img_dst, file)

        cv2.imwrite(save_path, img)

    # copy labels
    for file in os.listdir(lbl_src):

        shutil.copy(
            os.path.join(lbl_src, file),
            os.path.join(lbl_dst, file)
        )

print("Dataset preprocessing complete")