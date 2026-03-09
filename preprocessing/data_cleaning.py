import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")

splits = ["train", "valid", "test"]

for split in splits:

    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    for img in os.listdir(image_dir):

        img_path = os.path.join(image_dir, img)
        label_path = os.path.join(label_dir, img.replace(".png", ".txt"))

        # check label exists
        if not os.path.exists(label_path):
            print("Missing label:", img)

        # check corrupted image
        try:
            Image.open(img_path).verify()
        except:
            print("Corrupted image:", img)

print("Check completed")