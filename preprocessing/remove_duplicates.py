import os
import hashlib
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")

splits = ["train", "valid", "test"]

for split in splits:

    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    hashes = {}

    for img in os.listdir(image_dir):

        img_path = os.path.join(image_dir, img)

        with open(img_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes:

            print("Duplicate found:", img)

            # remove duplicate image
            os.remove(img_path)

            # remove corresponding label
            label_path = os.path.join(label_dir, img.replace(".jpg", ".txt"))
            if os.path.exists(label_path):
                os.remove(label_path)

        else:
            hashes[file_hash] = img

print("Duplicate check completed")