import cv2
import numpy as np
from PIL import Image


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Convert a Streamlit uploaded image into a BGR ndarray for OpenCV/YOLO."""
    image = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for correct Streamlit display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
