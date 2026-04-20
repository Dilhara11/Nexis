from pathlib import Path

import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load and cache a YOLO model for reuse across Streamlit reruns."""
    resolved_path = Path(model_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")

    return YOLO(str(resolved_path))
