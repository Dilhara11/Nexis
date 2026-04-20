from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from postprocessing import annotate_detections
from ui.utils.image_processor import bgr_to_rgb, uploaded_file_to_bgr
from ui.utils.model_loader import load_model

st.set_page_config(page_title="Nexis - Mask Detection", page_icon=":mag:", layout="wide")

st.title("Nexis - Face Mask Detection")
st.caption("Upload one image, run detection, and view the annotated output.")

default_model_path = ROOT_DIR / "runs" / "detect" / "train6" / "weights" / "best.pt"

left_col, right_col = st.columns([1, 3], gap="large")

with left_col:
    st.subheader("Input")
    with st.container(border=True):
        st.write("Upload an image and click Run.")
        model_path = st.text_input("Model path", value=str(default_model_path))
        uploaded_file = st.file_uploader(
            "Upload or drag and drop image",
            type=["jpg", "jpeg", "png", "bmp"],
        )
        run_detection = st.button("Run Detection", type="primary", use_container_width=True)

with right_col:
    st.subheader("Output")

    if "annotated_rgb" not in st.session_state:
        st.session_state.annotated_rgb = None

    if run_detection:
        if uploaded_file is None:
            st.warning("Please upload an image before running detection.")
        else:
            with st.spinner("Loading model and running inference..."):
                model = load_model(model_path)
                image_bgr = uploaded_file_to_bgr(uploaded_file)
                result = model(image_bgr)[0]
                annotated_bgr = annotate_detections(result, model.names)
                st.session_state.annotated_rgb = bgr_to_rgb(annotated_bgr)

    if st.session_state.annotated_rgb is not None:
        st.image(
            st.session_state.annotated_rgb,
            caption="Detection output",
            use_container_width=True,
        )
    else:
        st.info("Output will appear here after you upload an image and run detection.")
