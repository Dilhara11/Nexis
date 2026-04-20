# Streamlit UI

Minimal UI for uploading one image and viewing YOLO mask-detection output.

## Run locally

From project root:

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

## Model path

Default model path is:

`runs/detect/train6/weights/best.pt`

You can change it from the sidebar input in the app.
