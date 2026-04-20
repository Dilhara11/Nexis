import cv2


def _get_class_name(class_names, class_id):
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_id))

    if isinstance(class_names, (list, tuple)) and 0 <= class_id < len(class_names):
        return str(class_names[class_id])

    return str(class_id)


def annotate_detections(
    result,
    class_names,
    box_color=(0, 255, 0),
    text_color=(255, 255, 255),
    thickness=2,
    font_scale=0.6,
):
    """Draw YOLO detection boxes and labels manually on the original image."""
    annotated = result.orig_img.copy()
    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return annotated

    image_height, image_width = annotated.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        class_id = int(box.cls[0]) if box.cls is not None else -1
        class_name = _get_class_name(class_names, class_id)
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        text_x = x1
        text_y = y1 - 8

        if text_y - text_h - baseline < 0:
            text_y = min(image_height - baseline - 1, y1 + text_h + 8)

        bg_top_left = (text_x, max(0, text_y - text_h - baseline))
        bg_bottom_right = (
            min(image_width - 1, text_x + text_w + 4),
            min(image_height - 1, text_y + baseline),
        )

        cv2.rectangle(annotated, bg_top_left, bg_bottom_right, box_color, -1)
        cv2.putText(
            annotated,
            label,
            (text_x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return annotated
