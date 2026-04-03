from pathlib import Path
from types import SimpleNamespace
import cv2

from models.bounding_box_detection import BoundingBoxModule

# -------------------------
# SETTINGS
# -------------------------
IMAGE_PATH = r"C:\Users\65811\Downloads\BMIEstimation\BMIEstimation\filter_images\0_F_22_149860_5443109.jpg"
CONFIDENCE_THRESHOLD = 0.80
RATIO_THRESHOLD = 0.20

# -------------------------
# LOAD IMAGE
# -------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Could not read image: {IMAGE_PATH}")

image_h, image_w = img.shape[:2]
image_area = image_h * image_w

# -------------------------
# CREATE A DUMMY IMAGE OBJECT
# -------------------------
# Their code expects an object with .original_path and later writes .bounding_box
sample = SimpleNamespace(
    original_path=Path(IMAGE_PATH),
    bounding_box=None
)

# -------------------------
# RUN BOUNDING BOX DETECTION
# -------------------------
detector = BoundingBoxModule(batch_size=1)
detector.run([sample])

# -------------------------
# CHECK RESULT
# -------------------------
if sample.bounding_box is None:
    print("REJECTED")
    print("Reason: No person detected")
else:
    bbox = sample.bounding_box

    bbox_width = bbox.x2 - bbox.x1
    bbox_height = bbox.y2 - bbox.y1
    bbox_area = bbox_width * bbox_height
    bbox_ratio = bbox_area / image_area

    accepted = (
        bbox.confidence >= CONFIDENCE_THRESHOLD
        and bbox_ratio >= RATIO_THRESHOLD
    )

    print("Bounding box:")
    print(f"  x1={bbox.x1}, y1={bbox.y1}, x2={bbox.x2}, y2={bbox.y2}")
    print(f"Confidence: {bbox.confidence:.4f}")
    print(f"BBox area: {bbox_area}")
    print(f"Image area: {image_area}")
    print(f"Ratio: {bbox_ratio:.4f}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Ratio threshold: {RATIO_THRESHOLD}")
    print("Result:", "ACCEPTED" if accepted else "REJECTED")

    if bbox.confidence < CONFIDENCE_THRESHOLD:
        print("Reason: confidence below threshold")
    elif bbox_ratio < RATIO_THRESHOLD:
        print("Reason: bbox ratio below threshold")