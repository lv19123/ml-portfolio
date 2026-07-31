"""Face detection with OpenCV Haar Cascade and YuNet."""

from math import floor, isfinite
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np


DEFAULT_YUNET_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


def _validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with three color channels.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image width and height must be positive.")


def _validate_probability(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    normalized_value = float(value)
    if not isfinite(normalized_value):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 <= normalized_value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive.")
    return normalized_value


def _round_half_up(value: float) -> int:
    return floor(value + 0.5)


def detect_faces_haar(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> list[tuple[int, int, int, int]]:
    """Return Haar face boxes as ``(x, y, width, height)`` tuples."""
    _validate_bgr_image(image)
    if scale_factor <= 1.0:
        raise ValueError("scale_factor must be greater than 1.0.")
    if min_neighbors < 0:
        raise ValueError("min_neighbors must be non-negative.")
    if len(min_size) != 2 or min_size[0] <= 0 or min_size[1] <= 0:
        raise ValueError("min_size must contain two positive values.")

    cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade: {cascade_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    return [
        (int(x), int(y), int(width), int(height))
        for x, y, width, height in detected_faces
    ]


def detect_faces_yunet(
    image: np.ndarray,
    model_path: str | Path,
    score_threshold: float = 0.9,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> list[tuple[int, int, int, int]]:
    """Return YuNet face boxes as ``(x, y, width, height)`` tuples."""
    _validate_bgr_image(image)
    normalized_score = _validate_probability(
        score_threshold,
        "score_threshold",
    )
    normalized_nms = _validate_probability(nms_threshold, "nms_threshold")
    if not isinstance(top_k, Integral) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not isinstance(model_path, (str, Path)):
        raise TypeError("model_path must be a string or pathlib.Path.")

    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"YuNet model does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"YuNet model path is not a file: {path}")
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "This OpenCV build does not provide cv2.FaceDetectorYN."
        )

    image_height, image_width = image.shape[:2]
    input_size = (int(image_width), int(image_height))
    try:
        face_detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            input_size,
            normalized_score,
            normalized_nms,
            int(top_k),
        )
    except cv2.error as error:
        raise RuntimeError(f"Failed to load YuNet model: {path}") from error

    try:
        face_detector.setInputSize(input_size)
        _, detected_faces = face_detector.detect(image)
    except cv2.error as error:
        raise RuntimeError("YuNet face detection failed.") from error

    if detected_faces is None:
        return []

    faces_array = np.asarray(detected_faces)
    if faces_array.ndim != 2 or faces_array.shape[1] != 15:
        raise RuntimeError(
            "YuNet returned an unexpected result; expected shape "
            "[num_faces, 15]."
        )

    bounding_boxes: list[tuple[int, int, int, int]] = []
    for face in faces_array:
        coordinates = [float(value) for value in face[:4]]
        if not all(isfinite(value) for value in coordinates):
            continue

        x_value, y_value, width_value, height_value = coordinates
        if width_value <= 0 or height_value <= 0:
            continue

        x = _round_half_up(x_value)
        y = _round_half_up(y_value)
        width = _round_half_up(width_value)
        height = _round_half_up(height_value)
        if width <= 0 or height <= 0:
            continue

        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + width, image_width)
        y2 = min(y + height, image_height)
        clipped_width = x2 - x1
        clipped_height = y2 - y1
        if clipped_width <= 0 or clipped_height <= 0:
            continue

        bounding_boxes.append((x1, y1, clipped_width, clipped_height))

    return bounding_boxes


def detect_faces(
    image: np.ndarray,
    detector: str = "yunet",
    yunet_model_path: str | Path | None = None,
    score_threshold: float = 0.9,
) -> list[tuple[int, int, int, int]]:
    """Dispatch face detection to the selected OpenCV detector."""
    if detector == "haar":
        return detect_faces_haar(image)
    if detector == "yunet":
        model_path = (
            DEFAULT_YUNET_MODEL_PATH
            if yunet_model_path is None
            else yunet_model_path
        )
        return detect_faces_yunet(
            image=image,
            model_path=model_path,
            score_threshold=score_threshold,
        )
    raise ValueError(
        f"Unknown detector {detector!r}; expected 'haar' or 'yunet'."
    )
